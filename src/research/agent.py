"""Furniture Research agent: finds suitable furniture, first in the local furniture database
(LanceDB, already downloaded and described), then - only if that is not enough - through the asset
server's MCP tools. It looks at preview images and shortlists candidates.

The agent decides its own research steps in a tool loop (search, inspect, look, refine). It never
downloads anything: that happens in asset_download, and only for external assets the user approved.
Local finds need no approval, since nothing is downloaded.
"""
import json
import time
from contextlib import AsyncExitStack

import inventory
from design.routing import LIMITS, caller
from design.state import DesignState, model_for
from furniture.store import CATEGORIES, category_of, compact, get_store
from llm import LLMError, supports_image_input, tool_completion
from observability import progress
from research.mcp_client import AssetMCP, AssetMCPError
from research.previews import PreviewError, fetch_preview

MAX_RESEARCH_ROUNDS = LIMITS["research_rounds"]  # research + approval rounds per request
MAX_TURNS = 10           # LLM calls per round
MAX_PREVIEWS = 6
MAX_CANDIDATES = 5
LOADABLE_FORMATS = {"glb", "gltf"}
LOCAL_MIN_SCORE = 0.25   # cosine similarity below this is not shown as a local match

SYSTEM_PROMPT = f"""You are the Furniture Research agent of RoomCraft, a 3D interior design app. You find 3D furniture models that match a request: first in the user's own furniture database (already downloaded, ready to use), then, only if needed, as free downloadable models from an external 3D asset search. You recommend; the user decides what is downloaded. You never place furniture or design layouts.

How to work:
- Work out what is needed: type of furniture, style, colours, materials and size limits. The room data helps to judge what fits.
- Local first: the first message already contains matches from the furniture database. Refine with search_local_furniture (category and size filters, other wording) when useful. Local items need no download.
- Search externally (search_assets) only if the local items do not satisfy the request: none match, too few to choose from, they violate important constraints (type, style, size, colour), or the user explicitly wants new furniture. Say why in local_assessment.
- Call search_assets with short, concrete English queries (2-4 words). If the results are poor, refine the query and search again (synonyms, broader or narrower terms).
- Results are already limited to free, directly downloadable 3D models; models without GLB/glTF are removed because the app cannot load them.
- Call get_asset for promising candidates to read description, tags, formats and licence.
- Call view_preview to look at a candidate when its appearance matters or the metadata is ambiguous. Check that it really is the requested kind of furniture (not a whole scene, a pack, a texture or something unrelated), and note its style, colours and materials. Images can mislead, so treat what you see as uncertain and never derive dimensions from an image. A missing preview is no reason to reject an otherwise good asset.
- Never invent properties. Mention dimensions only when the metadata states them.
- Prefer variety: do not propose near-identical assets unless the user asks for alternatives of one style.
- You have at most {MAX_TURNS} steps and {MAX_PREVIEWS} previews per round, so work purposefully.
- Finish by calling submit_candidates with up to {MAX_CANDIDATES} candidates in total: suitable local items (by item_id) and/or external assets (by asset id), an empty list if nothing suitable exists. Do not propose an external asset when a local item already fits as well. Write the reasons and the summary in the language of the user's request."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_local_furniture",
        "description": "Semantic search in the user's furniture database (downloaded and described models). "
                       "Returns item_id, name, category, style, colours, materials, size in metres and a score (0-1).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Natural-language description, e.g. 'minimalist wooden chair'"},
            "category": {"type": "string", "enum": list(CATEGORIES), "description": "Optional hard filter"},
            "max_width": {"type": "number", "description": "Optional, metres"},
            "max_depth": {"type": "number", "description": "Optional, metres"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 15},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_assets",
        "description": "Search free, directly downloadable 3D models. Returns compact results.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Short query, e.g. 'oak coffee table'"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max results (default 10)"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_asset",
        "description": "Details of one asset: description, tags, formats, files, licence.",
        "parameters": {"type": "object", "properties": {
            "asset_id": {"type": "string", "description": "Id from search results, '<provider>:<id>'"},
        }, "required": ["asset_id"]}}},
    {"type": "function", "function": {
        "name": "view_preview",
        "description": "Look at the preview image of an asset (attached to the next message).",
        "parameters": {"type": "object", "properties": {
            "asset_id": {"type": "string"},
        }, "required": ["asset_id"]}}},
    {"type": "function", "function": {
        "name": "submit_candidates",
        "description": "Finish the research with the shortlist for the user.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "1-2 sentences on what you looked for and found"},
            "local_assessment": {"type": "string",
                                 "description": "Why the local items were enough, or why external search was needed"},
            "candidates": {"type": "array", "maxItems": MAX_CANDIDATES, "items": {"type": "object", "properties": {
                "asset_id": {"type": "string", "description": "item_id of a local item or id of an external asset"},
                "reason": {"type": "string", "description": "Why it fits the request (1-2 sentences)"},
                "visual_check": {"type": "string", "description": "What the preview showed, or 'not inspected'"},
            }, "required": ["asset_id", "reason"]}},
        }, "required": ["summary", "local_assessment", "candidates"]}}},
]
SUBMIT_CHOICE = {"type": "function", "function": {"name": "submit_candidates"}}


def _license(value) -> dict | None:
    if not value:
        return None
    return value if isinstance(value, dict) else {"name": str(value)}


def _formats(asset: dict) -> list[str]:
    return [f.lower() for f in asset.get("formats") or []]


def _loadable(asset: dict) -> bool:
    formats = set(_formats(asset))
    files = (asset.get("selection") or {}).get("files") or []
    formats |= {str(f.get("format", "")).lower() for f in files}
    return not formats or bool(formats & LOADABLE_FORMATS)


def inventory_asset_ids() -> dict[str, str]:
    """Source asset id -> inventory item id, to recognise external results that are already downloaded."""
    return {item["assetId"]: item["id"] for item in inventory.load_index().values()}


class Research:
    """One research round: the tool implementations plus everything seen so far. The MCP session is
    only opened when the agent searches externally."""

    def __init__(self, stack: AsyncExitStack, excluded: set[str], vision: bool):
        self.stack = stack
        self._mcp: AssetMCP | None = None
        self.excluded = excluded
        self.vision = vision
        self.seen: dict[str, dict] = {}       # external asset id -> metadata from search / get_asset
        self.local: dict[str, dict] = {}      # inventory item id -> local search hit
        self.in_inventory = inventory_asset_ids()
        self.previewed: dict[str, str] = {}   # asset id -> "ok" or why not
        self.stats = {"local_searches": 0, "searches": 0, "details": 0, "previews": 0, "found": 0}

    async def mcp(self) -> AssetMCP:
        if self._mcp is None:
            self._mcp = await self.stack.enter_async_context(AssetMCP())
        return self._mcp

    async def search_local_furniture(self, query: str, category: str | None = None, max_width: float | None = None,
                                     max_depth: float | None = None, limit: int = 8) -> tuple[str, None]:
        self.stats["local_searches"] += 1
        try:
            hits = await get_store().search(query, category=category_of(category), max_width=max_width,
                                            max_depth=max_depth, limit=max(1, min(int(limit or 8), 15)),
                                            min_score=LOCAL_MIN_SCORE)
        except Exception as exc:  # embedding or database problems: the external search still works
            return json.dumps({"query": query, "error": f"Local search unavailable ({type(exc).__name__})."}), None
        results = []
        for hit in hits:
            row = compact(hit)
            row["item_id"] = row.pop("id")
            row.pop("model_url", None)
            self.local[row["item_id"]] = hit
            results.append(row)
        return json.dumps({"query": query, "category": category_of(category), "results": results}), None

    async def search_assets(self, query: str, limit: int = 10) -> tuple[str, None]:
        limit = max(1, min(int(limit or 10), 20))
        data = await (await self.mcp()).search(query, limit=limit)
        self.stats["searches"] += 1
        results, filtered = [], 0
        for asset in data.get("results", []):
            if (asset.get("type") not in (None, "model") or asset["id"] in self.excluded
                    or asset.get("downloadable") is False or not _loadable(asset)):
                filtered += 1
                continue
            self.seen.setdefault(asset["id"], {}).update(asset)
            results.append({
                "id": asset["id"], "title": asset.get("title"), "source": asset.get("source"),
                "formats": _formats(asset), "license": asset.get("license"), "tags": (asset.get("tags") or [])[:8],
                "polyCount": asset.get("polyCount"), "hasPreview": bool(asset.get("thumbnail")),
                **({"alreadyInInventory": self.in_inventory[asset["id"]]} if asset["id"] in self.in_inventory else {}),
            })
        self.stats["found"] += len(results)
        payload = {"query": query, "results": results, "filteredOut": filtered}
        return json.dumps(payload), None

    async def get_asset(self, asset_id: str) -> tuple[str, None]:
        details = await (await self.mcp()).get_asset(asset_id)
        self.stats["details"] += 1
        self.seen.setdefault(asset_id, {}).update(details)
        files = (details.get("selection") or {}).get("files") or []
        payload = {
            "id": asset_id, "title": details.get("title"), "description": (details.get("description") or "")[:600],
            "source": details.get("source"), "author": details.get("author"), "license": details.get("license"),
            "tags": details.get("tags"), "formats": _formats(details), "polyCount": details.get("polyCount"),
            "dimensions": details.get("dimensions"),  # only if the source provides them
            "files": [{"filename": f.get("filename"), "format": f.get("format"), "size": f.get("size")} for f in files][:6],
            "loadableInApp": _loadable(details), "excluded": asset_id in self.excluded,
        }
        return json.dumps(payload), None

    async def view_preview(self, asset_id: str) -> tuple[str, str | None]:
        if not self.vision:
            self.previewed[asset_id] = "model without image input"
            return "The selected model cannot see images. Judge this asset from its metadata.", None
        if len(self.previewed) >= MAX_PREVIEWS:
            return f"Preview limit of {MAX_PREVIEWS} reached for this round.", None
        if asset_id in self.local:
            return "This is a local item; its description already comes from an image analysis.", None
        thumbnail = (self.seen.get(asset_id) or {}).get("thumbnail")
        if not thumbnail:
            self.previewed[asset_id] = "no preview available"
            return "This asset has no preview image. Do not reject it for that reason alone.", None
        try:
            image = await fetch_preview(thumbnail)
        except PreviewError as exc:
            self.previewed[asset_id] = str(exc)
            return f"The preview could not be loaded ({exc}). Judge from metadata.", None
        self.stats["previews"] += 1
        self.previewed[asset_id] = "ok"
        return "Preview attached in the next message.", image

    def _local_candidate(self, item_id: str, entry: dict) -> dict:
        hit = self.local.get(item_id) or {}
        return {"itemId": item_id, "name": hit.get("name", item_id), "category": hit.get("category"),
                "description": hit.get("description"), "thumbnailUrl": hit.get("thumbnail_url"),
                "score": hit.get("score"), "reason": entry.get("reason", ""), "source": "local"}

    def candidates(self, submitted: list[dict]) -> tuple[list[dict], list[dict]]:
        """Validated shortlist split into (external candidates for approval, local items). Unknown,
        excluded or duplicate ids are dropped: only items the agent actually found can be proposed.
        External assets that are already in the inventory count as local."""
        shortlist, local, used = [], [], set()
        for entry in submitted or []:
            asset_id = entry.get("asset_id")
            item_id = asset_id if asset_id in self.local else self.in_inventory.get(asset_id)
            if item_id and item_id not in used:
                used.add(item_id)
                local.append(self._local_candidate(item_id, entry))
                continue
            asset = self.seen.get(asset_id)
            if not asset or asset_id in self.excluded or asset_id in used or not _loadable(asset):
                continue
            if len(shortlist) + len(local) >= MAX_CANDIDATES:
                break
            used.add(asset_id)
            shortlist.append({
                "assetId": asset_id,
                "name": asset.get("title") or asset_id,
                "description": (asset.get("description") or "")[:400] or None,
                "provider": asset.get("source"),
                "author": asset.get("author"),
                "formats": _formats(asset),
                "license": _license(asset.get("license")),
                "thumbnailUrl": asset.get("thumbnail"),
                "sourceUrl": asset.get("url"),
                "tags": (asset.get("tags") or [])[:8],
                "dimensions": asset.get("dimensions"),
                "reason": entry.get("reason", ""),
                "visualCheck": entry.get("visual_check") or None,
                "previewInspected": self.previewed.get(asset_id) == "ok",
                "source": "external",
            })
        return shortlist, local


def build_messages(state: DesignState, research_round: int, vision: bool, note: str | None = None) -> list[dict]:
    """`note` explains why another round is needed (user feedback or failed downloads)."""
    room = state["room"]
    request = state.get("furniture_request")
    lines = [f"User request: {state['request']}"]
    if request:
        lines.append(f"Furniture needed by the interior designer: {json.dumps(request)}")
    lines.append(f"Room: {room.width} × {room.depth} m, height {room.height} m.")
    if state.get("furniture"):
        lines.append("Furniture already in the room: " + ", ".join(
            f"{f.name} ({f.width:.2f}×{f.depth:.2f} m)" if f.width and f.depth else f.name for f in state["furniture"]))
    if not vision:
        lines.append("Note: your model cannot see images; rely on metadata.")
    if note:
        lines.append(f"This is research round {research_round} of {MAX_RESEARCH_ROUNDS}. {note} "
                     "Those assets are excluded now; find different candidates.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


async def _run_tool(research: Research, name: str, arguments: dict) -> tuple[str, str | None]:
    started = time.perf_counter()
    data = {"tool": name, "input": arguments}
    try:
        if name == "search_local_furniture":
            text, image = await research.search_local_furniture(
                arguments.get("query", ""), arguments.get("category"), arguments.get("max_width"),
                arguments.get("max_depth"), arguments.get("limit", 8))
            parsed = json.loads(text)
            data["resultCount"] = len(parsed.get("results", []))
            data["scope"] = "local"
            if parsed.get("error"):
                data["error"] = parsed["error"]
        elif name == "search_assets":
            text, image = await research.search_assets(arguments.get("query", ""), arguments.get("limit", 10))
            data["resultCount"] = len(json.loads(text)["results"])
        elif name == "get_asset":
            text, image = await research.get_asset(arguments.get("asset_id", ""))
        elif name == "view_preview":
            text, image = await research.view_preview(arguments.get("asset_id", ""))
            data["imageAttached"] = image is not None
            data["note"] = research.previewed.get(arguments.get("asset_id", ""))
        else:
            text, image = f"Unknown tool {name}.", None
        data["status"] = "completed"
    except AssetMCPError as exc:
        text, image = f"Tool error: {exc}", None
        data.update(status="failed", error=str(exc))
    data["durationMs"] = round((time.perf_counter() - started) * 1000)
    progress.emit("tool", **data)
    return text, image


def _finish(state: DesignState, status: str, summary: str, downloads: dict | None = None,
            local: list[dict] | None = None, local_assessment: str | None = None) -> dict:
    """Ends the research and hands the result to the caller: the designer, or the user (END).
    Consumed approval/download data is cleared so a later research request starts clean."""
    downloads = downloads or {}
    local = (state.get("local_candidates") or []) if local is None else local
    result = {"status": status, "summary": summary, "added": downloads.get("added", []),
              "existing": downloads.get("existing", []), "failed": downloads.get("failed", []),
              "local": local, "local_assessment": local_assessment or state.get("local_assessment")}
    if local and status != "found_locally":
        summary = f"{summary} Already in your inventory: {', '.join(c['name'] for c in local)}."
        result["summary"] = summary
    requester = caller(state, "furniture_research")
    progress.step("finish", f"Research {status}; returning to {requester}", **{k: len(v) for k, v in result.items()
                                                                              if isinstance(v, list)})
    update = {"research_status": "done", "research_result": result, "approval": None, "download_result": None,
              "workflow_log": [f"furniture_research: {status} -> {requester}"]}
    if requester == "end":
        update["outcome"] = {"status": "failed" if status == "download_failed" else "completed",
                             "kind": "research", "message": summary}
    return update


def _download_summary(downloads: dict) -> str:
    parts = []
    if downloads.get("added"):
        parts.append(f"Added to the inventory: {', '.join(a['name'] for a in downloads['added'])}.")
    if downloads.get("existing"):
        parts.append(f"Already in the inventory: {', '.join(e['name'] for e in downloads['existing'])}.")
    if downloads.get("failed"):
        parts.append("Failed: " + "; ".join(f"{f['name']} ({f['error']})" for f in downloads["failed"]) + ".")
    return " ".join(parts) or "Nothing was downloaded."


async def furniture_research(state: DesignState) -> dict:
    """Entry point for all research phases: new round, evaluate downloads, or finish."""
    approval = state.get("approval")
    downloads = state.get("download_result")
    if downloads is not None:  # back from asset_download
        if downloads.get("added") or downloads.get("existing") or not downloads.get("failed"):
            return _finish(state, "completed", _download_summary(downloads), downloads)
        # Every approved asset failed: look for alternatives if a round is left.
        note = ("The approved assets could not be downloaded: "
                + "; ".join(f"{f['name']} ({f['error']})" for f in downloads["failed"]) + ".")
        if state.get("research_round", 0) >= MAX_RESEARCH_ROUNDS:
            return _finish(state, "download_failed", _download_summary(downloads), downloads)
        return await _research_round(state, note, exclude=[f["assetId"] for f in downloads["failed"]])
    if approval and approval["action"] == "reject":
        return _finish(state, "rejected", "All suggestions were rejected; nothing was downloaded.")
    if approval and approval["action"] == "retry":
        names = ", ".join(c["name"] for c in state.get("candidates") or [])
        feedback = f" User feedback: {approval['feedback']}" if approval.get("feedback") else ""
        return await _research_round(state, f"The user asked for other suggestions than {names}.{feedback}")
    return await _research_round(state)


async def _research_round(state: DesignState, note: str | None = None, exclude: list[str] | None = None) -> dict:
    research_round = state.get("research_round", 0) + 1
    if research_round > MAX_RESEARCH_ROUNDS:
        progress.step("limit", "Research round limit reached", rounds=MAX_RESEARCH_ROUNDS)
        return {**_finish(state, "limit_reached",
                          f"No further suggestions: the limit of {MAX_RESEARCH_ROUNDS} research rounds was reached."),
                "research_round": research_round}

    model = model_for(state, "furniture_research")
    vision = await supports_image_input(model) is not False
    excluded = set(state.get("rejected_asset_ids") or []) | set(exclude or [])
    with progress.stage("prepare", "Understand the request") as details:
        messages = build_messages(state, research_round, vision, note)
        details.update(round=research_round, excluded=len(excluded), vision=vision,
                       requester=caller(state, "furniture_research"))

    submitted = None
    async with AsyncExitStack() as stack:
        research = Research(stack, excluded, vision)
        # Local first: the furniture database is searched before the agent decides anything.
        query = (state.get("furniture_request") or {}).get("description") or state["request"]
        local_text, _ = await _run_tool(research, "search_local_furniture", {"query": query, "limit": 8})
        messages[1]["content"] += f"\n\nMatches in the user's furniture database (local, no download needed):\n{local_text}"
        for turn in range(MAX_TURNS):
            last_turn = turn == MAX_TURNS - 1
            try:
                message, info = await tool_completion(model, messages, TOOLS, SUBMIT_CHOICE if last_turn else "auto")
            except LLMError as exc:
                if exc.info:
                    progress.emit("llm", **exc.info.as_dict())
                raise
            progress.emit("llm", **info.as_dict())
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                messages.append({"role": "user", "content": "Use the tools, and finish with submit_candidates."})
                continue
            images = []
            for call in calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = None
                if name == "submit_candidates" and isinstance(arguments, dict):
                    submitted = arguments
                    text = "Submitted."
                elif arguments is None:
                    text = "Invalid JSON arguments."
                else:
                    text, image = await _run_tool(research, name, arguments)
                    if image:
                        images.append((arguments.get("asset_id"), image))
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            if submitted is not None:
                break
            if images:  # tool results are text only, so previews follow as a user message with image parts
                content = [{"type": "text", "text": "Requested preview images:"}]
                for asset_id, image in images:
                    content += [{"type": "text", "text": f"Preview of {asset_id}:"},
                                {"type": "image_url", "image_url": {"url": image}}]
                messages.append({"role": "user", "content": content})

    candidates, local = research.candidates((submitted or {}).get("candidates", []))
    summary = (submitted or {}).get("summary") or ""
    assessment = (submitted or {}).get("local_assessment") or ""
    progress.step("shortlist", f"{len(local)} local, {len(candidates)} external candidate(s)", **research.stats,
                  local=[c["itemId"] for c in local], candidates=[c["assetId"] for c in candidates],
                  localAssessment=assessment)
    base = {"research_round": research_round, "research_summary": summary, "counters": {"research_rounds": 1},
            "rejected_asset_ids": sorted(excluded), "local_candidates": local, "local_assessment": assessment}
    if not candidates:
        if local:  # everything needed is already downloaded: no approval, no download
            return {**base, **_finish(state, "found_locally", summary, local=local, local_assessment=assessment),
                    "candidates": []}
        return {**base, **_finish(state, "no_results",
                                  summary or "No suitable, directly downloadable 3D models were found for this request.",
                                  local=[], local_assessment=assessment),
                "candidates": []}
    return {**base, "candidates": candidates, "research_status": "awaiting_approval", "approval": None,
            "download_result": None,
            "workflow_log": [f"furniture_research: round {research_round}, {len(local)} local, "
                             f"{len(candidates)} external candidate(s) for approval"]}
