"""Furniture Research: works through a furniture LIST (one or more missing pieces) item by item.

The list is processed in code, not in one big LLM loop. Every item has an explicit, persistent
state in `research_items` and moves through:

    pending -> searching_local -> (found_locally | pending)            direct user search only
    pending -> searching_online (run 1) -> awaiting_approval | downloading | retrying | not_found
    retrying -> searching_online (run 2) -> awaiting_approval | downloading | not_found
    awaiting_approval -> downloading (approve) | retrying (reject in run 1) | rejected | skipped
    downloading -> completed | failed

Final states: found_locally, completed, failed, skipped, rejected, not_found.

Local search only for direct user searches (orchestrator, SEARCH_FURNITURE): the interior designer
sees the inventory and only asks for what is missing there (`searches_locally`), so its items go
online right away (EXTERNAL_ONLY_PROMPT, EXTERNAL_TOOLS). External results that are already in the
inventory (`alreadyInInventory`) count as local finds either way: no download, no approval.

Search mode (design.routing.research_mode, from the user's options):
- local_only: no online search at all; items without a local match are reported as not found.
- guided: up to MAX_SUGGESTIONS_PER_ITEM suggestions per item; the user approves, rejects (second
  run with other terms) or skips one item at a time (human_approval).
- automatic: the best suitable candidate is downloaded without asking - only when that option is
  set, and only candidates that pass the deterministic check in `auto_acceptable`.

This node is the coordinator. Per-item searches and downloads run as background jobs
(research.jobs) with bounded parallelism, so the user can decide on one item while others are
still being searched or downloaded. The coordinator merges finished jobs, starts the next ones,
and then either pauses for the next item that is ready for approval, hands queued downloads to
asset_download, waits for running jobs, or returns the finished list to its caller.

Technical limits per item (constants below):
- MAX_SUGGESTIONS_PER_ITEM (4): suggestions shown to the user per item and online run. In
  automatic mode the best suitable candidate is enough; the agent ranks the rest only as needed.
- MAX_IMAGE_CHECKS_PER_ITEM (6): preview images the agent may look at per item and online run.
- MAX_LLM_STEPS_PER_ITEM (3): LLM calls of one online run of one item (the last one must submit).
- MAX_ONLINE_RUNS_PER_ITEM (2): online runs per item. The second run follows a rejection (or a
  first run without a suitable result), excludes what was shown before, uses the user's feedback
  and other search terms. A rejection in the second run ends the item; there is no third run.
- MAX_ITEMS_PER_REQUEST (6, design.routing): items per list; the rest is cut off and reported in the
  workflow log.
The user's limit is counted here for the designer's lists: research_requests (lists with at least
one online search), see design.routing. With at most six items per list it also bounds the pieces.
"""
import copy
import json
import re
import time
import uuid
from contextlib import AsyncExitStack
from functools import partial

import inventory
from design.routing import MAX_ITEMS_PER_REQUEST, caller, research_mode
from design.state import DesignState, model_for
from furniture.store import CATEGORIES, category_of, compact, get_store
from llm import LLMError, supports_image_input, tool_completion
from observability import progress
from prompts import load_prompt
from research import jobs as research_jobs
from research.asset_download import download_key
from research.mcp_client import AssetMCP, AssetMCPError
from research.previews import PreviewError, fetch_preview

MAX_SUGGESTIONS_PER_ITEM = 4
MAX_IMAGE_CHECKS_PER_ITEM = 6
MAX_LLM_STEPS_PER_ITEM = 3
MAX_ONLINE_RUNS_PER_ITEM = 2
MAX_LOCAL_STEPS = 2          # LLM calls of the local check of one item (refine once, then choose)
LOCAL_HITS = 6               # local matches shown to the local check
LOADABLE_FORMATS = {"glb", "gltf"}
LOCAL_MIN_SCORE = 0.25       # cosine similarity below this is not shown as a local match
PROGRESS_INTERVAL_S = 0.5    # how often buffered job progress is forwarded while waiting
RATE_LIMIT = re.compile(r"rate.?limit|too many requests|\b429\b", re.IGNORECASE)

FINAL = ("found_locally", "completed", "failed", "skipped", "rejected", "not_found")
ACTIVE = ("active", "awaiting_approval", "download")  # research_status while a list is being worked on
CHECK_KEYS = ("category", "size", "style", "quality", "license")

# The interior designer sees the inventory and only asks for research when nothing there fits,
# so a second local search would just repeat its check (scope_designer instead of scope_direct).
_RULE_LIMITS = {"max_llm_steps": MAX_LLM_STEPS_PER_ITEM, "max_image_checks": MAX_IMAGE_CHECKS_PER_ITEM}
SYSTEM_PROMPT = load_prompt("furniture_research_system", scope=load_prompt("furniture_research_scope_direct"),
                            **_RULE_LIMITS)
EXTERNAL_ONLY_PROMPT = load_prompt("furniture_research_system", scope=load_prompt("furniture_research_scope_designer"),
                                   **_RULE_LIMITS)
GUIDED_RULE = load_prompt("furniture_research_guided", max_suggestions=MAX_SUGGESTIONS_PER_ITEM)
AUTOMATIC_RULE = load_prompt("furniture_research_automatic")
LOCAL_PROMPT = load_prompt("furniture_research_local")


def searches_locally(state: DesignState) -> bool:
    """Local search only when the user asked directly (orchestrator); the designer already checked the inventory."""
    return caller(state, "furniture_research") == "end"


SEARCH_LOCAL_TOOL = {"type": "function", "function": {
    "name": "search_local_furniture",
    "description": "Semantic search in the user's furniture database (downloaded and described models). "
                   "Returns item_id, name, category, style, colours, materials, size in metres and a score (0-1).",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Natural-language description, e.g. 'minimalist wooden chair'"},
        "category": {"type": "string", "enum": list(CATEGORIES), "description": "Optional hard filter"},
        "max_width": {"type": "number", "description": "Optional, metres"},
        "max_depth": {"type": "number", "description": "Optional, metres"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 15},
    }, "required": ["query"]}}}
LOCAL_TOOLS = [
    SEARCH_LOCAL_TOOL,
    {"type": "function", "function": {
        "name": "choose_local_item",
        "description": "Finish the local check: the item that meets every essential requirement, or null.",
        "parameters": {"type": "object", "properties": {
            "item_id": {"type": ["string", "null"], "description": "item_id of the match, or null"},
            "requirements": {"type": "array", "items": {"type": "object", "properties": {
                "requirement": {"type": "string"},
                "met": {"type": "boolean"},
                "evidence": {"type": "string", "description": "What in the item's data shows it"},
            }, "required": ["requirement", "met"]}},
            "reason": {"type": "string", "description": "1-2 sentences, in the language of the user's request"},
        }, "required": ["item_id", "requirements", "reason"]}}},
]
EXTERNAL_TOOLS = [
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
        "description": "Finish the research of this item with the ranked candidates.",
        "parameters": {"type": "object", "properties": {
            "interpretation": {"type": "string", "description": "One sentence: how you understood the item"},
            "summary": {"type": "string", "description": "1-2 sentences on what you looked for and found"},
            "candidates": {"type": "array", "maxItems": MAX_SUGGESTIONS_PER_ITEM, "items": {"type": "object", "properties": {
                "asset_id": {"type": "string", "description": "Id of an external asset"},
                "reason": {"type": "string", "description": "Why it fits the item (1-2 sentences)"},
                "visual_check": {"type": "string", "description": "What the preview showed, or 'not inspected'"},
                "suitable": {"type": "boolean", "description": "true only if it really fits the item"},
                "checks": {"type": "object", "properties": {
                    key: {"type": "string", "enum": ["ok", "unclear", "fails"]} for key in CHECK_KEYS}},
            }, "required": ["asset_id", "reason", "suitable"]}},
        }, "required": ["summary", "candidates"]}}},
]


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


def auto_acceptable(candidate: dict) -> bool:
    """The deterministic gate for automatic downloads: the agent must call it suitable, the category
    must be confirmed, no check may fail and the licence must be known."""
    checks = candidate.get("checks") or {}
    return (candidate.get("suitable") is True and checks.get("category") == "ok"
            and not any(value == "fails" for value in checks.values()) and bool(candidate.get("license")))


class Research:
    """One search of one item: the tool implementations plus everything seen so far. The MCP
    session (one per item search, so parallel items never share a session) is only opened when the
    agent searches externally."""

    def __init__(self, stack: AsyncExitStack, excluded: set[str], vision: bool):
        self.stack = stack
        self._mcp: AssetMCP | None = None
        self.excluded = excluded
        self.vision = vision
        self.seen: dict[str, dict] = {}       # external asset id -> metadata from search / get_asset
        self.local: dict[str, dict] = {}      # inventory item id -> local search hit
        self.in_inventory = inventory_asset_ids()
        self.previewed: dict[str, str] = {}   # asset id -> "ok" or why not
        self.stats = {"local_searches": 0, "searches": 0, "details": 0, "previews": 0, "found": 0, "server_errors": 0}
        # Set when the asset server cannot be used for the rest of this run (no session, rate limit):
        # further calls fail at once instead of sending more requests to an overloaded server.
        self.blocked: str | None = None

    async def mcp(self) -> AssetMCP:
        if self.blocked:
            raise AssetMCPError(self.blocked)
        if self._mcp is None:
            try:
                self._mcp = await self.stack.enter_async_context(AssetMCP())
            except AssetMCPError as exc:
                self.blocked = str(exc)
                raise
        return self._mcp

    def server_error(self, exc: AssetMCPError) -> None:
        self.stats["server_errors"] += 1
        if RATE_LIMIT.search(str(exc)):
            self.blocked = self.blocked or str(exc)

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
        if len(self.previewed) >= MAX_IMAGE_CHECKS_PER_ITEM:
            return f"Preview limit of {MAX_IMAGE_CHECKS_PER_ITEM} reached for this item.", None
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

    def local_candidate(self, item_id: str, reason: str = "", asset_id: str | None = None) -> dict:
        hit = self.local.get(item_id) or {}
        name = hit.get("name") or (self.seen.get(asset_id) or {}).get("title") or item_id
        return {"itemId": item_id, "name": name, "category": hit.get("category"),
                "description": hit.get("description"), "thumbnailUrl": hit.get("thumbnail_url"),
                "score": hit.get("score"), "reason": reason, "source": "local"}

    def local_match(self, choice: dict | None) -> dict | None:
        """The local check's choice, only if the item was really found and every listed essential
        requirement is met (a category match alone is not enough)."""
        choice = choice or {}
        item_id, requirements = choice.get("item_id"), choice.get("requirements") or []
        if not item_id or item_id not in self.local or not requirements:
            return None
        if not all(r.get("met") is True for r in requirements):
            return None
        return self.local_candidate(item_id, choice.get("reason", ""))

    def candidates(self, submitted: list[dict]) -> tuple[list[dict], list[dict]]:
        """Validated, ranked candidates split into (external candidates, items already in the
        inventory). Unknown, excluded, unloadable or duplicate ids and candidates the agent itself
        marked unsuitable are dropped: only items the agent actually found can be proposed."""
        shortlist, local, used = [], [], set()
        for entry in submitted or []:
            asset_id = entry.get("asset_id")
            if entry.get("suitable") is False:
                continue
            item_id = asset_id if asset_id in self.local else self.in_inventory.get(asset_id)
            if item_id and item_id not in used:
                used.add(item_id)
                local.append(self.local_candidate(item_id, entry.get("reason", ""), asset_id))
                continue
            asset = self.seen.get(asset_id)
            if not asset or asset_id in self.excluded or asset_id in used or not _loadable(asset):
                continue
            if len(shortlist) >= MAX_SUGGESTIONS_PER_ITEM:
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
                "suitable": entry.get("suitable"),
                "checks": {k: v for k, v in (entry.get("checks") or {}).items() if k in CHECK_KEYS},
                "source": "external",
            })
        return shortlist, local


# --- one item: local check and online search (background jobs) ------------------------------------

def job_context(state: DesignState) -> dict:
    """Everything a job needs, copied from the state when it starts (jobs never read the live state)."""
    room = state["room"]
    furniture = [f"{f.name} ({f.width:.2f}×{f.depth:.2f} m)" if f.width and f.depth else f.name
                 for f in state.get("furniture") or []]
    return {"request": state["request"], "room": f"{room.width} × {room.depth} m, height {room.height} m",
            "furniture": furniture, "model": model_for(state, "furniture_research"),
            "local": searches_locally(state), "mode": research_mode(state),
            "shared": {k: v for k, v in (state.get("research_batch") or {}).get("shared", {}).items() if v}}


def _item_query(request: dict) -> str:
    return ", ".join(str(request[k]) for k in ("description", "style", "colors_materials") if request.get(k))


def _item_lines(context: dict, item: dict) -> list[str]:
    request = {k: v for k, v in item["request"].items() if v not in (None, "", False)}
    lines = [f"User request: {context['request']}",
             f"Item {item['index']} of the furniture list: {json.dumps(request, ensure_ascii=False)}"]
    if context["shared"]:
        lines.append(f"For the whole list: {json.dumps(context['shared'], ensure_ascii=False)}")
    lines.append(f"Room: {context['room']}.")
    if context["furniture"]:
        lines.append("Furniture already in the room: " + ", ".join(context["furniture"]))
    return lines


def build_messages(context: dict, item: dict, vision: bool) -> list[dict]:
    """Messages of one online run of one item."""
    lines = _item_lines(context, item)
    if context["local"] and item.get("local_assessment"):
        lines.append(f"Local check: {item['local_assessment']}")
    lines.append(AUTOMATIC_RULE if context["mode"] == "automatic" else GUIDED_RULE)
    if not vision:
        lines.append("Note: your model cannot see images; rely on metadata.")
    if item["run"] > 1:
        feedback = f" User feedback: {item['feedback']}" if item.get("feedback") else ""
        lines.append(f"This is online search run {item['run']} of {MAX_ONLINE_RUNS_PER_ITEM} for this item. "
                     f"{item.get('retry_reason') or 'The first suggestions did not fit.'}{feedback} Those assets are "
                     "excluded now. Broaden the search with other terms and synonyms and find different candidates.")
    prompt = SYSTEM_PROMPT if context["local"] else EXTERNAL_ONLY_PROMPT
    return [{"role": "system", "content": prompt}, {"role": "user", "content": "\n".join(lines)}]


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
        research.server_error(exc)
        text, image = f"Tool error: {exc}", None
        data.update(status="failed", error=str(exc))
    data["durationMs"] = round((time.perf_counter() - started) * 1000)
    progress.emit("tool", **data)
    return text, image


async def _tool_loop(model: str, messages: list[dict], tools: list[dict], finish: str, max_steps: int,
                     research: Research) -> dict | None:
    """Runs the agent's tool calls until it calls `finish` (forced in the last step); returns its arguments."""
    allowed = {t["function"]["name"] for t in tools}
    final_choice = {"type": "function", "function": {"name": finish}}
    for step in range(max_steps):
        try:
            message, info = await tool_completion(model, messages, tools,
                                                  final_choice if step == max_steps - 1 else "auto")
        except LLMError as exc:
            if exc.info:
                progress.emit("llm", **exc.info.as_dict())
            raise
        progress.emit("llm", **info.as_dict())
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": f"Use the tools, and finish with {finish}."})
            continue
        images, submitted = [], None
        for call in calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = None
            if name == finish and isinstance(arguments, dict):
                submitted, text = arguments, "Submitted."
            elif arguments is None:
                text = "Invalid JSON arguments."
            elif name not in allowed:
                text = ("Not available: the interior designer already checked the inventory. Use search_assets."
                        if name == "search_local_furniture" else f"Tool {name} is not available here.")
            else:
                text, image = await _run_tool(research, name, arguments)
                if image:
                    images.append((arguments.get("asset_id"), image))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        if submitted is not None:
            return submitted
        if research.blocked:  # the asset server cannot be used: further LLM steps cannot find anything
            return None
        if images:  # tool results are text only, so previews follow as a user message with image parts
            content = [{"type": "text", "text": "Requested preview images:"}]
            for asset_id, image in images:
                content += [{"type": "text", "text": f"Preview of {asset_id}:"},
                            {"type": "image_url", "image_url": {"url": image}}]
            messages.append({"role": "user", "content": content})
    return None


def _label(item: dict) -> str:
    return f"Item {item['index']} ({item['request']['description'][:60]})"


async def check_local(context: dict, item: dict) -> dict:
    """Local search for one item of a direct search: the furniture database is searched first, an LLM
    only checks whether the best matches meet the item's essential requirements."""
    async with AsyncExitStack() as stack:
        research = Research(stack, set(), vision=False)
        text, _ = await _run_tool(research, "search_local_furniture",
                                  {"query": _item_query(item["request"]), "limit": LOCAL_HITS})
        if not research.local:
            progress.step(f"local:{item['key']}", f"{_label(item)}: no local match", item=item["key"], localMatch=False)
            return {"match": None, "assessment": "No similar item in the furniture database."}
        messages = [{"role": "system", "content": LOCAL_PROMPT},
                    {"role": "user", "content": "\n".join(_item_lines(context, item))
                     + f"\n\nMatches in the user's furniture database:\n{text}"}]
        choice = await _tool_loop(context["model"], messages, LOCAL_TOOLS, "choose_local_item", MAX_LOCAL_STEPS, research)
        match = research.local_match(choice)
    assessment = (choice or {}).get("reason") or ("A local item fits." if match else "No local item meets the requirements.")
    progress.step(f"local:{item['key']}", f"{_label(item)}: local match {'yes: ' + match['name'] if match else 'no'}",
                  item=item["key"], localMatch=bool(match), requirements=(choice or {}).get("requirements"))
    return {"match": match, "assessment": assessment}


async def search_online(context: dict, item: dict) -> dict:
    """One online run for one item: search, inspect, look at previews and submit ranked candidates."""
    model = context["model"]
    vision = await supports_image_input(model) is not False
    messages = build_messages(context, item, vision)
    async with AsyncExitStack() as stack:
        research = Research(stack, set(item.get("rejected_asset_ids") or []), vision)
        submitted = await _tool_loop(model, messages, EXTERNAL_TOOLS, "submit_candidates", MAX_LLM_STEPS_PER_ITEM,
                                     research) or {}
    candidates, local = research.candidates(submitted.get("candidates") or [])
    if research.blocked and not candidates and not local:
        # Nothing was really searched: report the server problem instead of "nothing suitable found",
        # and do not spend the item's second online run on it.
        raise AssetMCPError(f"asset server unavailable: {research.blocked}")
    progress.step(f"search:{item['key']}:{item['run']}",
                  f"{_label(item)}: online run {item['run']}, {len(candidates)} candidate(s)"
                  + (f", {len(local)} already in the inventory" if local else ""),
                  item=item["key"], run=item["run"], **research.stats)
    return {"candidates": candidates, "local": local, "summary": submitted.get("summary") or "",
            "interpretation": submitted.get("interpretation") or "", "stats": research.stats}


# --- the list: coordinator node --------------------------------------------------------------------

def _new_item(key: str, index: int, request: dict) -> dict:
    request = {"description": str(request.get("description") or "").strip()[:300],
               "purpose": request.get("purpose"), "size": request.get("size"), "style": request.get("style"),
               "colors_materials": request.get("colors_materials"), "essential": bool(request.get("essential"))}
    return {"key": key, "index": index, "request": request, "status": "pending", "run": 0, "job": None,
            "online": False, "local_checked": False, "local_assessment": "", "ready_seq": None,
            "candidates": [], "interpretation": "", "summary": "", "local": [], "selected": [],
            "approved_by": None, "approved_run": None, "rejected_asset_ids": [], "feedback": None,
            "retry_reason": None, "downloads": [], "note": None}


def new_batch(state: DesignState) -> tuple[list[dict], dict, list[str]]:
    """The list from the request: the designer's items (essential first), or the user's search."""
    request = state.get("furniture_request") or {}
    raw = [i for i in request.get("items") or [] if str((i or {}).get("description") or "").strip()]
    raw = raw or [{"description": state["request"]}]
    ordered = sorted(raw, key=lambda i: not i.get("essential"))  # stable: the designer's order otherwise
    batch_id = uuid.uuid4().hex[:8]
    items = [_new_item(f"{batch_id}-{n}", n, r) for n, r in enumerate(ordered[:MAX_ITEMS_PER_REQUEST], 1)]
    log = [f"furniture_research: new list with {len(items)} item(s) from {caller(state, 'furniture_research')}"]
    if len(ordered) > MAX_ITEMS_PER_REQUEST:
        dropped = ", ".join(i["description"] for i in ordered[MAX_ITEMS_PER_REQUEST:])
        log.append(f"furniture_research: list cut to {MAX_ITEMS_PER_REQUEST} items, not searched: {dropped}")
        progress.step("truncated", f"List cut to {MAX_ITEMS_PER_REQUEST} items", dropped=dropped)
    shared = {k: request.get(k) for k in ("style", "budget")}
    return items, {"id": batch_id, "online": False, "seq": 0, "shared": shared}, log


class Coordinator:
    """One execution of the furniture_research node over the list. Only this node, human_approval and
    asset_download change items, one node at a time; jobs only produce results."""

    def __init__(self, state: DesignState, items: list[dict], batch: dict, log: list[str]):
        self.state, self.items, self.batch, self.log = state, items, batch, log
        self.jobs = research_jobs.for_run()
        self.mode = research_mode(state)
        self.local = searches_locally(state)
        self.context = job_context({**state, "research_batch": batch})
        self.counters: dict[str, int] = {}
        self.downloads = {"added": [], "existing": [], "failed": []}  # merged in this execution
        self._reported = None

    # budget
    def _spend(self, counter: str) -> None:
        self.counters[counter] = self.counters.get(counter, 0) + 1

    def _go_online(self, item: dict) -> None:
        """Next online run of an item, if the settings allow it."""
        if self.mode == "local_only":
            item.update(status="not_found", note="online search is disabled and nothing suitable is in the furniture database")
            return
        if not item["online"]:
            if not self.local and not self.batch["online"]:
                self._spend("research_requests")  # a designer list counts once it really goes online
            self.batch["online"] = item["online"] = True
        item.update(status="searching_online", run=item["run"] + 1, job=None, candidates=[], selected=[])

    def _next_seq(self) -> int:
        self.batch["seq"] = self.batch.get("seq", 0) + 1
        return self.batch["seq"]

    # scheduling
    def schedule(self) -> None:
        for item in self.items:
            if item["status"] == "pending":
                if self.local and not item["local_checked"]:
                    item["status"] = "searching_local"
                else:
                    self._go_online(item)
            elif item["status"] == "retrying":
                self._go_online(item)
            status = item["status"]
            if status in ("searching_local", "searching_online"):
                kind = "local" if status == "searching_local" else "search"
                item["job"] = item["job"] or f"{kind}:{item['key']}:{item['run']}"
                state = self.jobs.state(item["job"])
                if state in ("missing", "lost"):
                    work = check_local if kind == "local" else search_online
                    if self.jobs.start(item["job"], "research", partial(work, self.context, copy.deepcopy(item))):
                        label = "local search" if kind == "local" else f"online search run {item['run']}"
                        progress.step(f"start:{item['job']}", f"{_label(item)}: {label}"
                                      + (" (restarted)" if state == "lost" else ""), item=item["key"])
            elif status == "downloading":
                for entry in item["downloads"]:
                    if entry["status"] == "running" and self.jobs.state(download_key(entry["assetId"])) in ("missing", "lost"):
                        entry["status"] = "queued"  # asset_download starts it again (idempotent)

    # merging results
    def collect(self) -> None:
        for item in self.items:
            status, key = item["status"], item["job"]
            if status in ("searching_local", "searching_online") and key and self.jobs.state(key) == "done":
                value, error = self.jobs.result(key)
                if status == "searching_local":
                    self._apply_local(item, value, error)
                elif error:
                    item.update(status="failed", note=f"online search failed ({type(error).__name__}: {error})"[:300])
                    self.log.append(f"furniture_research: {_label(item)} search failed")
                else:
                    self._apply_search(item, value)
            elif status == "downloading":
                self._apply_downloads(item)

    def _apply_local(self, item: dict, value: dict | None, error: Exception | None) -> None:
        item["local_checked"] = True
        if error or not value:
            item.update(status="pending", local_assessment=f"Local check failed ({type(error).__name__}).")
        elif value["match"]:
            item.update(status="found_locally", local=[value["match"]], local_assessment=value["assessment"])
            self.log.append(f"furniture_research: {_label(item)} found locally: {value['match']['name']}")
        else:
            item.update(status="pending", local_assessment=value["assessment"])

    def _apply_search(self, item: dict, value: dict) -> None:
        item.update(interpretation=value["interpretation"], summary=value["summary"])
        if value["local"]:  # already in the inventory: no download, no approval
            item.update(status="found_locally", local=value["local"], note="already in the inventory")
            self.log.append(f"furniture_research: {_label(item)} already in the inventory")
            return
        candidates = value["candidates"]
        if self.mode == "automatic":
            best = next((c for c in candidates if auto_acceptable(c)), None)
            if best:
                item.update(candidates=[best], selected=[best["assetId"]], approved_by="auto", approved_run=item["run"],
                            downloads=[{"assetId": best["assetId"], "name": best["name"], "status": "queued"}],
                            status="downloading")
                self.log.append(f"furniture_research: {_label(item)} chose {best['name']} automatically")
                progress.step(f"auto:{item['key']}", f"{_label(item)}: chose {best['name']} automatically", item=item["key"])
                return
            reason = "No candidate of the first run was suitable enough to download unattended."
        elif candidates:
            item.update(candidates=candidates, status="awaiting_approval", ready_seq=self._next_seq())
            progress.step(f"ready:{item['key']}", f"{_label(item)}: {len(candidates)} suggestion(s), waiting for approval",
                          item=item["key"], run=item["run"])
            return
        else:
            reason = "The first run found no suitable model."
        seen = [c["assetId"] for c in candidates]
        if item["run"] < MAX_ONLINE_RUNS_PER_ITEM:
            item.update(status="retrying", retry_reason=reason, candidates=[],
                        rejected_asset_ids=list(dict.fromkeys([*item["rejected_asset_ids"], *seen])))
        else:
            item.update(status="not_found", candidates=[], note=f"nothing suitable found in {item['run']} online run(s)")
            self.log.append(f"furniture_research: {_label(item)} not found")

    def _apply_downloads(self, item: dict) -> None:
        for entry in item["downloads"]:
            key = download_key(entry["assetId"])
            if entry["status"] != "running" or self.jobs.state(key) != "done":
                continue
            value, error = self.jobs.result(key)
            if error:
                value = {"assetId": entry["assetId"], "name": entry["name"], "status": "failed",
                         "error": f"{type(error).__name__}: {error}"[:300]}
            entry.update(value)
            if value["status"] == "done":
                bucket = "added" if value.get("created") else "existing"
                if value["assetId"] not in {a["assetId"] for a in self.downloads[bucket]}:
                    self.downloads[bucket].append(_entry(value))
            else:
                self.downloads["failed"].append({"assetId": value["assetId"], "name": value["name"], "error": value["error"]})
            progress.step(f"downloaded:{entry['assetId']}", f"{_label(item)}: download "
                          + ("completed" if value["status"] == "done" else f"failed ({value.get('error')})"), item=item["key"])
        if item["downloads"] and all(e["status"] in ("done", "failed") for e in item["downloads"]):
            ok = any(e["status"] == "done" for e in item["downloads"])
            item["status"] = "completed" if ok else "failed"
            if not ok:
                item["note"] = "download failed"
            self.log.append(f"furniture_research: {_label(item)} {item['status']}")

    # outcomes of this execution
    def _base(self) -> dict:
        update = {"research_items": self.items, "research_batch": self.batch, "counters": self.counters,
                  "workflow_log": self.log}
        if any(self.downloads.values()):
            update["download_result"] = self.downloads
        return update

    def pause_for(self, item: dict) -> dict:
        self.log.append(f"furniture_research: {_label(item)} ready for approval (run {item['run']}, "
                        f"{len(item['candidates'])} suggestion(s))")
        return {**self._base(), "approval_item": item["key"], "research_status": "awaiting_approval", "approval": None}

    def dispatch(self) -> dict:
        return {**self._base(), "research_status": "download", "approval_item": None}

    def finish(self) -> dict:
        result = research_result(self.items)
        requester = caller(self.state, "furniture_research")
        self.log.append(f"furniture_research: {result['status']} -> {requester}")
        progress.step("finish", f"Research {result['status']}; returning to {requester}",
                      **{k: len(v) for k, v in result.items() if isinstance(v, list)})
        update = {**self._base(), "research_status": "done", "research_result": result, "approval_item": None,
                  "approval": None}
        if requester == "end":
            update["outcome"] = {"status": "failed" if result["status"] == "download_failed" else "completed",
                                 "kind": "research", "message": result["summary"]}
        return update

    def report(self) -> None:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        snapshot = (tuple(sorted(counts.items())), dict(self.counters))
        if snapshot != self._reported:
            self._reported = snapshot
            progress.step("research_status", "Furniture list: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())),
                          counts=counts, total=len(self.items))

    async def run(self) -> dict:
        while True:
            self.collect()
            self.schedule()
            progress.replay(self.jobs.take_events())
            self.report()
            if self.mode == "guided":
                ready = sorted((i for i in self.items if i["status"] == "awaiting_approval"), key=lambda i: i["ready_seq"])
                if ready:
                    return self.pause_for(ready[0])
            if any(e["status"] == "queued" for i in self.items for e in i["downloads"]):
                return self.dispatch()
            if all(i["status"] in FINAL for i in self.items):
                return self.finish()
            waiting = [i["job"] for i in self.items if i["status"] in ("searching_local", "searching_online")]
            waiting += [download_key(e["assetId"]) for i in self.items if i["status"] == "downloading"
                        for e in i["downloads"] if e["status"] == "running"]
            if not any(self.jobs.state(k) == "running" for k in waiting):
                if self.jobs.closed:
                    raise RuntimeError("The research jobs of this run were closed.")
                continue  # results arrived meanwhile (or lost jobs were restarted): merge them
            await self.jobs.wait(waiting, timeout=PROGRESS_INTERVAL_S)


def _entry(value: dict) -> dict:
    return {k: v for k, v in value.items() if k not in ("status", "created")}


def research_result(items: list[dict]) -> dict:
    """What the caller gets: one entry per item plus flat lists over the whole list."""
    per_item, added, existing, failed, local = [], [], [], [], []
    for item in items:
        done = [e for e in item["downloads"] if e["status"] == "done"]
        entry = {"key": item["key"], "description": item["request"]["description"],
                 "essential": item["request"]["essential"], "status": item["status"], "runs": item["run"],
                 "auto": item.get("approved_by") == "auto", "local": item["local"],
                 "added": [_entry(e) for e in done if e.get("created")],
                 "existing": [_entry(e) for e in done if not e.get("created")],
                 "failed": [{"assetId": e["assetId"], "name": e["name"], "error": e.get("error")}
                            for e in item["downloads"] if e["status"] == "failed"],
                 "note": item.get("note")}
        per_item.append(entry)
        for bucket, values in ((added, entry["added"]), (existing, entry["existing"]), (failed, entry["failed"]),
                               (local, entry["local"])):
            for value in values:
                ident = value.get("assetId") or value.get("itemId")
                if ident not in {v.get("assetId") or v.get("itemId") for v in bucket}:
                    bucket.append(value)
    statuses = {i["status"] for i in items}
    if added or existing:
        status = "completed"
    elif local:
        status = "found_locally"
    elif failed:
        status = "download_failed"
    elif statuses and statuses <= {"skipped", "rejected"}:
        status = "rejected"
    else:
        status = "no_results"
    return {"status": status, "summary": " ".join(_item_summary(e) for e in per_item), "items": per_item,
            "added": added, "existing": existing, "failed": failed, "local": local}


def _names(entries: list[dict]) -> str:
    return ", ".join(e["name"] for e in entries)


def _item_summary(entry: dict) -> str:
    status = entry["status"]
    if status == "found_locally":
        text = f"already in your inventory: {_names(entry['local'])}"
    elif status in ("completed", "failed"):
        parts = []
        if entry["added"]:
            parts.append(f"added to the inventory{' (chosen automatically)' if entry['auto'] else ''}: "
                         f"{_names(entry['added'])}")
        if entry["existing"]:
            parts.append(f"already in the inventory: {_names(entry['existing'])}")
        if entry["failed"]:
            parts.append("download failed: " + "; ".join(f"{f['name']} ({f['error']})" for f in entry["failed"]))
        text = "; ".join(parts)
    else:
        text = {"skipped": "skipped by you",
                "rejected": f"all suggestions were rejected in {entry['runs']} search runs; nothing was downloaded",
                }.get(status) or entry.get("note") or "nothing suitable found"
    return f"{entry['description']}: {text}."


async def furniture_research(state: DesignState) -> dict:
    """Coordinator of a furniture list: starts a new list, or continues it after an approval or after
    downloads were started. Pauses for one item at a time, never for the whole list."""
    if state.get("research_status") in ACTIVE:
        items = copy.deepcopy(state.get("research_items") or [])
        batch, log = dict(state.get("research_batch") or {}), []
    else:
        items, batch, log = new_batch(state)
        if not searches_locally(state):
            for item in items:
                progress.step(f"local:{item['key']}", f"{_label(item)}: local search skipped (designer)", item=item["key"])
        with progress.stage("prepare", "Understand the furniture list") as details:
            details.update(items=[i["request"]["description"] for i in items], mode=research_mode(state),
                           requester=caller(state, "furniture_research"))
    return await Coordinator(state, items, batch, log).run()
