"""Asset Download node: starts the shared asset pipeline (furniture.pipeline) for approved assets:
download -> measure -> add description -> save to the furniture database.

It does not wait for the downloads. Each queued download of a list item becomes a background job
(research.jobs, at most PARALLEL["downloads"] at a time), so the user can already decide on the next
item; the research coordinator collects the results (fixed edge asset_download -> furniture_research).

Only explicitly authorised assets are downloaded (`download_authorized`): a candidate of the item's
current search run that the user selected in human_approval, or - only when the user switched off
"Let me review furniture suggestions" - the candidate the research picked automatically. A job key
per asset id makes sure no asset is downloaded twice, also when two items chose the same asset.
"""
import asyncio
import re
from functools import partial

from fastapi import HTTPException

from design.routing import research_mode
from design.state import DesignState
from furniture.pipeline import process_asset
from observability import progress
from research import jobs as research_jobs

CANDIDATE_METADATA = ("description", "tags", "license")  # what the research saw, passed on to enrichment


def download_key(asset_id: str) -> str:
    return f"download:{asset_id}"


def download_authorized(state: DesignState, item: dict, asset_id: str) -> bool:
    """A download needs a selection of this item's current candidates, made by the user for this
    run, or made automatically while the user's options allow automatic downloads."""
    if asset_id not in {c["assetId"] for c in item.get("candidates") or []} or asset_id not in item.get("selected", []):
        return False
    if item.get("approved_by") == "user":
        return item.get("approved_run") == item.get("run")
    if item.get("approved_by") == "auto":
        return research_mode(state) == "automatic"
    return False


MAX_RATE_LIMIT_WAIT_S = 30


def rate_limit_wait(exc: HTTPException) -> float | None:
    """Seconds to wait before one more try after the asset server's rate limit (None: do not retry)."""
    if exc.status_code != 429:
        return None
    match = re.search(r"retry after (\d+(?:\.\d+)?)\s*s", str(exc.detail), re.IGNORECASE)
    wait = float(match.group(1)) + 1 if match else 20.0
    return wait if wait <= MAX_RATE_LIMIT_WAIT_S else None


async def download(asset_id: str, name: str, metadata: dict) -> dict:
    """One asset through the pipeline; failures are reported per asset, not raised. A rate-limited
    download is tried once more after the wait the server asks for."""
    with progress.stage(f"process:{asset_id}", f"Download, describe and save: {name}") as details:
        try:
            try:
                result = await process_asset(asset_id, metadata)
            except HTTPException as exc:
                if (wait := rate_limit_wait(exc)) is None:
                    raise
                progress.step(f"ratelimit:{asset_id}", f"{name}: asset server rate limit, retrying in {wait:.0f} s")
                await asyncio.sleep(wait)
                result = await process_asset(asset_id, metadata)
        except HTTPException as exc:
            details.update(error=str(exc.detail))
            return {"assetId": asset_id, "name": name, "status": "failed", "error": str(exc.detail)}
        details.update(result.as_dict())
    item = result.item
    entry = {"assetId": asset_id, "itemId": item["id"], "name": item["name"], "dimensions": item.get("dimensions"),
             "unitScale": item.get("unitScale", 1.0), "searchable": result.stored, "enrichment": result.enrichment,
             "status": "done", "created": result.created}
    if result.warnings:
        entry["warning"] = " ".join(result.warnings)
    return entry


async def asset_download(state: DesignState) -> dict:
    jobs = research_jobs.for_run()
    items = [{**i, "downloads": [dict(e) for e in i.get("downloads") or []]} for i in state.get("research_items") or []]
    started, refused = [], []
    for item in items:
        candidates = {c["assetId"]: c for c in item.get("candidates") or []}
        for entry in item["downloads"]:
            if entry["status"] != "queued":
                continue
            asset_id = entry["assetId"]
            if not download_authorized(state, item, asset_id):
                entry.update(status="failed", error="Not approved for download.")
                refused.append(asset_id)
                continue
            metadata = {k: candidates[asset_id].get(k) for k in CANDIDATE_METADATA}
            # An already running or finished download of the same asset is shared, not repeated.
            jobs.start(download_key(asset_id), "downloads", partial(download, asset_id, entry["name"], metadata))
            entry["status"] = "running"
            started.append(entry["name"])
            progress.step(f"download:{asset_id}", f"Item {item['index']}: download started: {entry['name']}",
                          item=item["key"], assetId=asset_id)
    log = f"asset_download: started {len(started)} download(s)" + (f", refused {len(refused)}" if refused else "")
    return {"research_items": items, "research_status": "active", "workflow_log": [log]}
