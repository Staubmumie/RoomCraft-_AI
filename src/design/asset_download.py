"""Asset Download node: runs the shared asset pipeline (furniture.pipeline) for every asset the
user approved: download -> measure -> add description -> save to the furniture database.

Only ids from the approval decision that are also among the proposed candidates are processed.
Always returns to furniture_research (fixed edge), which checks the result.
"""
from fastapi import HTTPException

from design.state import DesignState
from furniture.pipeline import process_asset
from observability import progress

CANDIDATE_METADATA = ("description", "tags", "license")  # what the research saw, passed on to enrichment


async def asset_download(state: DesignState) -> dict:
    candidates = {c["assetId"]: c for c in state.get("candidates") or []}
    approved = [a for a in (state.get("approval") or {}).get("selected", []) if a in candidates]
    added, existing, failed = [], [], []

    for asset_id in approved:
        candidate = candidates[asset_id]
        name = candidate["name"]
        with progress.stage(f"process:{asset_id}", f"Download, describe and save: {name}") as details:
            try:
                result = await process_asset(asset_id, {k: candidate.get(k) for k in CANDIDATE_METADATA})
            except HTTPException as exc:
                failed.append({"assetId": asset_id, "name": name, "error": str(exc.detail)})
                details.update(error=str(exc.detail))
                continue
            details.update(result.as_dict())
        item = result.item
        entry = {"assetId": asset_id, "itemId": item["id"], "name": item["name"],
                 "dimensions": item.get("dimensions"), "unitScale": item.get("unitScale", 1.0),
                 "searchable": result.stored, "enrichment": result.enrichment}
        if result.warnings:
            entry["warning"] = " ".join(result.warnings)
        (added if result.created else existing).append(entry)

    progress.step("inventory", f"{len(added)} asset(s) added to the inventory",
                  added=len(added), existing=len(existing), failed=len(failed))
    # Always back to furniture_research (fixed edge): it checks the result and decides how to continue.
    return {"download_result": {"added": added, "existing": existing, "failed": failed},
            "workflow_log": [f"asset_download: {len(added)} added, {len(existing)} existing, {len(failed)} failed"]}
