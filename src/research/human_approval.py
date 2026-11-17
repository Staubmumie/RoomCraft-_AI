"""Human-in-the-loop: pauses the workflow with LangGraph's interrupt() until the user has decided on
ONE item of a furniture list (state["approval_item"]). Other items keep being searched or downloaded
in the background meanwhile (research.jobs); the research coordinator shows the next ready item.

Payload (approval_request): {"kind": "asset_selection", "itemKey", "index", "total", "run", "maxRuns",
"request", "intent", "summary", "candidates", "localDone", "progress"}

Resume value (sent through POST /api/design/runs/{id}/approval):
    {"itemKey": "...", "action": "approve" | "reject" | "skip", "selected": [asset ids], "feedback": "..."}

approve  -> asset_download starts the downloads right away
reject   -> run 1: a second online run starts (rejected assets excluded, feedback used); run 2: item rejected
skip     -> the item ends without another search
"""
from langgraph.types import interrupt

from design.state import DesignState
from research.agent import MAX_ONLINE_RUNS_PER_ITEM as MAX_RUNS

ACTIONS = ("approve", "reject", "skip")


class ApprovalError(ValueError):
    pass


def validate_decision(request: dict, decision: dict) -> dict:
    """Normalises a decision for the pending `request`. Stale decisions (another item, an older run)
    are refused, and selected ids must belong to this item's candidates of this run."""
    action = decision.get("action")
    if action not in ACTIONS:
        raise ApprovalError(f"Unknown action {action!r}.")
    if decision.get("itemKey") != request.get("itemKey"):
        raise ApprovalError("This decision is for another furniture item; the approval has moved on.")
    if decision.get("run") is not None and decision["run"] != request.get("run"):
        raise ApprovalError("This decision is for an earlier search run of this item.")
    known = {c["assetId"] for c in request.get("candidates") or []}
    selected = list(dict.fromkeys(decision.get("selected") or []))
    unknown = [s for s in selected if s not in known]
    if unknown:
        raise ApprovalError(f"Not among the proposed candidates: {', '.join(unknown)}")
    if action == "approve" and not selected:
        raise ApprovalError("Select at least one asset to approve.")
    return {
        "itemKey": request["itemKey"],
        "action": action,
        "selected": selected if action == "approve" else [],
        "feedback": str(decision.get("feedback") or "").strip()[:1000],
    }


def _item(state: DesignState) -> dict:
    key = state.get("approval_item")
    item = next((i for i in state.get("research_items") or [] if i["key"] == key), None)
    if item is None or item["status"] != "awaiting_approval":
        raise ValueError(f"No furniture item {key!r} is waiting for approval.")
    return item


def approval_request(state: DesignState) -> dict:
    """What the frontend shows while the workflow waits: exactly one item."""
    items = state.get("research_items") or []
    item = _item(state)
    counts: dict[str, int] = {}
    for other in items:
        if other["key"] != item["key"]:
            counts[other["status"]] = counts.get(other["status"], 0) + 1
    done = [{"description": i["request"]["description"], "status": i["status"],
             "items": [{"itemId": c["itemId"], "name": c["name"]} for c in i.get("local") or []]
             + [{"itemId": d.get("itemId"), "name": d["name"]} for d in i.get("downloads") or [] if d["status"] == "done"]}
            for i in items if i["status"] in ("found_locally", "completed")]
    return {
        "kind": "asset_selection",
        "itemKey": item["key"],
        "index": item["index"],
        "total": len(items),
        "run": item["run"],
        "maxRuns": MAX_RUNS,
        "request": item["request"],
        "intent": item.get("interpretation") or "",
        "summary": item.get("summary") or "",
        "candidates": item["candidates"],
        "localDone": done,
        "progress": counts,
    }


def human_approval(state: DesignState) -> dict:
    request = approval_request(state)
    # Execution stops here until the user answers; on resume interrupt() returns the answer. The
    # node runs again from the top then, so nothing above this line may have side effects.
    decision = validate_decision(request, interrupt(request))
    items = [dict(i) for i in state["research_items"]]
    item = next(i for i in items if i["key"] == request["itemKey"])
    action = decision["action"]
    if action == "approve":
        item.update(selected=decision["selected"], approved_by="user", approved_run=item["run"], status="downloading",
                    downloads=[{"assetId": c["assetId"], "name": c["name"], "status": "queued"}
                               for c in item["candidates"] if c["assetId"] in decision["selected"]])
    else:
        # Not proposed again in a later run of this item.
        item["rejected_asset_ids"] = list(dict.fromkeys([*item.get("rejected_asset_ids", []),
                                                         *(c["assetId"] for c in item["candidates"])]))
        item["feedback"] = decision["feedback"] or None
        if action == "skip":
            item.update(status="skipped", candidates=[])
        elif item["run"] < MAX_RUNS:
            item.update(status="retrying", candidates=[], retry_reason="The user rejected all suggestions.")
        else:
            item.update(status="rejected", candidates=[])
    entry = f"human_approval: item {item['index']} {action}" + (
        f" {len(decision['selected'])} of {len(request['candidates'])}" if decision["selected"] else "")
    # approve -> asset_download; reject/skip -> furniture_research, which continues with the list.
    return {"research_items": items, "approval": decision, "approval_item": None, "research_status": "active",
            "workflow_log": [entry]}
