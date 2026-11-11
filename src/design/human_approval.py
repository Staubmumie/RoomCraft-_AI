"""Human-in-the-loop: pauses the workflow with LangGraph's interrupt() until the user has chosen
from the proposed assets. It only reads `candidates`, so any agent that proposes assets can use it.

Resume value (sent through POST /api/design/runs/{id}/approval):
    {"action": "approve" | "reject" | "retry", "selected": [asset ids], "feedback": "..."}
"""
from langgraph.types import interrupt

from design.state import DesignState

ACTIONS = ("approve", "reject", "retry")


class ApprovalError(ValueError):
    pass


def validate_decision(candidates: list[dict], decision: dict) -> dict:
    """Normalises a decision; selected ids must belong to the pending candidates."""
    action = decision.get("action")
    if action not in ACTIONS:
        raise ApprovalError(f"Unknown action {action!r}.")
    known = {c["assetId"] for c in candidates}
    selected = list(dict.fromkeys(decision.get("selected") or []))
    unknown = [s for s in selected if s not in known]
    if unknown:
        raise ApprovalError(f"Not among the proposed candidates: {', '.join(unknown)}")
    if action == "approve" and not selected:
        raise ApprovalError("Select at least one asset to approve.")
    return {
        "action": action,
        "selected": selected if action == "approve" else [],
        "feedback": str(decision.get("feedback") or "").strip()[:1000],
    }


def approval_request(state: DesignState) -> dict:
    """What the frontend shows while the workflow waits."""
    return {
        "kind": "asset_selection",
        "round": state.get("research_round", 1),
        "summary": state.get("research_summary", ""),
        "candidates": state.get("candidates") or [],
    }


def human_approval(state: DesignState) -> dict:
    candidates = state.get("candidates") or []
    # Execution stops here until the user answers; on resume interrupt() returns the answer.
    decision = validate_decision(candidates, interrupt(approval_request(state)))
    update: dict = {"approval": decision, "download_result": None,
                    "workflow_log": [f"human_approval: {decision['action']}"
                                     + (f" {len(decision['selected'])} of {len(candidates)}" if decision["selected"] else "")]}
    if decision["action"] in ("reject", "retry"):
        # Not proposed again in later rounds
        update["rejected_asset_ids"] = list(dict.fromkeys(
            [*(state.get("rejected_asset_ids") or []), *(c["assetId"] for c in candidates)]))
    # approve -> asset_download; reject/retry -> furniture_research, which decides how to continue.
    return update
