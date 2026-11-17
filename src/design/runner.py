"""Runs the RoomCraft graph and turns LangGraph's stream into observability events.

- `tasks` stream  -> node.started / node.completed / node.failed / node.waiting / node.resumed
                     (keyed by the LangGraph task id)
- `custom` stream -> node.progress (stages), node.llm (LLM calls), node.tool (MCP tools) from inside nodes
State changes are taken from each task's result rather than the `updates` stream, because task
results carry the execution id and stay unambiguous for repeated nodes. How a run ends is decided by
the nodes (state["outcome"]); the runner only reports it.
"""
import asyncio
import math
import time
from typing import get_type_hints

from langgraph.types import Command

from design.graph import design_graph
from design.routing import next_node
from design.state import DesignState, Furniture
from llm import LLMError
from observability.store import Run, RunStore
from rendering.service import RenderError
from research import jobs as research_jobs
from research.mcp_client import AssetMCPError

CUSTOM_EVENT_TYPES = {"stage": "node.progress", "llm": "node.llm", "tool": "node.tool"}
# The reducers declared on DesignState (Annotated[type, reducer]), so the runner's view of the
# state can never drift from LangGraph's.
REDUCERS = {key: hint.__metadata__[0] for key, hint in get_type_hints(DesignState, include_extras=True).items()
            if getattr(hint, "__metadata__", None)}


def apply_update(state: dict, update: dict) -> None:
    """Applies a node's update like LangGraph does (state reducers, e.g. counters and the log)."""
    for key, value in update.items():
        state[key] = REDUCERS[key](state.get(key) or type(value)(), value) if key in REDUCERS else value


def _position(x: float, y: float, z: float, rotation: float) -> dict:
    return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "rotationDeg": round(math.degrees(rotation)) % 360}


def describe_changes(state: dict, update: dict) -> dict:
    """Compact summary of what a node changed: changed keys plus a furniture position diff
    against the confirmed room."""
    changes: dict = {"keys": sorted(update)}
    placements = update.get("layout") or update.get("proposed_layout")
    if placements is not None and ("layout" in update or "proposed_layout" in update):
        furniture: dict[str, Furniture] = {f.id: f for f in state.get("furniture", [])}
        present = {p.id for p in placements}
        changes["added"] = [p.name for p in placements if p.id not in furniture]
        changes["removed"] = [f.name for f in furniture.values() if f.id not in present]
        layout = []
        for placement in placements:
            item = furniture.get(placement.id)
            if not item:
                continue
            before = _position(item.x, item.y, item.z, item.rotation)
            after = _position(placement.x, placement.y, placement.z, placement.rotation)
            layout.append({
                "id": placement.id,
                "name": item.name,
                "before": before,
                "after": after,
                "moved": (before["x"], before["y"], before["z"]) != (after["x"], after["y"], after["z"]),
                "rotated": before["rotationDeg"] != after["rotationDeg"],
            })
        changes["layout"] = layout
    return changes


def summarize(node: str, update: dict) -> str:
    """One line for the timeline."""
    if node == "orchestrator":
        return f"Intent: {update.get('intent')}" + (" (blocked)" if update.get("security_flags") else "")
    if node == "render_scene":
        folder = next(iter(update.get("scene_images", {}).values()), "").rsplit("/", 2)[-2:-1]
        return (f"{len(update.get('scene_images', {}))} screenshots of layout v{update.get('rendered_layout_version', 0)}"
                + (f" · {folder[0]}" if folder else ""))
    if node == "interior_designer":
        action = update.get("designer_action", "?")
        outcome = update.get("outcome")
        if outcome:
            return f"{outcome['status']}: finish"
        moved = f"v{update['layout_version']} · " if "layout_version" in update else ""
        return f"{moved}next: {action}"
    if node == "layout_validator":
        v = update["validation"]
        text = f"v{update.get('validated_layout_version')} {v.status}"
        if v.stats.corrections:
            text += f" · {v.stats.corrections} correction(s)"
        if v.errors:
            text += f" · {len(v.errors)} error(s)"
        if update.get("render_after_validation"):
            text += " · new screenshots"
        return text
    if node == "interior_critic":
        if update.get("critic_action") == "request_render":
            return "requested a new render"
        critique = update.get("critique") or {}
        return f"v{update.get('critiqued_layout_version')} · {len(critique.get('issues', []))} issue(s)"
    if node == "furniture_research":
        items = update.get("research_items") or []
        status = update.get("research_status")
        if status == "awaiting_approval":
            item = next((i for i in items if i["key"] == update.get("approval_item")), None) or {}
            return (f"item {item.get('index')}/{len(items)} · run {item.get('run')} · "
                    f"{len(item.get('candidates') or [])} suggestion(s)")
        if status == "download":
            return f"{sum(e['status'] == 'queued' for i in items for e in i['downloads'])} download(s) to start"
        return f"finished: {(update.get('research_result') or {}).get('status', '?')} · {len(items)} item(s)"
    if node == "human_approval":
        approval = update["approval"]
        return {"approve": f"approved {len(approval['selected'])} asset(s)",
                "reject": "nothing fits", "skip": "skipped"}[approval["action"]]
    if node == "asset_download":
        running = sum(e["status"] == "running" for i in update.get("research_items") or [] for e in i["downloads"])
        return f"{running} download(s) running"
    return ""


def _downloads(state: dict, added: list[dict]) -> dict | None:
    """What the run added to the inventory. A research request reports its own result; a design run
    reports every asset downloaded by its delegated searches (research_result only keeps the last one),
    so the UI knows to reload the inventory before applying the layout."""
    outcome = state.get("outcome") or {}
    if outcome.get("kind") == "research":
        return state.get("research_result")
    return {"added": added} if added else None


def _result(state: dict, added: list[dict] | None = None) -> dict:
    outcome = state.get("outcome") or {}
    return {
        "intent": state.get("intent"),
        "kind": outcome.get("kind"),
        "message": outcome.get("message"),
        "explanation": state.get("explanation", ""),
        # All pieces of the validated layout (new ones have ids "new-..." and an inventory assetId)
        # plus the confirmed pieces it no longer contains.
        "placements": state.get("layout", []) if outcome.get("kind") == "design" else [],
        "removed": state.get("removed", []) if outcome.get("kind") == "design" else [],
        "warnings": state.get("warnings", []),
        # Essential pieces the design still lacks (also listed in warnings); the UI offers a search for each.
        "missing": [g for g in state.get("furniture_gaps") or [] if g.get("essential")]
        if outcome.get("kind") == "design" else [],
        "validation": state.get("validation"),
        "critique": state.get("critique"),
        "downloads": _downloads(state, added or []),
        "steps": (state.get("counters") or {}).get("designer_steps", 0),
        "model": state["model"],
        "models": state.get("models") or {},
    }


def _ms(since: float) -> int:
    return round((time.perf_counter() - since) * 1000)


RECURSION_LIMIT = 200


def graph_config(run: Run) -> dict:
    """The run id is the LangGraph thread: a paused run resumes exactly where it stopped. Every step
    is bounded by the limits in design.routing; the recursion limit only has to leave room for them
    (a furniture list adds coordinator/download steps per item)."""
    return {"configurable": {"thread_id": run.id}, "recursion_limit": RECURSION_LIMIT}


async def execute_design_run(store: RunStore, run: Run, inputs: DesignState):
    # Bookkeeping that must survive a pause (the resume is a separate call).
    run.context.update(started=time.perf_counter(), wait_ms=0, executions={}, state=dict(inputs), downloaded=[])
    store.emit(run, "workflow.started", status="running", payload={
        "model": inputs["model"],
        "models": inputs.get("models") or {},
        "request": inputs["request"],
        "furnitureCount": len(inputs["furniture"]),
        "input": inputs,
    })
    await _drive(store, run, inputs)


async def resume_design_run(store: RunStore, run: Run, decision: dict):
    """Continues a run paused for approval with the user's decision."""
    waited = _ms(run.context.pop("wait_started"))
    run.context["wait_ms"] += waited
    store.emit(run, "workflow.resumed", status="running", payload={"decision": decision, "waitMs": waited})
    await _drive(store, run, Command(resume=decision))


async def _drive(store: RunStore, run: Run, graph_input):
    ctx = run.context
    executions: dict[str, dict] = ctx["executions"]  # task id -> {"node", "started", "activeMs", ...}
    state: dict = ctx["state"]
    timing = lambda: {"durationMs": _ms(ctx["started"]), "waitMs": ctx["wait_ms"],
                      "activeMs": _ms(ctx["started"]) - ctx["wait_ms"]}
    try:
        async for mode, chunk in design_graph.astream(graph_input, graph_config(run), stream_mode=["tasks", "custom"]):
            if mode == "tasks" and "input" in chunk:
                execution = executions.get(chunk["id"])
                if execution and execution.get("waiting"):
                    # LangGraph re-runs the interrupted task with the same id when resuming.
                    waited = _ms(execution.pop("wait_started"))
                    execution.update(waiting=False, started=time.perf_counter())
                    execution["waitMs"] = execution.get("waitMs", 0) + waited
                    store.emit(run, "node.resumed", node=chunk["name"], execution_id=chunk["id"], status="running",
                               payload={"waitMs": waited})
                else:
                    executions[chunk["id"]] = {"node": chunk["name"], "started": time.perf_counter(), "activeMs": 0}
                    store.emit(run, "node.started", node=chunk["name"], execution_id=chunk["id"], status="running",
                               payload={"input": chunk["input"], "triggers": list(chunk.get("triggers") or [])})
            elif mode == "tasks":
                execution = executions.setdefault(chunk["id"], {"node": chunk["name"], "started": ctx["started"], "activeMs": 0})
                active = execution["activeMs"] + _ms(execution["started"])
                if chunk.get("interrupts"):
                    execution.update(waiting=True, wait_started=time.perf_counter(), activeMs=active)
                    store.emit(run, "node.waiting", node=chunk["name"], execution_id=chunk["id"], status="waiting",
                               payload={"activeMs": active, "request": [i["value"] for i in chunk["interrupts"]]})
                    continue
                timings = {"durationMs": active, "waitMs": execution.get("waitMs", 0)}
                if chunk.get("error") is not None:
                    error = chunk["error"]
                    store.emit(run, "node.failed", node=chunk["name"], execution_id=chunk["id"], status="failed",
                               payload={**timings, "error": str(error), "errorType": type(error).__name__})
                else:
                    update = chunk.get("result") or {}
                    changes = describe_changes(state, update)
                    apply_update(state, update)
                    # Downloads finish in the background; the research coordinator reports them as they arrive.
                    for added in (update.get("download_result") or {}).get("added", []):
                        if added["assetId"] not in {a["assetId"] for a in ctx["downloaded"]}:
                            ctx["downloaded"].append(added)
                    store.emit(run, "node.completed", node=chunk["name"], execution_id=chunk["id"], status="completed",
                               payload={**timings, "output": update, "summary": summarize(chunk["name"], update),
                                        "changes": changes, "next": next_node(chunk["name"], state)})
                execution["done"] = True
            elif mode == "custom":
                data = dict(chunk)
                kind = data.pop("kind", "custom")
                execution_id = data.pop("executionId", None)
                node = executions.get(execution_id, {}).get("node")
                store.emit(run, CUSTOM_EVENT_TYPES.get(kind, "node.custom"), node=node, execution_id=execution_id,
                           status=data.get("status"), payload=data)

        snapshot = await design_graph.aget_state(graph_config(run))
        if snapshot.interrupts:
            # Paused for the user. Not terminal: the event stream stays open until the run is resumed.
            run.pending_approval = snapshot.interrupts[0].value
            ctx["wait_started"] = time.perf_counter()
            store.emit(run, "workflow.waiting", status="waiting_for_approval",
                       payload={**timing(), "approval": run.pending_approval})
            return  # background research jobs keep running during the pause

        research_jobs.close(run.id)  # nothing may outlive the run
        final = snapshot.values
        _forget_checkpoint(run.id)  # the run is over: its result and events live on the Run
        outcome = final.get("outcome") or {"status": "failed", "message": "The workflow ended without a result."}
        if outcome["status"] == "failed":
            store.emit(run, "workflow.failed", status="failed", payload={
                **timing(), "error": outcome["message"], "statusCode": 422, "validation": final.get("validation"),
            })
            return
        # completed or rejected (a rejected request is answered, not an error)
        run.result = _result(final, ctx.get("downloaded"))
        store.emit(run, "workflow.completed", status="completed", payload={**timing(), "result": run.result})
    except asyncio.CancelledError:
        research_jobs.close(run.id)  # no new searches or downloads after an abort
        if not run.finished:
            _close_open_executions(store, run, executions, "Cancelled")
            store.emit(run, "workflow.cancelled", status="cancelled", payload=timing())
        raise
    except Exception as exc:
        research_jobs.close(run.id)
        _forget_checkpoint(run.id)
        error_code = None
        if isinstance(exc, LLMError):
            message, status_code = str(exc), exc.status_code
        elif isinstance(exc, RenderError):
            message, error_code = f"Rendering failed ({exc.code}): {exc}", exc.code
            status_code = 422 if exc.code in ("invalid_room", "missing_asset") else 502
        elif isinstance(exc, AssetMCPError):
            message, error_code, status_code = f"Asset search failed: {exc}", "mcp_unavailable", 502
        else:
            message, status_code = f"Unexpected error: {type(exc).__name__}: {exc}", 500
        _close_open_executions(store, run, executions, message)
        store.emit(run, "workflow.failed", status="failed",
                   payload={**timing(), "error": message, "statusCode": status_code, "errorCode": error_code})


def cancel_waiting_run(store: RunStore, run: Run):
    """Ends a run that is paused for approval (there is no task to cancel)."""
    if run.finished:
        return
    run.pending_approval = None
    _close_open_executions(store, run, run.context.get("executions", {}), "Cancelled while waiting for approval")
    store.emit(run, "workflow.cancelled", status="cancelled", payload=_cancel_timing(run))
    discard_run(run)


def settle_cancelled_task(store: RunStore, run: Run):
    """Called when a run's task ended by cancellation. A task cancelled before it started never
    reached its own handler: emit the terminal event here, so every run ends exactly once."""
    if not run.finished:
        _close_open_executions(store, run, run.context.get("executions", {}), "Cancelled")
        store.emit(run, "workflow.cancelled", status="cancelled", payload=_cancel_timing(run))
    discard_run(run)


def discard_run(run: Run):
    """A cancelled run is never resumed: forget its bookkeeping and its LangGraph checkpoint.
    The events stay, so the run remains visible as cancelled. Its background research jobs are
    cancelled: no new downloads start, finished downloads stay in the inventory."""
    research_jobs.close(run.id)
    run.pending_approval = None
    run.context.clear()
    _forget_checkpoint(run.id)


def _forget_checkpoint(run_id: str) -> None:
    """Drops the LangGraph checkpoint of a run that can no longer resume (finished, failed or
    cancelled); the in-memory saver would otherwise keep every run's state for the process lifetime."""
    try:
        design_graph.checkpointer.delete_thread(run_id)
    except Exception:  # a checkpointer without delete support only keeps a little memory
        pass


def _cancel_timing(run: Run) -> dict:
    """Timing for a cancellation outside _drive, including a pause that is still going on."""
    ctx = run.context
    if "started" not in ctx:
        return {"durationMs": 0, "waitMs": 0}
    waited = ctx.get("wait_ms", 0) + (_ms(ctx["wait_started"]) if "wait_started" in ctx else 0)
    return {"durationMs": _ms(ctx["started"]), "waitMs": waited}


def _close_open_executions(store, run, executions, reason):
    for execution_id, execution in executions.items():
        if not execution.get("done"):
            execution["done"] = True
            store.emit(run, "node.failed", node=execution["node"], execution_id=execution_id, status="failed",
                       payload={"durationMs": execution.get("activeMs", 0) + (0 if execution.get("waiting") else _ms(execution["started"])),
                                "error": reason})
