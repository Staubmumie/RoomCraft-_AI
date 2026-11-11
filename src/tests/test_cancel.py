"""Aborting a run through the API: it ends exactly once with workflow.cancelled, nothing is applied
and nothing is kept for a later resume (bookkeeping and LangGraph checkpoint are discarded)."""
import asyncio

import design.interior_designer as designer_module
import design_api
from design.graph import design_graph
from design.state import Furniture, Room
from tests.conftest import decide
from tests.research_fakes import install_research

ROOM = Room(width=8, depth=6, height=2.8)
FURNITURE = [Furniture(id="a", name="Sofa", width=1, depth=1, x=-2, z=0)]


def _body(request="Make it cozy") -> design_api.DesignRequest:
    return design_api.DesignRequest(request=request, room=ROOM, furniture=FURNITURE)


def _types(run) -> list[str]:
    return [e["type"] for e in run.events]


def _discarded(run) -> bool:
    checkpoint = design_graph.checkpointer.get_tuple({"configurable": {"thread_id": run.id}})
    return run.context == {} and run.pending_approval is None and checkpoint is None


async def _settle(run):
    """Lets the task and its done callback finish."""
    if run.task:
        await asyncio.gather(run.task, return_exceptions=True)
    await asyncio.sleep(0)


async def _until(predicate, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def test_abort_during_a_running_node(fake_llm, monkeypatch):
    fake_llm(decide())

    async def hang(*args, **kwargs):
        await asyncio.sleep(3600)  # an LLM call that never answers
    monkeypatch.setattr(designer_module, "structured_completion", hang)

    async def scenario():
        run_id = (await design_api.start_run(_body()))["runId"]
        run = design_api.runs.get(run_id)
        await _until(lambda: any(e["type"] == "node.started" and e["node"] == "interior_designer" for e in run.events))
        first = await design_api.cancel_run(run_id)
        second = await design_api.cancel_run(run_id)  # double click
        await _settle(run)
        third = await design_api.cancel_run(run_id)  # after the end
        return run, first, second, third

    run, first, second, third = asyncio.run(scenario())
    assert first["status"] == "cancelling" and second["status"] == "cancelling" and third["status"] == "cancelled"
    assert _types(run).count("workflow.cancelled") == 1 and _types(run)[-1] == "workflow.cancelled"
    assert run.result is None and _discarded(run)
    designer = [e for e in run.events if e["node"] == "interior_designer"]
    assert designer[-1]["type"] == "node.failed"  # the open execution is closed


def test_abort_while_waiting_for_approval(fake_llm, monkeypatch):
    fake_llm(intent="SEARCH_FURNITURE")
    research = install_research(monkeypatch)

    async def scenario():
        run_id = (await design_api.start_run(_body("Find a modern wooden cabinet.")))["runId"]
        run = design_api.runs.get(run_id)
        await _settle(run)
        assert run.status == "waiting_for_approval"
        await design_api.cancel_run(run_id)
        await design_api.cancel_run(run_id)
        return run

    run = asyncio.run(scenario())
    assert _types(run).count("workflow.cancelled") == 1
    assert run.result is None and _discarded(run) and research.downloads == []


def test_abort_before_the_task_started(fake_llm):
    fake_llm(decide())

    async def scenario():
        run_id = (await design_api.start_run(_body()))["runId"]
        await design_api.cancel_run(run_id)  # the task has not run a single step yet
        run = design_api.runs.get(run_id)
        await _settle(run)
        return run

    run = asyncio.run(scenario())
    assert _types(run) == ["workflow.cancelled"]
    assert run.status == "cancelled" and _discarded(run)


def test_abort_while_resuming(fake_llm, monkeypatch):
    fake_llm(intent="SEARCH_FURNITURE")
    research = install_research(monkeypatch)

    async def scenario():
        run_id = (await design_api.start_run(_body("Find a modern wooden cabinet.")))["runId"]
        run = design_api.runs.get(run_id)
        await _settle(run)
        await design_api.answer_approval(run_id, design_api.ApprovalDecision(
            action="approve", selected=["polyhaven:oak_cabinet"]))
        assert run.status == "resuming"
        await design_api.cancel_run(run_id)
        await _settle(run)
        return run

    run = asyncio.run(scenario())
    assert _types(run).count("workflow.cancelled") == 1 and run.result is None
    assert research.downloads == [] and _discarded(run)
