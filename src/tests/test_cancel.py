"""Aborting a run through the API: it ends exactly once with workflow.cancelled, nothing is applied
and nothing is kept for a later resume (bookkeeping and LangGraph checkpoint are discarded)."""
import asyncio

import design.interior_designer as designer_module
import design_api
import research.jobs as research_jobs
from design.graph import design_graph
from design.runner import execute_design_run, resume_design_run
from design.state import Furniture, Room
from observability.store import RunStore
from research.human_approval import validate_decision
from tests.conftest import decide
from tests.research_fakes import found, install_research

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


def test_a_finished_run_keeps_no_checkpoint(fake_llm):
    """Only paused runs need their LangGraph checkpoint; a completed run must not keep it in memory."""
    fake_llm(decide())

    async def scenario():
        run = design_api.runs.get((await design_api.start_run(_body()))["runId"])
        await _settle(run)
        return run

    run = asyncio.run(scenario())
    assert _types(run)[-1] == "workflow.completed" and run.result is not None
    assert design_graph.checkpointer.get_tuple({"configurable": {"thread_id": run.id}}) is None


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
            itemKey=run.pending_approval["itemKey"], action="approve", selected=["polyhaven:oak_cabinet"]))
        assert run.status == "resuming"
        await design_api.cancel_run(run_id)
        await _settle(run)
        return run

    run = asyncio.run(scenario())
    assert _types(run).count("workflow.cancelled") == 1 and run.result is None
    assert research.downloads == [] and _discarded(run)


# --- furniture lists: background searches and downloads ----------------------------------------------

LIST = ["a grey sofa", "a floor lamp", "a wool rug"]


def _list_research(fake_llm, monkeypatch, items=LIST, **by_item):
    fake_llm(intent="SEARCH_FURNITURE", furniture_items=items)  # before the research fakes (it blocks the network)
    return install_research(monkeypatch, by_item=by_item or {
        "sofa": [*found("grey sofa", "polyhaven:grey_sofa"), *found("corner sofa", "polyhaven:corner_sofa")],
        "lamp": found("floor lamp", "polyhaven:floor_lamp"), "rug": found("wool rug", "polyhaven:wool_rug")})


def _no_job_left(run) -> bool:
    return not research_jobs.active(run.id) and run.id not in research_jobs._runs


def _approve(run, *selected, action="approve"):
    return design_api.ApprovalDecision(itemKey=run.pending_approval["itemKey"], action=action, selected=list(selected))


def test_abort_during_parallel_searches(fake_llm, monkeypatch):
    research = _list_research(fake_llm, monkeypatch)
    for word in ("sofa", "lamp", "rug"):
        research.agent.hold[word] = asyncio.Event()  # every search hangs in its LLM call

    async def scenario():
        run_id = (await design_api.start_run(_body("A sofa, a lamp and a rug")))["runId"]
        run = design_api.runs.get(run_id)
        await _until(lambda: len(research.agent.calls_by_item) == 3)  # three searches at once
        jobs = research_jobs.for_run(run_id)
        tasks = [job.task for job in jobs.jobs.values() if not job.task.done()]
        assert len(tasks) == 3
        await design_api.cancel_run(run_id)
        await _settle(run)
        for word in ("sofa", "lamp", "rug"):  # a late answer must not revive anything
            research.agent.hold[word].set()
        await asyncio.sleep(0.05)
        return run, tasks

    run, tasks = asyncio.run(scenario())
    assert _types(run).count("workflow.cancelled") == 1 and _types(run)[-1] == "workflow.cancelled"
    assert all(t.cancelled() for t in tasks) and _no_job_left(run)
    assert research.downloads == [] and _discarded(run)


def test_abort_during_a_second_search_run(fake_llm, monkeypatch):
    research = _list_research(fake_llm, monkeypatch, ["a grey sofa"],
                              sofa=[*found("grey sofa", "polyhaven:grey_sofa"), *found("corner sofa", "polyhaven:corner_sofa")])

    async def scenario():
        run_id = (await design_api.start_run(_body("A grey sofa")))["runId"]
        run = design_api.runs.get(run_id)
        await _settle(run)
        research.agent.hold["sofa"] = asyncio.Event()  # the second run hangs
        await design_api.answer_approval(run_id, _approve(run, action="reject"))
        await _until(lambda: research.agent.calls_by_item.get("sofa") == 3)
        await design_api.cancel_run(run_id)
        await _settle(run)
        research.agent.hold["sofa"].set()
        await asyncio.sleep(0.05)
        return run

    run = asyncio.run(scenario())
    assert _types(run).count("workflow.cancelled") == 1 and run.result is None
    assert research.downloads == [] and _no_job_left(run) and _discarded(run)


def test_abort_with_one_download_done_and_one_running(fake_llm, monkeypatch):
    """The finished download stays; the running one is cancelled; no new download starts."""
    research = _list_research(fake_llm, monkeypatch)
    research.download_hold["polyhaven:floor_lamp"] = asyncio.Event()
    research.agent.hold["rug"] = asyncio.Event()

    async def scenario():
        run_id = (await design_api.start_run(_body("A sofa, a lamp and a rug")))["runId"]
        run = design_api.runs.get(run_id)
        approved = set()
        while True:
            await _until(lambda: run.status == "waiting_for_approval")
            pending = run.pending_approval
            if pending["request"]["description"] == "a wool rug":
                break
            await design_api.answer_approval(run_id, _approve(run, pending["candidates"][0]["assetId"]))
            approved.add(pending["request"]["description"])
            if approved == {"a grey sofa", "a floor lamp"}:
                research.agent.hold["rug"].set()  # the rug only becomes ready after both approvals
        # sofa downloaded, lamp still downloading, the rug waits for approval
        jobs = research_jobs.for_run(run_id)
        await _until(lambda: jobs.state("download:polyhaven:grey_sofa") == "done")
        assert jobs.state("download:polyhaven:floor_lamp") == "running"
        lamp = jobs.jobs["download:polyhaven:floor_lamp"].task
        await design_api.cancel_run(run_id)
        await _settle(run)
        research.download_hold["polyhaven:floor_lamp"].set()
        await asyncio.sleep(0.05)
        return run, lamp

    run, lamp = asyncio.run(scenario())
    assert _types(run).count("workflow.cancelled") == 1 and run.result is None
    assert lamp.cancelled() and _no_job_left(run)
    assert sorted(research.downloads) == ["polyhaven:floor_lamp", "polyhaven:grey_sofa"]  # nothing new, no rug


def test_resume_in_a_new_event_loop_restarts_lost_searches_once(fake_llm, monkeypatch):
    """A search that was lost with its event loop while the run waited is started again exactly once
    under the same key; approved downloads are not repeated."""
    research = _list_research(fake_llm, monkeypatch, ["a grey sofa", "a floor lamp"],
                              sofa=found("grey sofa", "polyhaven:grey_sofa"), lamp=found("floor lamp", "polyhaven:floor_lamp"))
    research.agent.hold["lamp"] = asyncio.Event()
    store = RunStore()
    run = store.create("design")
    asyncio.run(execute_design_run(store, run, {"request": "A sofa and a lamp", "model": "fake", "room": ROOM,
                                                "furniture": FURNITURE, "options": {}}))
    assert run.pending_approval["request"]["description"] == "a grey sofa"
    jobs = research_jobs.for_run(run.id)
    lamp_keys = [k for k in jobs.jobs if k.split(":")[1].endswith("-2")]  # local check or online search
    assert jobs.state(lamp_keys[-1]) == "lost"  # its event loop is gone
    del research.agent.hold["lamp"]
    decision = validate_decision(run.pending_approval, {"itemKey": run.pending_approval["itemKey"],
                                                        "action": "approve", "selected": ["polyhaven:grey_sofa"]})
    asyncio.run(resume_design_run(store, run, decision))
    assert run.pending_approval["request"]["description"] == "a floor lamp"
    restarts = [e for e in run.events if e["type"] == "node.progress" and "(restarted)" in e["payload"]["label"]]
    assert len(restarts) == 1
    decision = validate_decision(run.pending_approval, {"itemKey": run.pending_approval["itemKey"],
                                                        "action": "approve", "selected": ["polyhaven:floor_lamp"]})
    asyncio.run(resume_design_run(store, run, decision))
    assert run.events[-1]["type"] == "workflow.completed"
    assert research.downloads == ["polyhaven:grey_sofa", "polyhaven:floor_lamp"]  # each exactly once
    assert research.agent.calls_by_item["sofa"] == 2
    # 2 calls after the restart, plus the call lost with the loop if it was already in its online search
    assert research.agent.calls_by_item["lamp"] in (2, 3)
