"""Furniture research as a list: per-item search, approval of one item at a time, background
downloads and the search settings, with fake MCP, LLM, previews and downloads."""
import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException

import design_api
import inventory
import research.agent as agent
import research.jobs as research_jobs
import research.previews as previews
from design.routing import next_node
from design.runner import cancel_waiting_run, execute_design_run, resume_design_run
from design.state import Furniture, Room
from furniture.pipeline import unit_scale
from observability.store import RunStore
from research.asset_download import asset_download, download_authorized
from research.human_approval import ApprovalError, validate_decision
from research.mcp_client import AssetMCPError, parse_tool_result
from tests.research_fakes import ROUND_1, ROUND_2, FakeMCP, call, found, install_research
from url_safety import UnsafeURLError, public_http_client

ROOM = Room(width=8, depth=6, height=2.8)
FURNITURE = [Furniture(id="sofa", name="Sofa", width=2, depth=0.9, x=0, z=-2.5)]
REQUEST = "Find a modern wooden cabinet that fits my living room."
GUIDED, AUTOMATIC, LOCAL_ONLY = ({}, {"review_suggestions": False}, {"search_online": False})


@pytest.fixture
def research(fake_llm, monkeypatch):
    """Fakes for MCP, LLM tool calls, previews and downloads (see tests/research_fakes.py)."""
    fake_llm(intent="SEARCH_FURNITURE")
    return install_research(monkeypatch)


def inputs(request=REQUEST, options=None):
    return {"request": request, "model": "fake", "room": ROOM, "furniture": FURNITURE, "options": options or {}}


def start(request=REQUEST, options=None):
    store = RunStore()
    run = store.create("design")
    asyncio.run(execute_design_run(store, run, inputs(request, options)))
    return store, run


def decision(run, **values):
    return validate_decision(run.pending_approval, {"itemKey": run.pending_approval["itemKey"], **values})


def resume(store, run, **values):
    asyncio.run(resume_design_run(store, run, decision(run, **values)))


def started(run):
    return [e["node"] for e in run.events if e["type"] == "node.started"]


def events(run, type_):
    return [e for e in run.events if e["type"] == type_]


def items(run):
    return {i["description"]: i for i in run.result["downloads"]["items"]}


async def until(predicate, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


# --- research and pause ----------------------------------------------------------------------

def test_search_request_goes_to_research_and_pauses_for_approval(research):
    store, run = start()
    assert started(run) == ["orchestrator", "furniture_research", "human_approval"]
    assert run.events[-1]["type"] == "workflow.waiting"
    assert run.status == "waiting_for_approval"
    assert research.downloads == []  # nothing downloaded before approval
    pending = run.pending_approval
    assert pending["index"] == 1 and pending["total"] == 1 and pending["run"] == 1 and pending["itemKey"]
    assert pending["request"]["description"] == REQUEST and pending["intent"] is not None


def test_shortlist_contains_only_found_loadable_assets_with_metadata(research):
    _, run = start()
    candidates = run.pending_approval["candidates"]
    assert [c["assetId"] for c in candidates] == ["polyhaven:oak_cabinet", "blenderkit:walnut_cabinet"]
    oak = candidates[0]
    assert oak["thumbnailUrl"] == "https://cdn.example.com/oak.png"
    assert oak["license"] == {"name": "CC0"} and oak["formats"] == ["gltf", "blend"]
    assert oak["description"].startswith("A modern oak cabinet")
    assert oak["reason"] and oak["previewInspected"] is True
    assert candidates[1]["previewInspected"] is False
    assert "base64" not in json.dumps(run.pending_approval)  # only image URLs, never image data


def test_at_most_four_suggestions_per_item_and_run(research):
    ids = ["polyhaven:grey_sofa", "polyhaven:green_sofa", "polyhaven:corner_sofa", "polyhaven:oak_cabinet",
           "blenderkit:walnut_cabinet", "grab3d:black_cabinet"]
    research.agent.turns = [[call("search_assets", query="sofa cabinet")],
                            [call("submit_candidates", summary="Many.", candidates=[
                                {"asset_id": a, "reason": "r", "suitable": True} for a in ids])]]
    _, run = start()
    assert len(run.pending_approval["candidates"]) == agent.MAX_SUGGESTIONS_PER_ITEM == 4


def test_candidates_the_agent_marks_unsuitable_are_not_shown(research):
    research.agent.turns = [[call("search_assets", query="cabinet")],
                            [call("submit_candidates", summary="s", candidates=[
                                {"asset_id": "polyhaven:oak_cabinet", "reason": "r", "suitable": False},
                                {"asset_id": "grab3d:black_cabinet", "reason": "r", "suitable": True}])]]
    _, run = start()
    assert [c["assetId"] for c in run.pending_approval["candidates"]] == ["grab3d:black_cabinet"]


def test_search_results_are_filtered_before_the_agent_sees_them(research):
    start()
    search_result = json.loads(next(m for m in research.agent.requests[1] if m["role"] == "tool")["content"])
    ids = [r["id"] for r in search_result["results"]]
    assert "opengameart:cabinet_scene" not in ids and "polyhaven:fbx_cabinet" not in ids
    assert search_result["filteredOut"] == 2


def test_asset_details_and_preview_images_reach_the_agent(research):
    start()
    third_call = research.agent.requests[2]
    tool_texts = [m["content"] for m in third_call if m["role"] == "tool"]
    details = json.loads(tool_texts[1])
    assert details["description"].startswith("A modern oak cabinet") and details["loadableInApp"]
    assert "no preview image" in tool_texts[3]  # dark cabinet has none: noted, not rejected
    image_message = third_call[-1]
    assert image_message["role"] == "user"
    assert any(p.get("type") == "image_url" for p in image_message["content"])
    assert research.previews == ["https://cdn.example.com/oak.png"]


def test_model_without_vision_works_from_metadata(research):
    research.vision = False
    _, run = start()
    assert research.previews == []
    assert "cannot see images" in research.agent.requests[0][1]["content"]
    tool_texts = [m["content"] for m in research.agent.requests[2] if m["role"] == "tool"]
    assert "cannot see images" in tool_texts[2]
    assert run.status == "waiting_for_approval"


def test_llm_steps_per_run_are_limited_and_the_last_one_must_submit(research):
    research.agent.turns = [[call("search_assets", query="cabinet")]] * 5  # never submits on its own
    start()
    first_run = [r for r in research.agent.requests if "run 2" not in r[1]["content"]]
    assert len(first_run) == agent.MAX_LLM_STEPS_PER_ITEM == 3


# --- approval ----------------------------------------------------------------------------------

def test_approve_downloads_only_selected_assets(research):
    store, run = start()
    resume(store, run, action="approve", selected=["polyhaven:oak_cabinet"])
    assert research.downloads == ["polyhaven:oak_cabinet"]
    assert run.events[-1]["type"] == "workflow.completed"
    result = run.result
    assert [a["assetId"] for a in result["downloads"]["added"]] == ["polyhaven:oak_cabinet"]
    assert result["downloads"]["added"][0]["unitScale"] == 0.01  # 180 units high -> centimetres
    assert research.updates[0][1]["description"].startswith("A modern oak cabinet")  # research metadata reaches enrichment
    assert "Oak Cabinet" in result["message"]
    # the approval starts the download; the coordinator collects it and ends the list
    assert started(run) == ["orchestrator", "furniture_research", "human_approval", "asset_download", "furniture_research"]
    assert run.result["downloads"]["status"] == "completed"
    assert items(run)[REQUEST]["status"] == "completed"
    assert [e["node"] for e in events(run, "node.resumed")] == ["human_approval"]


def test_reject_in_the_first_run_searches_again_without_the_rejected_assets(research):
    store, run = start()
    resume(store, run, action="reject", feedback="Show me darker wooden furniture.")
    assert research.downloads == []
    assert run.status == "waiting_for_approval"
    prompt = research.agent.requests[3][1]["content"]
    assert "Show me darker wooden furniture." in prompt and "run 2 of 2" in prompt and "other terms" in prompt
    candidates = [c["assetId"] for c in run.pending_approval["candidates"]]
    assert candidates == ["grab3d:dark_cabinet", "grab3d:black_cabinet"]  # oak and walnut were excluded
    assert run.pending_approval["run"] == 2

    resume(store, run, action="approve", selected=["grab3d:dark_cabinet", "grab3d:black_cabinet"])
    assert research.downloads == ["grab3d:dark_cabinet", "grab3d:black_cabinet"]
    nodes = started(run)
    assert nodes.count("human_approval") == 2
    ids = [e["executionId"] for e in events(run, "node.started")]
    assert len(ids) == len(set(ids))


def test_reject_in_the_second_run_ends_the_item_without_a_third_run(research):
    store, run = start()
    resume(store, run, action="reject")
    resume(store, run, action="reject", feedback="still not")
    assert run.events[-1]["type"] == "workflow.completed"
    assert research.downloads == []
    assert items(run)[REQUEST]["status"] == "rejected" and items(run)[REQUEST]["runs"] == 2
    assert run.result["downloads"]["status"] == "rejected" and "rejected" in run.result["message"]
    assert not any("run 3" in r[1]["content"] for r in research.agent.requests)
    assert len(research.agent.requests) == len(ROUND_1) + len(ROUND_2)


def test_skip_ends_the_item_immediately(research):
    store, run = start()
    resume(store, run, action="skip")
    assert run.events[-1]["type"] == "workflow.completed"
    assert items(run)[REQUEST]["status"] == "skipped" and items(run)[REQUEST]["runs"] == 1
    assert len(research.agent.requests) == len(ROUND_1) and research.downloads == []


def test_nothing_found_in_two_runs_is_reported_as_not_found(research):
    research.agent.turns = [[call("submit_candidates", summary="Nothing.", candidates=[])],
                            [call("submit_candidates", summary="Still nothing.", candidates=[])]]
    _, run = start()
    assert run.events[-1]["type"] == "workflow.completed"
    assert items(run)[REQUEST]["status"] == "not_found" and items(run)[REQUEST]["runs"] == 2
    assert run.result["downloads"]["status"] == "no_results" and research.downloads == []


def test_selection_must_come_from_the_item_and_its_run():
    request = {"itemKey": "b1-2", "run": 1, "candidates": [{"assetId": "a"}, {"assetId": "b"}]}
    with pytest.raises(ApprovalError):
        validate_decision(request, {"itemKey": "b1-2", "action": "approve", "selected": ["a", "zzz"]})
    with pytest.raises(ApprovalError):
        validate_decision(request, {"itemKey": "b1-2", "action": "approve", "selected": []})
    with pytest.raises(ApprovalError):
        validate_decision(request, {"itemKey": "b1-2", "action": "delete"})
    with pytest.raises(ApprovalError, match="another furniture item"):  # stale: another item
        validate_decision(request, {"itemKey": "b1-1", "action": "approve", "selected": ["a"]})
    with pytest.raises(ApprovalError, match="earlier search run"):  # stale: older run of the item
        validate_decision(request, {"itemKey": "b1-2", "run": 2, "action": "skip"})
    assert validate_decision(request, {"itemKey": "b1-2", "action": "reject", "selected": ["a"]})["selected"] == []


def test_approval_endpoint_resumes_once_and_rejects_stale_decisions(research, monkeypatch):
    store = RunStore()
    monkeypatch.setattr(design_api, "runs", store)

    async def scenario():
        run = store.create("design")
        await execute_design_run(store, run, inputs())
        key = run.pending_approval["itemKey"]
        bad = design_api.ApprovalDecision(itemKey=key, action="approve", selected=["ghost:asset"])
        with pytest.raises(HTTPException) as err:
            await design_api.answer_approval(run.id, bad)
        assert err.value.status_code == 422
        stale = design_api.ApprovalDecision(itemKey="other-1", action="approve", selected=["blenderkit:walnut_cabinet"])
        with pytest.raises(HTTPException) as err:
            await design_api.answer_approval(run.id, stale)
        assert err.value.status_code == 409
        good = design_api.ApprovalDecision(itemKey=key, action="approve", selected=["blenderkit:walnut_cabinet"])
        await design_api.answer_approval(run.id, good)
        with pytest.raises(HTTPException) as err:  # double click
            await design_api.answer_approval(run.id, good)
        assert err.value.status_code == 409
        await run.task
        return run

    run = asyncio.run(scenario())
    assert research.downloads == ["blenderkit:walnut_cabinet"]
    assert run.events[-1]["type"] == "workflow.completed"


def test_cancel_while_waiting(research):
    store, run = start()
    cancel_waiting_run(store, run)
    assert run.events[-1]["type"] == "workflow.cancelled"
    assert events(run, "node.failed")[-1]["node"] == "human_approval"
    assert research.downloads == []


# --- downloads ---------------------------------------------------------------------------------

def test_download_failures_are_reported_per_asset(research):
    research.download_errors["blenderkit:walnut_cabinet"] = "No freely downloadable GLB/glTF file is available for this asset."
    store, run = start()
    resume(store, run, action="approve", selected=["polyhaven:oak_cabinet", "blenderkit:walnut_cabinet"])
    downloads = run.result["downloads"]
    assert [a["assetId"] for a in downloads["added"]] == ["polyhaven:oak_cabinet"]
    assert downloads["failed"][0]["assetId"] == "blenderkit:walnut_cabinet"
    assert "No freely downloadable" in run.result["message"]
    assert len(research.agent.requests) == len(ROUND_1)  # no new search after a download failure


def test_failed_download_of_every_asset_fails_the_item_without_a_new_search(research):
    research.download_errors["polyhaven:oak_cabinet"] = "broken"
    store, run = start()
    resume(store, run, action="approve", selected=["polyhaven:oak_cabinet"])
    assert run.result is None and run.events[-1]["type"] == "workflow.failed"  # direct search: the run fails
    assert "download failed: Oak Cabinet (broken)" in run.events[-1]["payload"]["error"]
    assert len(research.agent.requests) == len(ROUND_1)


def test_a_download_needs_a_matching_approval_or_the_automatic_mode(research):
    candidate = {"assetId": "polyhaven:oak_cabinet", "name": "Oak Cabinet"}
    item = {"key": "b-1", "index": 1, "run": 1, "candidates": [candidate], "selected": ["polyhaven:oak_cabinet"],
            "downloads": [{"assetId": "polyhaven:oak_cabinet", "name": "Oak Cabinet", "status": "queued"}]}
    guided = {"options": {}}
    assert not download_authorized(guided, {**item, "approved_by": None}, "polyhaven:oak_cabinet")
    assert not download_authorized(guided, {**item, "approved_by": "auto"}, "polyhaven:oak_cabinet")  # LLM cannot bypass
    assert not download_authorized(guided, {**item, "approved_by": "user", "approved_run": 2}, "polyhaven:oak_cabinet")
    assert not download_authorized(guided, {**item, "approved_by": "user", "approved_run": 1}, "grab3d:dark_cabinet")
    assert download_authorized(guided, {**item, "approved_by": "user", "approved_run": 1}, "polyhaven:oak_cabinet")
    assert download_authorized({"options": AUTOMATIC}, {**item, "approved_by": "auto"}, "polyhaven:oak_cabinet")

    update = asyncio.run(asset_download({"options": {}, "research_items": [{**item, "approved_by": "auto"}]}))
    entry = update["research_items"][0]["downloads"][0]
    assert entry["status"] == "failed" and "Not approved" in entry["error"]
    assert research.downloads == []


def test_duplicate_assets_are_not_downloaded_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "DATA_DIR", tmp_path)
    monkeypatch.setattr(inventory, "ASSETS_DIR", tmp_path / "inventory")
    monkeypatch.setattr(inventory, "INDEX_FILE", tmp_path / "inventory.json")
    downloads = []

    async def fake_download(asset_id):
        downloads.append(asset_id)
        return {"id": "x", "assetId": asset_id, "name": "Oak", "license": {"name": "CC0"}, "format": "glb",
                "modelPath": "m.glb", "modelUrl": "/api/inventory/x/files/m.glb", "downloadedAt": "2026-01-01",
                "dimensions": None, "unitScale": 1.0}
    monkeypatch.setattr(inventory, "download_asset", fake_download)

    first, created = asyncio.run(inventory.acquire_asset("polyhaven:oak"))
    again, created_again = asyncio.run(inventory.acquire_asset("polyhaven:oak"))
    assert created and not created_again and downloads == ["polyhaven:oak"]
    stored = inventory.load_index()["x"]
    assert stored["assetId"] == "polyhaven:oak" and stored["license"] == {"name": "CC0"}

    updated = asyncio.run(inventory.update_item("x", dimensions={"width": 1, "height": 2, "depth": 3}, unitScale=0.01))
    assert updated["unitScale"] == 0.01 and inventory.load_index()["x"]["dimensions"]["height"] == 2


def mock_client(handler):
    return public_http_client(transport=httpx.MockTransport(handler))


def test_invalid_download_responses_are_rejected(tmp_path, monkeypatch):
    async def scenario(handler, url="https://cdn.example.com/model.glb"):
        async with mock_client(handler) as client:
            return await inventory.download_file(client, url, tmp_path / "model.glb")

    with pytest.raises(ValueError, match="empty"):
        asyncio.run(scenario(lambda request: httpx.Response(200, content=b"")))
    monkeypatch.setattr(inventory, "MAX_FILE_BYTES", 10)
    with pytest.raises(ValueError, match="larger than"):
        asyncio.run(scenario(lambda request: httpx.Response(200, content=b"x" * 100)))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(scenario(lambda request: httpx.Response(404)))

    # a redirect to a private address is blocked before it is followed
    def redirect(request):
        if request.url.host == "cdn.example.com":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8000/secret"})
        return httpx.Response(200, content=b"glTF....")
    with pytest.raises(UnsafeURLError):
        asyncio.run(scenario(redirect))

    (tmp_path / "page.glb").write_text("<html>Login required</html>")
    with pytest.raises(ValueError, match="neither GLB nor glTF"):
        inventory.verify_model(tmp_path, "page.glb")


def test_preview_fetch_validates_url_and_content(monkeypatch):
    def handler(request):
        if request.url.path.endswith(".png"):
            return httpx.Response(200, content=b"\x89PNG data", headers={"content-type": "image/png"})
        return httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})
    monkeypatch.setattr(previews, "public_http_client",
                        lambda **kw: public_http_client(transport=httpx.MockTransport(handler), **kw))
    assert asyncio.run(previews.fetch_preview("https://cdn.example.com/a.png")).startswith("data:image/png;base64,")
    with pytest.raises(previews.PreviewError, match="not a supported image"):
        asyncio.run(previews.fetch_preview("https://cdn.example.com/page"))
    with pytest.raises(previews.PreviewError):
        asyncio.run(previews.fetch_preview("http://localhost/a.png"))


def test_unit_scale():
    assert unit_scale({"width": 0.8, "height": 1.8, "depth": 0.4}) == 1.0
    assert unit_scale({"width": 80, "height": 180, "depth": 40}) == 0.01
    assert unit_scale({"width": 800, "height": 1800, "depth": 400}) == 0.001


# --- lists: several items in parallel ----------------------------------------------------------

LIST = ["a grey sofa", "a floor lamp", "a wool rug"]


def list_research(fake_llm, monkeypatch, by_item):
    fake_llm(intent="SEARCH_FURNITURE", furniture_items=LIST)
    return install_research(monkeypatch, by_item=by_item)


def test_items_are_searched_one_by_one_and_shown_one_at_a_time(fake_llm, monkeypatch):
    research = list_research(fake_llm, monkeypatch, {
        "sofa": found("grey sofa", "polyhaven:grey_sofa", "polyhaven:green_sofa"),
        "lamp": found("floor lamp", "polyhaven:floor_lamp"),
        "rug": found("wool rug", "polyhaven:wool_rug")})
    store, run = start("A sofa, a lamp and a rug")
    assert run.pending_approval["total"] == 3
    seen = []
    while run.status == "waiting_for_approval":
        pending = run.pending_approval
        seen.append(pending["request"]["description"])
        resume(store, run, action="approve", selected=[pending["candidates"][0]["assetId"]])
    assert sorted(seen) == sorted(LIST)  # every item once, in whatever order its results arrived
    assert sorted(research.downloads) == ["polyhaven:floor_lamp", "polyhaven:grey_sofa", "polyhaven:wool_rug"]
    assert {i["status"] for i in run.result["downloads"]["items"]} == {"completed"}
    assert len(run.result["downloads"]["added"]) == 3
    # one MCP session per item search, never shared between parallel items
    assert len(FakeMCP.instances) == 3


def test_parallel_reject_retry_approve_and_download(fake_llm, monkeypatch):
    """A is rejected and searched again while B is approved and C is reviewed during B's download;
    A's new results come back to the queue later. No item is downloaded twice and the run ends only
    after every item is final and every download finished."""
    research = list_research(fake_llm, monkeypatch, {
        "sofa": [*found("grey sofa", "polyhaven:grey_sofa"), *found("corner sofa", "polyhaven:corner_sofa")],
        "lamp": found("floor lamp", "polyhaven:floor_lamp"),
        "rug": found("wool rug", "polyhaven:wool_rug")})
    hold = research.agent.hold
    hold["lamp"], hold["rug"] = asyncio.Event(), asyncio.Event()
    research.download_hold["polyhaven:floor_lamp"] = asyncio.Event()

    async def scenario():
        store = RunStore()
        run = store.create("design")
        await execute_design_run(store, run, inputs("A sofa, a lamp and a rug"))
        jobs = research_jobs.for_run(run.id)
        # 1. the sofa is ready first; lamp and rug are still being searched
        assert run.pending_approval["request"]["description"] == "a grey sofa"
        sofa_key = run.pending_approval["itemKey"]
        hold["sofa"] = asyncio.Event()  # the sofa's second run stays busy until released
        hold["lamp"].set()
        await resume_design_run(store, run, decision(run, action="reject", feedback="a corner sofa"))
        # 2. the sofa's second run is running, the lamp can be approved meanwhile
        assert run.pending_approval["request"]["description"] == "a floor lamp"
        assert jobs.state(f"search:{sofa_key}:2") == "running"
        hold["rug"].set()
        await resume_design_run(store, run, decision(run, action="approve", selected=["polyhaven:floor_lamp"]))
        # 3. the rug is reviewed while the lamp is still downloading
        assert run.pending_approval["request"]["description"] == "a wool rug"
        assert jobs.state("download:polyhaven:floor_lamp") == "running"
        assert research.downloads == ["polyhaven:floor_lamp"]
        hold["sofa"].set()
        await resume_design_run(store, run, decision(run, action="approve", selected=["polyhaven:wool_rug"]))
        # 4. the sofa's new suggestions are offered again (run 2), without the rejected one
        assert run.pending_approval["request"]["description"] == "a grey sofa" and run.pending_approval["run"] == 2
        assert [c["assetId"] for c in run.pending_approval["candidates"]] == ["polyhaven:corner_sofa"]
        task = asyncio.create_task(resume_design_run(store, run, decision(
            run, action="approve", selected=["polyhaven:corner_sofa"])))
        # 5. not finished while the lamp still downloads
        await until(lambda: "polyhaven:corner_sofa" in research.downloads)
        await asyncio.sleep(0.05)
        assert not run.finished
        research.download_hold["polyhaven:floor_lamp"].set()
        await task
        return run

    run = asyncio.run(scenario())
    assert run.events[-1]["type"] == "workflow.completed"
    assert sorted(research.downloads) == ["polyhaven:corner_sofa", "polyhaven:floor_lamp", "polyhaven:wool_rug"]
    assert len(research.downloads) == len(set(research.downloads))
    result = {i["description"]: i for i in run.result["downloads"]["items"]}
    assert result["a grey sofa"]["runs"] == 2 and result["a grey sofa"]["status"] == "completed"
    assert {i["status"] for i in result.values()} == {"completed"}
    assert research.agent.calls_by_item["sofa"] == 4  # two runs of two steps, nothing repeated


def test_two_items_choosing_the_same_asset_share_one_download(fake_llm, monkeypatch):
    fake_llm(intent="SEARCH_FURNITURE", furniture_items=["a side table", "a glass side table"])
    research = install_research(monkeypatch, by_item={
        "a side table": found("side table", "polyhaven:side_table"),
        "glass": found("side table", "polyhaven:side_table")})
    store, run = start("Two side tables", AUTOMATIC)
    assert research.downloads == ["polyhaven:side_table"]
    assert [i["status"] for i in run.result["downloads"]["items"]] == ["completed", "completed"]
    assert len(run.result["downloads"]["added"]) == 1


def test_list_is_cut_to_the_maximum_and_reported(fake_llm, monkeypatch):
    pieces = [f"piece number {n}" for n in range(1, 9)]
    fake_llm(intent="SEARCH_FURNITURE", furniture_items=pieces)
    install_research(monkeypatch, by_item={"piece": []})
    _, run = start("Eight pieces", LOCAL_ONLY)
    assert len(run.result["downloads"]["items"]) == agent.MAX_ITEMS_PER_REQUEST == 6
    log = [line for e in events(run, "node.completed") for line in e["payload"]["output"].get("workflow_log", [])]
    assert any("list cut to 6 items" in line and "piece number 7" in line for line in log)


def test_every_piece_of_a_direct_search_is_searched_online(fake_llm, monkeypatch):
    research = list_research(fake_llm, monkeypatch, {
        "sofa": found("grey sofa", "polyhaven:grey_sofa"), "lamp": found("floor lamp", "polyhaven:floor_lamp"),
        "rug": found("wool rug", "polyhaven:wool_rug")})
    store, run = start("A sofa, a lamp and a rug", AUTOMATIC)
    assert [i["status"] for i in run.result["downloads"]["items"]] == ["completed"] * 3
    assert len(research.downloads) == 3


# --- settings: local only, guided, automatic ----------------------------------------------------

def test_local_only_searches_nothing_online(research):
    _, run = start(options=LOCAL_ONLY)
    assert FakeMCP.instances == [] and research.agent.requests == [] and research.downloads == []
    assert "human_approval" not in started(run)
    item = items(run)[REQUEST]
    assert item["status"] == "not_found" and "online search is disabled" in item["note"]


def test_automatic_mode_downloads_the_best_suitable_match_without_approval(fake_llm, monkeypatch):
    fake_llm(intent="SEARCH_FURNITURE")
    research = install_research(monkeypatch, *found("cabinet", "polyhaven:oak_cabinet", "grab3d:black_cabinet"))
    _, run = start(options=AUTOMATIC)
    assert "human_approval" not in started(run) and not events(run, "workflow.waiting")
    assert research.downloads == ["polyhaven:oak_cabinet"]  # only the best one
    item = items(run)[REQUEST]
    assert item["status"] == "completed" and item["auto"] and item["runs"] == 1
    assert "chosen automatically" in run.result["message"]
    assert agent.AUTOMATIC_RULE in research.agent.requests[0][1]["content"]


def test_automatic_mode_never_downloads_an_insufficient_match(fake_llm, monkeypatch):
    # Both candidates are called suitable, but the checks do not allow an unattended download.
    fake_llm(intent="SEARCH_FURNITURE")
    research = install_research(
        monkeypatch,
        *found("cabinet", "polyhaven:oak_cabinet", checks={"category": "unclear", "size": "ok"}),
        *found("dark cabinet", "grab3d:dark_cabinet", checks={"category": "ok", "size": "fails"}))
    _, run = start(options=AUTOMATIC)
    assert research.downloads == []
    item = items(run)[REQUEST]
    assert item["status"] == "not_found" and item["runs"] == 2  # second run allowed, then not found
    assert "suitable enough" in research.agent.requests[2][1]["content"]


def test_automatic_mode_needs_a_known_license():
    candidate = {"suitable": True, "checks": {"category": "ok"}, "license": None}
    assert not agent.auto_acceptable(candidate)
    assert agent.auto_acceptable({**candidate, "license": {"name": "CC0"}})
    assert not agent.auto_acceptable({**candidate, "license": {"name": "CC0"}, "suitable": None})


# --- MCP errors ----------------------------------------------------------------------------------

def test_mcp_unavailable_is_reported_and_research_ends_without_results(research):
    FakeMCP.fail_on_connect = True
    _, run = start()
    failed = [e["payload"] for e in events(run, "node.tool") if e["payload"]["status"] == "failed"]
    assert failed and "not reachable" in failed[0]["error"]
    assert run.events[-1]["type"] == "workflow.completed"
    assert run.result["downloads"]["status"] == "no_results"
    assert research.downloads == []


def test_unreachable_server_is_not_reported_as_nothing_found(research):
    # Every tool call used to open a new session (more requests to an overloaded server), and the item
    # ended as "not_found" after two instant runs although nothing was searched.
    FakeMCP.fail_on_connect = True
    research.agent.turns = [[call("search_assets", query="cabinet")], [call("search_assets", query="wooden cabinet")]]
    _, run = start()
    item = items(run)[REQUEST]
    assert item["status"] == "failed" and "asset server unavailable" in item["note"] and item["runs"] == 1
    assert len(FakeMCP.instances) == 1  # one connection attempt, then the run stops
    assert len(research.agent.requests) == 1  # one online step, no further LLM steps (no local match: no LLM)


def test_rate_limited_tool_call_stops_the_run(research, monkeypatch):
    async def limited(self, query, limit=12):
        raise AssetMCPError("Rate limit exceeded: 120 requests per 60s. Retry after 19s.")
    monkeypatch.setattr(FakeMCP, "search", limited)
    _, run = start()
    item = items(run)[REQUEST]
    assert item["status"] == "failed" and "Rate limit exceeded" in item["note"] and item["runs"] == 1


def test_designer_sees_why_a_search_failed():
    from design.interior_designer import _research_item_text
    entry = {"description": "bedside lamp", "status": "failed", "added": [], "existing": [], "failed": [],
             "note": "online search failed (AssetMCPError: asset server unavailable: rate limit)"}
    assert "asset server unavailable" in _research_item_text(entry)


@pytest.mark.parametrize("detail, wait", [("Asset server rate limit reached. Rate limit exceeded: 120 requests per 60s. "
                                           "Retry after 19s.", 20.0),
                                          ("Asset server rate limit reached.", 20.0),
                                          ("Asset server rate limit reached. Retry after 300s.", None)])
def test_rate_limit_wait_follows_the_server(detail, wait):
    from research.asset_download import rate_limit_wait
    assert rate_limit_wait(HTTPException(429, detail)) == wait
    assert rate_limit_wait(HTTPException(422, detail)) is None


def test_rate_limited_download_is_tried_once_more(monkeypatch):
    import research.asset_download as download_module
    calls, sleeps = [], []

    async def process(asset_id, metadata):
        calls.append(asset_id)
        if len(calls) == 1:
            raise HTTPException(429, "Asset server rate limit reached. Retry after 2s.")
        raise HTTPException(422, "No freely downloadable GLB/glTF file is available for this asset.")

    async def no_sleep(seconds):
        sleeps.append(seconds)
    monkeypatch.setattr(download_module, "process_asset", process)
    monkeypatch.setattr(download_module.asyncio, "sleep", no_sleep)
    result = asyncio.run(download_module.download("a:1", "Lamp", {}))
    assert len(calls) == 2 and sleeps == [3.0]
    assert result["status"] == "failed" and "GLB/glTF" in result["error"]  # the second error is reported


def test_mcp_tool_errors_go_back_to_the_agent(research):
    FakeMCP.fail_search = True
    _, run = start()
    tool_events = events(run, "node.tool")
    assert tool_events[0]["payload"]["tool"] == "search_local_furniture"  # direct search: local first
    assert tool_events[1]["payload"]["status"] == "failed" and "timeout" in tool_events[1]["payload"]["error"]
    tool_text = next(m for m in research.agent.requests[1] if m["role"] == "tool")["content"]
    assert tool_text.startswith("Tool error")


def test_research_for_the_designer_skips_the_local_search(research):
    # The designer already saw the inventory, so the research goes straight to the external search.
    state = {"request": REQUEST, "model": "fake", "room": ROOM, "furniture": FURNITURE,
             "furniture_request": {"items": [{"description": "modern wooden cabinet"}]},
             "return_to": {"furniture_research": "interior_designer"}}
    update = asyncio.run(agent.furniture_research(state))
    assert research.agent.requests[0][0]["content"] == agent.EXTERNAL_ONLY_PROMPT
    assert "furniture database" not in research.agent.requests[0][1]["content"]
    item = update["research_items"][0]
    assert update["research_status"] == "awaiting_approval" and update["approval_item"] == item["key"]
    assert [c["assetId"] for c in item["candidates"]] == ["polyhaven:oak_cabinet", "blenderkit:walnut_cabinet"]
    assert update["counters"] == {"research_requests": 1}


def test_tools_for_the_designer_have_no_local_search():
    names = [t["function"]["name"] for t in agent.EXTERNAL_TOOLS]
    assert "search_local_furniture" not in names and "search_assets" in names
    assert "search_local_furniture" not in agent.EXTERNAL_ONLY_PROMPT


def test_parse_tool_result():
    text = type("C", (), {"text": '{"results": []}'})()
    ok = type("R", (), {"content": [text], "is_error": False, "structured_content": None})()
    assert parse_tool_result(ok) == {"results": []}
    error = type("R", (), {"content": [type("C", (), {"text": "Unknown provider"})()], "is_error": True})()
    with pytest.raises(AssetMCPError, match="Unknown provider"):
        parse_tool_result(error)


# --- observability and routing -------------------------------------------------------------------

def test_observability_events(research):
    store, run = start()
    tools = [(e["payload"]["tool"], e["payload"]["input"]) for e in events(run, "node.tool")]
    assert tools[0] == ("search_local_furniture", {"query": REQUEST, "limit": agent.LOCAL_HITS})  # local first
    assert tools[1] == ("search_assets", {"query": "wooden cabinet"})
    assert ("view_preview", {"asset_id": "polyhaven:oak_cabinet"}) in tools
    assert all(e["node"] == "furniture_research" for e in events(run, "node.tool"))  # job events replayed
    assert len(events(run, "node.llm")) == 1 + 3  # orchestrator + three research steps
    waiting = events(run, "node.waiting")[0]
    assert waiting["node"] == "human_approval" and waiting["status"] == "waiting"
    assert run.events[-1]["payload"]["approval"]["kind"] == "asset_selection"
    research_done = next(e for e in events(run, "node.completed") if e["node"] == "furniture_research")
    assert research_done["payload"]["summary"] == "item 1/1 · run 1 · 2 suggestion(s)"
    assert research_done["payload"]["next"] == "human_approval"
    labels = [e["payload"]["label"] for e in events(run, "node.progress")]
    assert any("local match" in label for label in labels) and any("waiting for approval" in label for label in labels)

    resume(store, run, action="approve", selected=["polyhaven:oak_cabinet"])
    resumed = events(run, "workflow.resumed")[0]["payload"]
    assert resumed["decision"]["selected"] == ["polyhaven:oak_cabinet"] and resumed["waitMs"] >= 0
    approval_done = next(e for e in events(run, "node.completed") if e["node"] == "human_approval")
    assert approval_done["payload"]["summary"] == "approved 1 asset(s)"
    assert "waitMs" in approval_done["payload"]
    final = run.events[-1]["payload"]
    assert final["durationMs"] >= final["activeMs"] and final["waitMs"] >= 0


def test_research_returns_to_its_caller():
    done = {"research_status": "done"}
    assert next_node("furniture_research", done) == "__end__"  # standalone search
    assert next_node("furniture_research", {**done, "return_to": {"furniture_research": "interior_designer"}}) == "interior_designer"
    assert next_node("furniture_research", {"research_status": "awaiting_approval", "approval_item": "x"}) == "human_approval"
    assert next_node("furniture_research", {"research_status": "download"}) == "asset_download"
    assert next_node("asset_download", {}) == "furniture_research"  # downloads never end the run
    assert next_node("human_approval", {"approval": {"action": "reject"}}) == "furniture_research"
    assert next_node("human_approval", {"approval": {"action": "skip"}}) == "furniture_research"
    assert next_node("human_approval", {"approval": {"action": "approve"}}) == "asset_download"
