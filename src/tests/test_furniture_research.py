"""Furniture research, human approval and asset download, with fake MCP, LLM, previews and downloads."""
import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException

import design_api
import inventory
import research.agent as agent
import research.previews as previews
from furniture.pipeline import unit_scale
from design.human_approval import ApprovalError, validate_decision
from design.routing import next_node
from design.runner import cancel_waiting_run, execute_design_run, resume_design_run
from design.state import Furniture, Room
from observability.store import RunStore
from tests.research_fakes import ROUND_1, ROUND_2, FakeMCP, call, install_research
from research.mcp_client import AssetMCPError, parse_tool_result
from url_safety import UnsafeURLError, public_http_client

ROOM = Room(width=8, depth=6, height=2.8)
FURNITURE = [Furniture(id="sofa", name="Sofa", width=2, depth=0.9, x=0, z=-2.5)]
REQUEST = "Find a modern wooden cabinet that fits my living room."


@pytest.fixture
def research(fake_llm, monkeypatch):
    """Fakes for MCP, LLM tool calls, previews and downloads (see tests/research_fakes.py)."""
    fake_llm(intent="SEARCH_FURNITURE")
    return install_research(monkeypatch)


def start(request=REQUEST):
    store = RunStore()
    run = store.create("design")
    asyncio.run(execute_design_run(store, run, {"request": request, "model": "fake", "room": ROOM, "furniture": FURNITURE}))
    return store, run


def resume(store, run, **decision):
    candidates = run.pending_approval["candidates"]
    asyncio.run(resume_design_run(store, run, validate_decision(candidates, decision)))


def started(run):
    return [e["node"] for e in run.events if e["type"] == "node.started"]


def events(run, type_):
    return [e for e in run.events if e["type"] == type_]


# --- research and pause ----------------------------------------------------------------------

def test_search_request_goes_to_research_and_pauses_for_approval(research):
    store, run = start()
    assert started(run) == ["orchestrator", "furniture_research", "human_approval"]
    assert run.events[-1]["type"] == "workflow.waiting"
    assert run.status == "waiting_for_approval"
    assert research.downloads == []  # nothing downloaded before approval


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
    # same run and execution resumed, not a new research
    # research ran, then checked the download result after asset_download (same run, no new research)
    assert started(run) == ["orchestrator", "furniture_research", "human_approval", "asset_download", "furniture_research"]
    assert run.result["downloads"]["status"] == "completed"
    assert [e["node"] for e in events(run, "node.resumed")] == ["human_approval"]


def test_no_results_when_everything_suitable_was_rejected(research):
    research.agent.turns = [*ROUND_1, *ROUND_2[:1],
                            [call("submit_candidates", summary="Nothing new found.", candidates=[
                                {"asset_id": "polyhaven:oak_cabinet", "reason": "already rejected"}])]]
    store, run = start()
    resume(store, run, action="retry", feedback="something else")
    assert run.events[-1]["type"] == "workflow.completed"
    assert run.result["message"] == "Nothing new found."
    assert research.downloads == []


def test_reject_downloads_nothing(research):
    store, run = start()
    resume(store, run, action="reject", feedback="Not my style")
    assert research.downloads == []
    assert run.events[-1]["type"] == "workflow.completed"
    assert "rejected" in run.result["message"]


def test_retry_with_feedback_searches_again_without_the_rejected_assets(research):
    store, run = start()
    resume(store, run, action="retry", feedback="Show me darker wooden furniture.")
    assert research.downloads == []
    assert run.status == "waiting_for_approval"
    prompt = research.agent.requests[3][1]["content"]
    assert "Show me darker wooden furniture." in prompt and "round 2" in prompt
    candidates = [c["assetId"] for c in run.pending_approval["candidates"]]
    assert candidates == ["grab3d:dark_cabinet", "grab3d:black_cabinet"]  # oak was excluded
    assert run.pending_approval["round"] == 2

    resume(store, run, action="approve", selected=["grab3d:dark_cabinet", "grab3d:black_cabinet"])
    assert research.downloads == ["grab3d:dark_cabinet", "grab3d:black_cabinet"]
    nodes = started(run)
    assert nodes.count("furniture_research") == 3 and nodes.count("human_approval") == 2  # 2 rounds + result check
    ids = [e["executionId"] for e in events(run, "node.started")]
    assert len(ids) == len(set(ids))


def test_research_rounds_are_limited(research):
    store, run = start()
    for _ in range(agent.MAX_RESEARCH_ROUNDS):
        assert run.status == "waiting_for_approval"
        resume(store, run, action="retry", feedback="other ones")
    assert run.events[-1]["type"] == "workflow.completed"
    assert "limit" in run.result["message"]
    assert research.downloads == []
    assert started(run).count("furniture_research") == agent.MAX_RESEARCH_ROUNDS + 1  # last one only reports the limit


def test_selection_must_come_from_the_candidates():
    candidates = [{"assetId": "a"}, {"assetId": "b"}]
    with pytest.raises(ApprovalError):
        validate_decision(candidates, {"action": "approve", "selected": ["a", "zzz"]})
    with pytest.raises(ApprovalError):
        validate_decision(candidates, {"action": "approve", "selected": []})
    with pytest.raises(ApprovalError):
        validate_decision(candidates, {"action": "delete"})
    assert validate_decision(candidates, {"action": "reject", "selected": ["a"]})["selected"] == []


def test_approval_endpoint_resumes_once_and_validates(research, monkeypatch):
    store = RunStore()
    monkeypatch.setattr(design_api, "runs", store)

    async def scenario():
        run = store.create("design")
        await execute_design_run(store, run, {"request": REQUEST, "model": "fake", "room": ROOM, "furniture": FURNITURE})
        bad = design_api.ApprovalDecision(action="approve", selected=["ghost:asset"])
        with pytest.raises(HTTPException) as err:
            await design_api.answer_approval(run.id, bad)
        assert err.value.status_code == 422
        good = design_api.ApprovalDecision(action="approve", selected=["blenderkit:walnut_cabinet"])
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


# --- MCP errors ----------------------------------------------------------------------------------

def test_mcp_unavailable_is_reported_and_research_ends_without_results(research):
    # The external search is optional (local first), so an unreachable MCP server does not crash the run.
    FakeMCP.fail_on_connect = True
    _, run = start()
    failed = [e["payload"] for e in events(run, "node.tool") if e["payload"]["status"] == "failed"]
    assert failed and "not reachable" in failed[0]["error"]
    assert run.events[-1]["type"] == "workflow.completed"
    assert run.result["downloads"]["status"] == "no_results"
    assert research.downloads == []


def test_mcp_tool_errors_go_back_to_the_agent(research):
    FakeMCP.fail_search = True
    _, run = start()
    tool_events = events(run, "node.tool")
    assert tool_events[0]["payload"]["tool"] == "search_local_furniture"  # always searched first
    assert tool_events[1]["payload"]["status"] == "failed" and "timeout" in tool_events[1]["payload"]["error"]
    tool_text = next(m for m in research.agent.requests[1] if m["role"] == "tool")["content"]
    assert tool_text.startswith("Tool error")


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
    assert tools[0] == ("search_local_furniture", {"query": REQUEST, "limit": 8})  # local first, automatically
    assert tools[1] == ("search_assets", {"query": "wooden cabinet"})
    assert ("view_preview", {"asset_id": "polyhaven:oak_cabinet"}) in tools
    assert all(e["node"] == "furniture_research" for e in events(run, "node.tool"))
    assert len(events(run, "node.llm")) == 1 + 3  # orchestrator + three research turns
    waiting = events(run, "node.waiting")[0]
    assert waiting["node"] == "human_approval" and waiting["status"] == "waiting"
    assert run.events[-1]["payload"]["approval"]["kind"] == "asset_selection"
    research_done = next(e for e in events(run, "node.completed") if e["node"] == "furniture_research")
    assert research_done["payload"]["summary"] == "round 1 · 2 candidate(s)"
    assert research_done["payload"]["next"] == "human_approval"

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
    assert next_node("asset_download", {}) == "furniture_research"  # downloads never end the run
    assert next_node("human_approval", {"approval": {"action": "reject"}}) == "furniture_research"
