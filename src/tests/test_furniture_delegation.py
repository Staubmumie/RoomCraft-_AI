"""When the interior designer delegates missing furniture to furniture research, and when it must not.
The designer's choices are scripted; these tests check what the workflow makes of them (context,
guards, routing, warnings), not how well a real LLM judges what is missing."""
import asyncio

import research.agent as agent
from design.interior_designer import SYSTEM_PROMPT, DesignerDecision
from design.runner import execute_design_run, resume_design_run
from observability.store import RunStore
from tests.conftest import decide
from tests.research_fakes import ROUND_1, found, install_research
from tests.test_workflows import CLEAN, FURNITURE, REQUEST, ROOM, answer, completed, final, run, started


def designer_text(fake, call):
    content = fake.designer_calls[call][1]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def designer_outputs(result):
    return [e["payload"]["output"] for e in completed(result, "interior_designer")]


def test_completeness_check_comes_before_the_action_and_the_explanation_last():
    fields = list(DesignerDecision.model_fields)
    assert fields.index("furniture_gaps") < fields.index("next_action")
    assert fields[-1] == "explanation"  # written after the decision, so it describes what actually happens
    assert "Never finish with an offer" in SYSTEM_PROMPT


# --- 1. kitchen requested, no kitchen furniture available ------------------------------------------

def test_essential_gap_starts_research_instead_of_finishing(fake_llm, monkeypatch):
    # The designer arranges the room and wants to finish although the kitchen is missing.
    fake = fake_llm(decide(*CLEAN, gaps=[("kitchen base cabinets", True)]),
                    decide(gaps=[("kitchen base cabinets", True)]))
    research = install_research(monkeypatch, *ROUND_1)
    store, result = run("Create a modern living room with a kitchen.", add=True)

    assert "Furniture research: AVAILABLE" in designer_text(fake, 0)
    assert started(result)[-2:] == ["furniture_research", "human_approval"]  # approval before any download
    first = designer_outputs(result)[0]
    assert first["designer_action"] == "request_furniture"
    assert "kitchen base cabinets" in first["furniture_request"]["items"][0]["description"]
    assert first["furniture_request"]["items"][0]["essential"] is True
    assert "essential furniture missing" in first["workflow_log"][0]
    assert "kitchen base cabinets" in research.agent.requests[0][1]["content"]

    # The user rejects every suggestion: the second run finds nothing new (rejected assets are excluded),
    # the same gap is not searched again (budget is left), the designer finishes and reports the gap.
    asyncio.run(resume_design_run(store, result, answer(result, action="reject")))
    assert started(result).count("furniture_research") == 2  # first run + the coordinator handling the rejection
    assert result.events[-1]["type"] == "workflow.completed"
    assert any("Not fulfilled: kitchen base cabinets" in w for w in final(result)["warnings"])
    assert [g["item"] for g in final(result)["missing"]] == ["kitchen base cabinets"]
    assert research.downloads == []


def test_all_essential_gaps_go_into_one_list(fake_llm, monkeypatch):
    gaps = [("kitchen base cabinets", True), ("dining table", True)]
    fake = fake_llm(decide(*CLEAN, gaps=gaps), decide(gaps=gaps), decide(gaps=gaps))
    install_research(monkeypatch, by_item={"kitchen": [], "dining": []})  # nothing found online
    _, result = run("Living room with a kitchen and a dining table.", add=True)
    requests = [o["furniture_request"] for o in designer_outputs(result) if "furniture_request" in o]
    assert len(requests) == 1  # one list, not one request per piece
    assert [i["description"].split(" (")[0] for i in requests[0]["items"]] == ["kitchen base cabinets", "dining table"]
    assert started(result).count("furniture_research") == 1
    research_out = completed(result, "furniture_research")[-1]["payload"]["output"]
    assert research_out["counters"] == {"research_requests": 1}
    assert {i["status"] for i in research_out["research_result"]["items"]} == {"not_found"}
    second = designer_text(fake, 1)
    assert "- kitchen base cabinets (requested) [essential]: not found" in second
    assert result.events[-1]["type"] == "workflow.completed"  # researched once: finish with both gaps reported
    assert len(final(result)["missing"]) == 2


def test_designer_list_is_merged_with_unresearched_essential_gaps(fake_llm, monkeypatch):
    fake_llm(decide(*CLEAN, action="request_furniture", finish=False, request=["low oak coffee table"],
                    gaps=[("coffee table", False), ("kitchen base cabinets", True)]), decide(*CLEAN))
    install_research(monkeypatch, by_item={"coffee": [], "kitchen": []})
    _, result = run(add=True)
    request = designer_outputs(result)[0]["furniture_request"]
    descriptions = [i["description"] for i in request["items"]]
    assert descriptions[0].startswith("kitchen base cabinets") and descriptions[1] == "low oak coffee table"  # essential first
    assert "added essential gaps to the list" in designer_outputs(result)[0]["workflow_log"][0]


def test_designer_list_skips_the_local_search_and_reuses_inventory_hits(fake_llm, monkeypatch):
    """Designer list with three pieces: no local search at all; an online result that is already in
    the inventory is taken over without download and approval; the rest is chosen automatically."""
    fake = fake_llm(decide(*CLEAN, action="request_furniture", finish=False,
                           request=["oak cabinet", "floor lamp", "wool rug"]), decide(*CLEAN))
    research = install_research(monkeypatch, by_item={
        "cabinet": found("oak cabinet", "polyhaven:oak_cabinet"), "lamp": found("floor lamp", "polyhaven:floor_lamp"),
        "rug": found("wool rug", "polyhaven:wool_rug")})
    monkeypatch.setattr(agent, "inventory_asset_ids", lambda: {"polyhaven:oak_cabinet": "polyhaven-oak_cabinet"})
    store = RunStore()
    result = store.create("design")
    asyncio.run(execute_design_run(store, result, {
        "request": REQUEST, "model": "fake", "room": ROOM, "furniture": FURNITURE,
        "options": {"allow_new_furniture": True, "review_suggestions": False}}))
    assert result.events[-1]["type"] == "workflow.completed"
    tools = [e["payload"]["tool"] for e in result.events if e["type"] == "node.tool"]
    assert "search_local_furniture" not in tools
    assert all(r[0]["content"] == agent.EXTERNAL_ONLY_PROMPT for r in research.agent.requests)
    assert "human_approval" not in started(result)
    assert sorted(research.downloads) == ["polyhaven:floor_lamp", "polyhaven:wool_rug"]  # not the cabinet
    second = designer_text(fake, 1)
    assert "- oak cabinet: already in the inventory: Oak Cabinet (item_id polyhaven-oak_cabinet)" in second
    assert "- floor lamp: chosen automatically and downloaded: Floor Lamp (item_id polyhaven-floor_lamp)" in second
    assert [a["assetId"] for a in result.result["downloads"]["added"]] != []  # the UI reloads its inventory


def test_list_limit_bounds_the_research_and_every_piece_of_a_list_is_searched(fake_llm, monkeypatch):
    fake = fake_llm(decide(*CLEAN, action="request_furniture", finish=False, request=["floor lamp", "wool rug"]),
                    decide(*CLEAN))
    install_research(monkeypatch, by_item={"lamp": [], "rug": []})
    store = RunStore()
    result = store.create("design")
    asyncio.run(execute_design_run(store, result, {
        "request": REQUEST, "model": "fake", "room": ROOM, "furniture": FURNITURE,
        "options": {"allow_new_furniture": True, "limits": {"research_requests": 1}}}))
    assert "AVAILABLE for 1 more list(s) of up to 6 pieces each" in designer_text(fake, 0)
    second = designer_text(fake, 1)
    assert "- floor lamp: not found" in second and "- wool rug: not found" in second  # no per-piece budget
    assert "NOT available (furniture research limit reached)" in second  # the one list is used


def test_skipped_piece_is_not_researched_again_automatically(fake_llm, monkeypatch):
    gaps = [("kitchen base cabinets", True)]
    fake = fake_llm(decide(*CLEAN, gaps=gaps), decide(gaps=gaps))
    research = install_research(monkeypatch, *ROUND_1)
    store, result = run("Create a modern living room with a kitchen.", add=True)
    asyncio.run(resume_design_run(store, result, answer(result, action="skip")))
    assert started(result).count("furniture_research") == 2 and started(result).count("human_approval") == 1
    assert result.events[-1]["type"] == "workflow.completed"
    assert len(research.agent.requests) == len(ROUND_1)
    assert "skipped by the user (do not request it again)" in designer_text(fake, 1)
    assert any("Not fulfilled: kitchen base cabinets" in w for w in final(result)["warnings"])


# --- 2. functional living room that could use a coffee table -----------------------------------------

def test_optional_gap_does_not_force_research(fake_llm):
    fake_llm(decide(*CLEAN, gaps=[("coffee table", False)]))
    _, result = run(add=True)
    assert "furniture_research" not in started(result)
    assert final(result)["warnings"] == []


def test_designer_may_still_request_an_optional_piece(fake_llm, monkeypatch):
    fake_llm(decide(*CLEAN, action="request_furniture", finish=False, request="low oak coffee table",
                    gaps=[("coffee table", False)]))
    install_research(monkeypatch, *ROUND_1)
    _, result = run(add=True)
    assert designer_outputs(result)[0]["designer_action"] == "request_furniture"
    assert result.status == "waiting_for_approval"


# --- 3. suitable furniture already in the inventory --------------------------------------------------

def test_inventory_item_is_added_directly_without_research(fake_llm):
    fake = fake_llm(decide(*CLEAN, add=[("chair-item", 0, 2)]))
    _, result = run("Add a reading chair.", add=True)
    assert "Armchair" in designer_text(fake, 0)  # the inventory is part of the context
    assert "furniture_research" not in started(result)
    assert [p["name"] for p in final(result)["placements"] if p["id"].startswith("new-")] == ["Armchair"]


# --- 4. adding / research not permitted, or no budget left ---------------------------------------------

def test_without_permission_the_essential_gap_becomes_a_warning(fake_llm):
    fake = fake_llm(decide(*CLEAN, gaps=[("kitchen base cabinets", True)]))
    _, result = run("Create a modern living room with a kitchen.", add=False)
    assert "Furniture research: NOT available (adding furniture is not allowed)" in designer_text(fake, 0)
    assert "furniture_research" not in started(result)
    assert result.events[-1]["type"] == "workflow.completed"
    assert any("Not fulfilled: kitchen base cabinets" in w for w in final(result)["warnings"])


def test_without_online_search_the_designer_never_researches(fake_llm):
    fake = fake_llm(decide(*CLEAN, action="request_furniture", finish=False, request="kitchen cabinets",
                           gaps=[("kitchen base cabinets", True)]),
                    decide(*CLEAN, gaps=[("kitchen base cabinets", True)]))
    store = RunStore()
    result = store.create("design")
    asyncio.run(execute_design_run(store, result, {
        "request": "Create a modern living room with a kitchen.", "model": "fake", "room": ROOM,
        "furniture": FURNITURE, "options": {"allow_new_furniture": True, "search_online": False}}))
    assert "NOT available (online furniture search is switched off" in designer_text(fake, 0)
    assert "Available inventory" in designer_text(fake, 0)  # the inventory is still offered directly
    assert "furniture_research" not in started(result)
    assert designer_outputs(result)[0]["designer_action"] == "validate"  # safe fallback
    assert any("Not fulfilled: kitchen base cabinets" in w for w in final(result)["warnings"])


def test_research_needs_a_designer_step_left_to_place_the_result(fake_llm):
    fake = fake_llm(decide(*CLEAN, action="request_furniture", finish=False, request="kitchen cabinets"))
    _, result = run(add=True, max_steps=1)
    assert "NOT available (no designer step left" in designer_text(fake, 0)
    assert "furniture_research" not in started(result)
    assert designer_outputs(result)[0]["designer_action"] == "validate"  # safe fallback, layout still validated
    assert result.events[-1]["type"] == "workflow.completed"
