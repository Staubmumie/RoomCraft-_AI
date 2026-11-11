"""When the interior designer delegates missing furniture to furniture research, and when it must not.
The designer's choices are scripted; these tests check what the workflow makes of them (context,
guards, routing, warnings), not how well a real LLM judges what is missing."""
import asyncio

from design.interior_designer import SYSTEM_PROMPT, DesignerDecision
from design.runner import resume_design_run
from tests.conftest import decide
from tests.research_fakes import ROUND_1, install_research
from tests.test_workflows import CLEAN, completed, final, run, started


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
    assert "kitchen base cabinets" in first["furniture_request"]["description"]
    assert "essential furniture missing" in first["workflow_log"][0]
    assert "kitchen base cabinets" in research.agent.requests[0][1]["content"]

    # The user rejects every suggestion: the same gap is not searched again (budget is left), the designer
    # finishes, and the gap is reported, not dropped.
    asyncio.run(resume_design_run(store, result, {"action": "reject", "selected": [], "feedback": ""}))
    assert started(result).count("furniture_research") == 2  # first round + the round handling the rejection
    assert result.events[-1]["type"] == "workflow.completed"
    assert any("Not fulfilled: kitchen base cabinets" in w for w in final(result)["warnings"])
    assert [g["item"] for g in final(result)["missing"]] == ["kitchen base cabinets"]
    assert research.downloads == []


def test_each_essential_gap_gets_its_own_request(fake_llm, monkeypatch):
    gaps = [("kitchen base cabinets", True), ("dining table", True)]
    fake_llm(decide(*CLEAN, gaps=gaps), decide(gaps=gaps), decide(gaps=gaps))
    install_research(monkeypatch, *ROUND_1, *ROUND_1)
    store, result = run("Living room with a kitchen and a dining table.", add=True)
    # Rejecting the first suggestions; the second search then finds nothing new (rejected assets are excluded).
    asyncio.run(resume_design_run(store, result, {"action": "reject", "selected": [], "feedback": ""}))
    requests = [o["furniture_request"]["description"] for o in designer_outputs(result) if "furniture_request" in o]
    assert [r.split(" (")[0] for r in requests] == ["kitchen base cabinets", "dining table"]
    assert result.events[-1]["type"] == "workflow.completed"  # budget used up: finish with both gaps reported
    assert len(final(result)["missing"]) == 2


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


def test_research_needs_a_designer_step_left_to_place_the_result(fake_llm):
    fake = fake_llm(decide(*CLEAN, action="request_furniture", finish=False, request="kitchen cabinets"))
    _, result = run(add=True, max_steps=1)
    assert "NOT available (no designer step left" in designer_text(fake, 0)
    assert "furniture_research" not in started(result)
    assert designer_outputs(result)[0]["designer_action"] == "validate"  # safe fallback, layout still validated
    assert result.events[-1]["type"] == "workflow.completed"
