"""Functional planning of the interior designer: the plan in the structured output, how it is kept
in the state, shown to the designer and the critic, and the deterministic plan-vs-layout hints.
These tests check the mechanism, not the design quality of real LLM layouts."""
from openai.lib._pydantic import to_strict_json_schema

from design.interior_designer import SYSTEM_PROMPT, DesignerDecision, plan_notes
from design.state import Furniture
from tests.conftest import INVENTORY, P, decide
from tests.test_workflows import completed, run

DINING_ROOM = [
    Furniture(id="table", name="Dining Table", width=1.6, depth=0.9, x=0, z=0),
    *(Furniture(id=f"chair{i}", name="Dining Chair", width=0.45, depth=0.5, x=-3 + i, z=2.5) for i in range(1, 5)),
]
DINING_PLAN = {
    "requirements": ["dining for four"],
    "zones": [{"name": "dining", "purpose": "meals for four", "area": "room centre", "anchor": "table",
               "pieces": ["table", "chair1", "chair2", "chair3", "chair4"],
               "arrangement": "two chairs per long side, facing the table", "missing": []}],
    "composition": "table centred, 0.9 m free around it",
}
AROUND_TABLE = [P("chair1", -0.4, -0.75, 0), P("chair2", 0.4, -0.75, 0), P("chair3", -0.4, 0.75, 180),
                P("chair4", 0.4, 0.75, 180)]


def designer_text(fake, call):
    content = fake.designer_calls[call][1]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def zone(name, anchor=None, pieces=(), missing=()):
    return {"name": name, "purpose": "", "area": "", "anchor": anchor, "pieces": list(pieces),
            "arrangement": "", "missing": list(missing)}


# --- schema and prompt ------------------------------------------------------------------------------

def test_plan_is_generated_before_the_placements():
    # Structured output is generated in schema order: the plan must come before any concrete change.
    fields = list(DesignerDecision.model_fields)
    assert fields.index("plan") < fields.index("placements") < fields.index("add")


def test_decision_schema_stays_compatible_with_strict_structured_output():
    schema = to_strict_json_schema(DesignerDecision)
    assert "plan" in schema["required"]


def test_prompt_keeps_the_technical_rules_and_adds_design_principles():
    for kept in ("Coordinate system", "Hard constraints", "Never invent furniture", "request_furniture", "finish_when_valid"):
        assert kept in SYSTEM_PROMPT
    for added in ("Function before style", "anchor", "never silently drop a wish", "do not replan the room"):
        assert added in SYSTEM_PROMPT


# --- plan vs. layout hints ----------------------------------------------------------------------------

def test_coherent_plan_gives_no_hints():
    assert plan_notes(DINING_PLAN, DINING_ROOM, {}) == []


def test_pieces_outside_every_zone_are_reported():
    plan = {**DINING_PLAN, "zones": [zone("dining", "table", ["table", "chair1", "chair2"])]}
    notes = plan_notes(plan, DINING_ROOM, {})
    assert notes == ["pieces not part of any zone (give them a role, or remove them if that is allowed): "
                     "Dining Chair (chair3), Dining Chair (chair4)"]


def test_unknown_pending_and_missing_pieces_are_reported():
    plan = {"requirements": ["kitchen"], "composition": "",
            "zones": [zone("kitchen", "sink-item", ["sink-item", "lamp-item"], missing=["kitchen cabinets"])]}
    notes = plan_notes(plan, [], INVENTORY)
    assert any("neither in the working layout nor in the available inventory: sink-item" in n for n in notes)
    assert any("not placed yet: lamp-item" in n for n in notes)
    assert any("still lacks: kitchen cabinets" in n for n in notes)


def test_planned_inventory_item_counts_as_placed_once_added():
    lamp = Furniture(id="new-1-1", name="Floor Lamp", assetId="lamp-item", x=0, z=0)
    plan = {"requirements": [], "composition": "", "zones": [zone("reading", None, ["lamp-item"])]}
    assert plan_notes(plan, [lamp], INVENTORY) == []


# --- workflow ---------------------------------------------------------------------------------------------

def test_plan_is_stored_and_shown_in_later_steps_and_to_the_critic(fake_llm):
    fake = fake_llm(decide(*AROUND_TABLE, action="request_critique", finish=False, plan=DINING_PLAN),
                    decide())  # second step keeps the plan (plan=None)
    _, result = run("A dining area with one table and four chairs.", furniture=DINING_ROOM)

    assert "No design plan yet" in designer_text(fake, 0)
    later = designer_text(fake, 1)
    assert "Your design plan" in later and "two chairs per long side" in later
    assert "Plan vs. working layout" not in later  # every chair belongs to the dining zone
    critic_text = fake.critic_calls[-1][1]["content"]
    critic_text = critic_text if isinstance(critic_text, str) else critic_text[0]["text"]
    assert "The designer's plan" in critic_text
    first = completed(result, "interior_designer")[0]["payload"]["output"]
    assert first["design_plan"]["zones"][0]["anchor"] == "table"
    assert "planned 1 zone(s)" in first["workflow_log"][0]
    assert "design_plan" not in completed(result, "interior_designer")[1]["payload"]["output"]
    assert result.result["kind"] == "design"


def test_small_edit_needs_no_plan(fake_llm):
    fake = fake_llm(decide(P("chair1", -2.5, 2.5)))
    _, result = run("Move one chair a little to the right.", furniture=DINING_ROOM)
    assert len(fake.designer_calls) == 1
    output = completed(result, "interior_designer")[0]["payload"]["output"]
    assert "design_plan" not in output and "planned" not in output["workflow_log"][0]
    assert [p.id for p in result.result["placements"]] == ["table", "chair1", "chair2", "chair3", "chair4"]
