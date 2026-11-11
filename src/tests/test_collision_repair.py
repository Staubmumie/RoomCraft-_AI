"""Collision repair: actionable validator feedback (ids, severity, separating moves with their
outcome), the larger automatic push for the smaller piece of a pair, and the repair progress the
designer sees across validations."""
import asyncio
import math

import pytest

from design.interior_designer import collision_clusters, failed_repairs, repair_progress_text
from design.layout_validator import layout_validator
from design.runner import execute_design_run
from design.state import Furniture, Placement, Room
from design.validator import validate_layout
from observability.store import RunStore
from tests.conftest import P, decide

ROOM = Room(width=8, depth=6, height=2.8)


def piece(id, w, d, h=0.8, x=0.0, z=0.0, y=0.0, rot=0.0, name=None):
    return Furniture(id=id, name=name or id.title(), width=w, depth=d, height=h, x=x, y=y, z=z,
                     rotation=math.radians(rot))


def validate(*pieces, moved=None):
    """All pieces count as placed by the designer unless `moved` names the ones that are."""
    proposed = [Placement(id=p.id, x=p.x, y=p.y, z=p.z, rotation=p.rotation)
                for p in pieces if moved is None or p.id in moved]
    return validate_layout(ROOM, list(pieces), proposed)


def at(result, id):
    return next(p for p in result.layout if p.id == id)


# 1. Two chairs overlapping by 20 cm: the larger automatic push separates them.
def test_two_chairs_overlapping_20cm_are_separated_automatically():
    result = validate(piece("chair_a", 0.5, 0.5), piece("chair_b", 0.5, 0.5, x=0.3))
    assert result.status == "auto_corrected"
    (correction,) = result.corrections
    assert correction.distance == pytest.approx(0.202, abs=1e-3)
    assert "larger push" in correction.reason


# 2. A chair 35 cm inside a table: a design decision with a calculated, clear option.
def test_chair_35cm_in_a_table_gets_a_calculated_clear_option():
    result = validate(piece("table", 1.6, 0.9, 0.75), piece("chair", 0.5, 0.5, 0.9, z=0.35))
    assert result.status == "invalid"
    (error,) = result.errors
    assert error.type == "collision" and error.furnitureIds == ["table", "chair"]
    assert error.severity == "large"
    chair_out = next(o for o in error.options if o.furnitureId == "chair" and o.move == "+z")
    assert chair_out.distance == pytest.approx(0.352, abs=1e-3)
    assert chair_out.target["z"] == pytest.approx(0.702, abs=1e-3)
    assert chair_out.outcome == "clear"


# 3. Chairs crowding a table: every collision is reported with ids and options that name the neighbours.
def test_clustered_chairs_report_options_that_would_hit_neighbours():
    table = piece("table", 1.2, 0.8, 0.75)
    chairs = [piece(f"chair{i}", 0.5, 0.5, 0.9, x=x, z=z) for i, (x, z) in
              enumerate([(-0.15, 0.3), (0.15, 0.3), (-0.15, -0.3), (0.15, -0.3)])]  # 0.35 m in the table, 0.2 m in each other
    result = validate(table, *chairs)
    assert result.status == "invalid"
    assert all(e.options and e.severity for e in result.errors if e.type == "collision")
    outcomes = [o.outcome for e in result.errors for o in e.options]
    assert any(o.startswith("hits") for o in outcomes)  # moving one chair alone runs into another
    assert all("'" in o and "(" in o for o in outcomes if o.startswith("hits"))  # names + ids
    (cluster,) = collision_clusters(result.errors)
    assert cluster == {"table", "chair0", "chair1", "chair2", "chair3"}


# 4. Near a wall: the shortest separation would leave the room; it is never applied and is marked as such.
def test_shortest_move_through_the_wall_is_not_applied_and_flagged():
    cabinet = piece("cabinet", 1.0, 0.5, 0.9, z=-2.2)          # confirmed, not moved
    chair = piece("chair", 0.6, 0.6, 0.9, z=-2.65)              # squeezed between wall and cabinet, 0.1 m overlap
    result = validate(cabinet, chair, moved={"chair"})
    assert result.status == "invalid"
    (error,) = result.errors
    through_wall = next(o for o in error.options if o.furnitureId == "chair" and o.move == "-z")
    assert through_wall.outcome.startswith("leaves the room")
    assert any(o.outcome == "clear" for o in error.options)       # e.g. the cabinet forward, or the chair sideways
    assert at(result, "chair").z - 0.3 >= -3.0 - 1e-6              # nothing was pushed through the wall


# 5. A crowded room after repeated failed repairs: the advice escalates to removing an optional piece.
def _entry(version, unresolved, severity, problems=None, strategy="shift"):
    return {"version": version, "status": "invalid", "strategy": strategy, "unresolved": unresolved,
            "collisions": unresolved, "severity": severity, "maxDepth": severity,
            "problems": problems or {"collision:a|b": severity}, "positions": {"a": [0, 0, 0, 0], "b": [0.5, 0, 0, 0]}}


class _Invalid:
    status = "invalid"


def _state(history, removal):
    return {"request": "x", "room": ROOM, "furniture": [piece("a", 1, 1), piece("b", 1, 1, x=0.5)],
            "repair_history": history, "validation": _Invalid(), "options": {"allow_removal": removal},
            "counters": {}}


def test_repeated_failed_repairs_escalate_to_removing_an_optional_piece():
    history = [_entry(1, 3, 0.9), _entry(2, 3, 0.88), _entry(3, 3, 0.9)]
    assert failed_repairs(history) == 2
    text = repair_progress_text(_state(history, removal=True))
    assert "NOT improved" in text and "remove an optional piece" in text
    assert "remove an optional piece" not in repair_progress_text(_state(history, removal=False))


def test_improvement_resets_the_failed_counter():
    history = [_entry(1, 3, 0.9), _entry(2, 3, 0.9), _entry(3, 1, 0.2)]
    assert failed_repairs(history) == 0
    assert "-> improved" in repair_progress_text(_state(history, removal=False))


# 6. + 7. Intentional overlaps stay valid.
def test_chair_on_a_rug_is_valid():
    rug = piece("rug", 2.0, 1.4, 0.02)
    assert validate(rug, piece("chair", 0.5, 0.5, 0.9, y=0.0)).status == "valid"


def test_tv_on_a_cabinet_is_valid():
    cabinet = piece("cabinet", 1.4, 0.45, 0.7, z=-2.7)
    tv = piece("tv", 1.1, 0.25, 0.65, z=-2.7, y=0.7)
    assert validate(cabinet, tv).status == "valid"


def test_a_cabinet_carrying_a_tv_is_not_pushed_far():
    sofa = piece("sofa", 2.2, 0.95, 0.85, z=-1.85)
    cabinet = piece("cabinet", 1.4, 0.45, 0.7, z=-2.3)  # 0.2 m into the sofa
    tv = piece("tv", 1.1, 0.25, 0.65, z=-2.3, y=0.7)
    assert validate(sofa, cabinet, moved={"cabinet"}).status == "auto_corrected"  # nothing on it: pushed
    assert validate(sofa, cabinet, tv, moved={"cabinet"}).status == "invalid"     # TV on it: designer decides


# 8. A repair that fixes one collision but causes another is detected as not improving.
def test_repair_that_moves_the_collision_elsewhere_is_reported():
    furniture = [piece("a", 1, 1, x=-2), piece("b", 1, 1, x=0), piece("c", 1, 1, x=2)]
    state = {"room": ROOM, "furniture": furniture, "repair_strategy": "shift",
             "proposed_layout": [furniture[0].model_copy(update={"x": -0.5}), furniture[1], furniture[2]],
             "layout_version": 1}
    first = layout_validator(state)  # a overlaps b by 0.5 m
    state |= {"proposed_layout": [furniture[0], furniture[1].model_copy(update={"x": 1.5}), furniture[2]],
              "layout_version": 2, "validation": first["validation"], "repair_history": first["repair_history"]}
    second = layout_validator(state)  # a fixed, but b now overlaps c by 0.5 m
    state |= {"validation": second["validation"], "repair_history": first["repair_history"] + second["repair_history"],
              "validated_layout_version": second["validated_layout_version"]}
    assert failed_repairs(state["repair_history"]) == 1
    text = repair_progress_text(state)
    assert "NOT improved" in text
    assert "new since the previous validation" in text and "'C' (c)" in text
    assert "do not repeat it" in text


# Measurable improvement in a full run: the designer sees ids, options and progress, and the run ends valid.
def test_designer_gets_actionable_feedback_and_progress_in_a_run(fake_llm):
    furniture = [Furniture(id="t", name="Table", width=1.6, depth=0.9, height=0.75, x=0, z=0),
                 Furniture(id="c1", name="Chair", width=0.5, depth=0.5, height=0.9, x=-2, z=2),
                 Furniture(id="c2", name="Chair", width=0.5, depth=0.5, height=0.9, x=2, z=2)]
    crowded = [P("c1", -0.3, 0.35), P("c2", 0.3, 0.35)]       # both 0.35 m inside the table, too close together
    nudged = [P("c1", -0.32, 0.37), P("c2", 0.32, 0.37)]      # a few cm: still colliding, hardly better
    regrouped = [P("c1", -0.4, 0.75), P("c2", 0.4, 0.75)]     # clear of the table and of each other
    fake = fake_llm(decide(*crowded), decide(*nudged, strategy="shift"), decide(*regrouped, strategy="regroup"))
    store = RunStore()
    run = store.create("design")
    asyncio.run(execute_design_run(store, run, {"request": "Dining for two", "model": "fake", "room": ROOM,
                                                "furniture": furniture, "options": {}}))
    second = fake.designer_calls[1][1]["content"][0]["text"]
    assert "(t, c1)" in second and "option: move c1" in second and "Repair progress" in second
    third = fake.designer_calls[2][1]["content"][0]["text"]
    assert "Unsuccessful repair attempts in a row: 1" in third and "you moved" in third
    assert run.events[-1]["payload"]["result"]["validation"]["status"] in ("valid", "auto_corrected")
