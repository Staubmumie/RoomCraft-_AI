import math

import pytest

from design.geometry import Footprint, RoomBounds, distance, overlap_depth
from design.state import Furniture, Placement, Room
from design.validator import DEFAULT_CONFIG, ValidatorConfig, validate_layout

ROOM = Room(width=8, depth=6, height=2.8)


def item(id, width=1.0, depth=1.0, x=0.0, z=0.0, rotation_deg=0.0):
    return Furniture(id=id, name=id.title(), width=width, depth=depth, height=1.0, x=x, z=z,
                     rotation=math.radians(rotation_deg))


def place(id, x, z, rotation_deg=0.0):
    return Placement(id=id, x=x, z=z, rotation=math.radians(rotation_deg))


def assert_clean(result, room, furniture, proposed, config=DEFAULT_CONFIG):
    """The returned layout has no overlaps, nothing outside the room, and every piece is
    within the correction limit of where the designer put it."""
    dims = {f.id: f for f in furniture}
    bounds = RoomBounds(room.width, room.depth)
    prints = {p.id: Footprint(p.x, p.z, dims[p.id].width, dims[p.id].depth, p.rotation) for p in result.layout}
    for fp in prints.values():
        assert bounds.violation(fp) <= config.boundary_tolerance
    ids = list(prints)
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            assert overlap_depth(prints[a], prints[b]) <= config.collision_tolerance, (a, b)
    proposed_by_id = {p.id: p for p in proposed}
    for p in result.layout:
        origin = proposed_by_id.get(p.id) or dims[p.id]
        assert distance((p.x, p.z), (origin.x, origin.z)) <= max(config.max_correction, config.max_collision_push) + 1e-6
        assert math.isclose(math.remainder(p.rotation - origin.rotation, 2 * math.pi), 0, abs_tol=1e-9)


def test_valid_layout_is_unchanged():
    furniture = [item("sofa", 2, 0.9), item("table", 1, 0.6)]
    proposed = [place("sofa", 0, -2.5), place("table", 0, -1)]
    result = validate_layout(ROOM, furniture, proposed)
    assert result.status == "valid"
    assert result.corrections == [] and result.errors == []
    assert [(p.x, p.z) for p in result.layout] == [(0, -2.5), (0, -1)]


def test_small_boundary_violation_is_corrected():
    furniture = [item("cabinet", 1, 0.5)]
    proposed = [place("cabinet", 3.505, 0)]  # 5 mm beyond the right wall
    result = validate_layout(ROOM, furniture, proposed)
    assert result.status == "auto_corrected"
    assert result.stats.boundaryViolations == 1
    assert result.layout[0].x == pytest.approx(3.5)
    assert result.corrections[0].distance == pytest.approx(0.005, abs=1e-4)
    assert_clean(result, ROOM, furniture, proposed)


def test_large_boundary_violation_is_rejected():
    furniture = [item("cabinet", 1, 0.5)]
    result = validate_layout(ROOM, furniture, [place("cabinet", 3.9, 0)])
    assert result.status == "invalid"
    (error,) = result.errors
    assert error.type == "out_of_bounds"
    assert error.measurements["violation"] == pytest.approx(0.4)
    assert error.measurements["requiredCorrection"] == pytest.approx(0.4)
    assert error.retryable


def test_small_overlap_is_corrected():
    furniture = [item("a"), item("b")]
    proposed = [place("a", 0, 0), place("b", 0.95, 0)]
    result = validate_layout(ROOM, furniture, proposed)
    assert result.status == "auto_corrected"
    assert result.stats.collisions == 1
    assert len(result.corrections) == 1
    assert result.corrections[0].distance == pytest.approx(0.052, abs=1e-3)
    assert_clean(result, ROOM, furniture, proposed)


def test_large_overlap_is_rejected():
    furniture = [item("a"), item("b")]
    result = validate_layout(ROOM, furniture, [place("a", 0, 0), place("b", 0.5, 0)])
    assert result.status == "invalid"
    (error,) = result.errors
    assert error.type == "collision" and error.furnitureIds == ["a", "b"]
    assert error.measurements["overlapDepth"] == pytest.approx(0.5)
    assert "limit" in error.message


def test_rotated_footprints_use_real_shape_not_bounding_box():
    # A long, thin piece at 45°: its axis-aligned box would cover (0.5, 0.5), the real footprint does not.
    furniture = [item("bench", 2, 0.4), item("stool", 0.4, 0.4)]
    result = validate_layout(ROOM, furniture, [place("bench", 0, 0, 45), place("stool", 0.5, 0.5)])
    assert result.status == "valid"


def test_rotated_collision_detected_and_small_one_corrected():
    furniture = [item("bench", 2, 0.4), item("stool", 0.4, 0.4)]
    # Along the bench's long axis: deep overlap
    deep = validate_layout(ROOM, furniture, [place("bench", 0, 0, 45), place("stool", 0.5, -0.5)])
    assert deep.status == "invalid" and deep.errors[0].type == "collision"
    # Just touching its long side (0.47 m from the centre line, needs 0.483 m): small overlap
    d = 0.47 / math.sqrt(2)
    proposed = [place("bench", 0, 0, 45), place("stool", d, d)]
    small = validate_layout(ROOM, furniture, proposed)
    assert small.status == "auto_corrected"
    assert_clean(small, ROOM, furniture, proposed)


def test_multiple_interacting_collisions_are_resolved_without_new_ones():
    furniture = [item("a"), item("b"), item("c")]
    proposed = [place("a", 0, 0), place("b", 0.95, 0), place("c", 1.9, 0)]
    result = validate_layout(ROOM, furniture, proposed)
    assert result.status == "auto_corrected"
    assert result.stats.collisions == 2
    assert_clean(result, ROOM, furniture, proposed)


def test_unsolvable_chain_is_reported_instead_of_pushing_through_walls():
    room = Room(width=3, depth=3, height=2.5)
    furniture = [item("a"), item("b"), item("c")]
    proposed = [place("a", -0.97, 0), place("b", 0, 0), place("c", 0.97, 0)]
    result = validate_layout(room, furniture, proposed)
    assert result.status == "invalid"
    assert all(e.type == "collision" for e in result.errors)
    bounds = RoomBounds(room.width, room.depth)
    for p in result.layout:  # partial corrections never left the room
        assert bounds.violation(Footprint(p.x, p.z, 1, 1, p.rotation)) <= DEFAULT_CONFIG.boundary_tolerance


def test_unknown_furniture_id():
    result = validate_layout(ROOM, [item("a")], [place("a", 0, 0), place("ghost", 1, 1)])
    assert result.status == "invalid"
    assert result.errors[0].type == "unknown_furniture"
    assert result.errors[0].furnitureIds == ["ghost"]


def test_duplicate_furniture_id():
    result = validate_layout(ROOM, [item("a")], [place("a", 0, 0), place("a", 2, 0)])
    assert result.status == "invalid"
    assert result.errors[0].type == "duplicate_furniture"


@pytest.mark.parametrize("x, rotation", [(math.nan, 0.0), (math.inf, 0.0), (0.0, math.nan)])
def test_invalid_numbers(x, rotation):
    result = validate_layout(ROOM, [item("a")], [Placement(id="a", x=x, z=0, rotation=rotation)])
    assert result.status == "invalid"
    assert result.errors[0].type == "invalid_value"


def test_missing_dimensions_are_not_retryable():
    furniture = [Furniture(id="a", name="A", width=None, depth=None, x=0, z=0)]
    result = validate_layout(ROOM, furniture, [place("a", 0, 0)])
    assert result.status == "invalid"
    assert result.errors[0].type == "invalid_dimensions"
    assert not result.retryable


def test_unchanged_furniture_is_preserved_and_not_moved():
    furniture = [item("a", x=-3, z=-2), item("b", x=0, z=0), item("c", x=3, z=2, rotation_deg=90)]
    proposed = [place("a", 0.95, 0)]  # only "a" moves, slightly into the unchanged "b"
    result = validate_layout(ROOM, furniture, proposed)
    assert [p.id for p in result.layout] == ["a", "b", "c"]
    by_id = {p.id: p for p in result.layout}
    assert (by_id["b"].x, by_id["b"].z) == (0, 0)
    assert (by_id["c"].x, by_id["c"].z) == (3, 2)
    assert by_id["c"].rotation == pytest.approx(math.pi / 2)
    assert result.status == "auto_corrected"
    assert [c.furnitureId for c in result.corrections] == ["a"]
    assert_clean(result, ROOM, furniture, proposed)


def test_correction_limit_is_configurable():
    furniture = [item("a"), item("b")]
    proposed = [place("a", 0, 0), place("b", 0.8, 0)]  # 0.2 m overlap
    strict = ValidatorConfig(max_collision_push=0.15)
    assert validate_layout(ROOM, furniture, proposed, config=strict).status == "invalid"
    relaxed = ValidatorConfig(max_correction=0.3, max_collision_push=0.3)
    result = validate_layout(ROOM, furniture, proposed, config=relaxed)
    assert result.status == "auto_corrected"
    assert_clean(result, ROOM, furniture, proposed, config=relaxed)
