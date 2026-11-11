"""Y-axis support: y = height of a piece's bottom above the floor. Checks the permissive 3D validation
(rugs, stacking, wall mounting, real collisions, floating pieces), compatibility with data without y,
and that y travels through the designer, the working layout, the result and the render config."""
import math

import pytest

import design.interior_designer as designer_module
from design.interior_designer import DesignerDecision
from design.state import Furniture, Placement, Room
from design.validator import validate_layout
from rendering.service import build_render_config
from tests.conftest import P, decide
from tests.test_workflows import completed, final, run as run_workflow

ROOM = Room(width=8, depth=6, height=2.8, walls=["back (z = -3)", "left (x = -4)"])


def piece(id, width, depth, height, x=0.0, z=0.0, y=0.0, rotation_deg=0.0):
    return Furniture(id=id, name=id.replace("_", " ").title(), width=width, depth=depth, height=height,
                     x=x, y=y, z=z, rotation=math.radians(rotation_deg))


def check(*pieces, room=ROOM):
    """Validates the pieces at their own positions (all of them proposed, so all are movable)."""
    proposed = [Placement(id=p.id, x=p.x, y=p.y, z=p.z, rotation=p.rotation) for p in pieces]
    return validate_layout(room, list(pieces), proposed)


def types(result):
    return sorted({e.type for e in result.errors})


# --- 1. floor coverings ---------------------------------------------------------------------------

@pytest.mark.parametrize("chair_y", [0.0, 0.01])
def test_chair_on_a_rug_is_valid(chair_y):
    rug = piece("rug", 2.0, 1.5, 0.01)
    chair = piece("chair", 0.5, 0.5, 0.9, x=0.3, y=chair_y)
    assert check(rug, chair).status == "valid"


# --- 2. standing on furniture ---------------------------------------------------------------------

def test_tv_on_a_sideboard_is_valid():
    sideboard = piece("sideboard", 1.4, 0.45, 0.75, z=1.0)
    tv = piece("tv", 1.1, 0.25, 0.65, z=1.0, y=0.75)
    assert check(sideboard, tv).status == "valid"


@pytest.mark.parametrize("tv_y", [0.73, 0.78])  # slightly sunk in / hovering: imperfect model bounds
def test_small_height_errors_on_a_support_are_tolerated(tv_y):
    sideboard = piece("sideboard", 1.4, 0.45, 0.75, z=1.0)
    tv = piece("tv", 1.1, 0.25, 0.65, z=1.0, y=tv_y)
    assert check(sideboard, tv).status == "valid"


def test_slightly_floating_piece_is_lowered_onto_its_support():
    table = piece("table", 1.2, 0.8, 0.75)
    lamp = piece("table_lamp", 0.3, 0.3, 0.5, x=0.2, y=0.85)
    result = check(table, lamp)
    assert result.status == "auto_corrected"
    assert {p.id: p.y for p in result.layout}["table_lamp"] == pytest.approx(0.75)


def test_clearly_floating_piece_is_rejected():
    sideboard = piece("sideboard", 1.4, 0.45, 0.75, z=1.0)
    tv = piece("tv", 1.1, 0.25, 0.65, z=1.0, y=1.3)  # mid-air, no wall nearby
    result = check(sideboard, tv)
    assert result.status == "invalid" and types(result) == ["unsupported"]
    assert result.errors[0].retryable


def test_heavy_piece_cannot_stand_on_a_small_one():
    stool = piece("stool", 0.4, 0.4, 0.45)
    bed = piece("bed", 2.0, 1.6, 0.5, y=0.45)
    assert types(check(stool, bed)) == ["unsupported"]


# --- 3./4. wall-mounted ----------------------------------------------------------------------------

def test_picture_on_the_back_wall_above_a_sofa_is_valid():
    sofa = piece("sofa", 2.0, 0.9, 0.8, z=-2.55)
    picture = piece("picture", 1.0, 0.03, 0.7, z=-2.985, y=1.4)
    assert check(sofa, picture).status == "valid"


def test_dartboard_on_the_left_wall_is_valid():
    dartboard = piece("dartboard", 0.45, 0.05, 0.45, x=-3.975, z=0.5, y=1.5, rotation_deg=90)
    assert check(dartboard).status == "valid"


def test_dartboard_on_a_side_without_a_wall_is_floating():
    dartboard = piece("dartboard", 0.45, 0.05, 0.45, x=3.975, z=0.5, y=1.5, rotation_deg=-90)  # right side: no wall
    assert types(check(dartboard)) == ["unsupported"]


def test_wall_cabinet_above_a_counter_is_valid():
    counter = piece("counter", 1.8, 0.6, 0.9, z=-2.7)
    wall_cabinet = piece("wall_cabinet", 1.8, 0.35, 0.7, z=-2.825, y=1.45)
    assert check(counter, wall_cabinet).status == "valid"


# --- 5./6. real collisions vs. different heights -----------------------------------------------------

def test_two_cabinets_in_the_same_volume_collide():
    a = piece("cabinet_a", 1.0, 0.5, 1.8)
    b = piece("cabinet_b", 1.0, 0.5, 1.8, x=0.3)
    result = check(a, b)
    assert result.status == "invalid" and types(result) == ["collision"]


def test_same_footprint_at_different_heights_does_not_collide():
    desk = piece("desk", 1.2, 0.6, 0.75, z=-2.7)
    shelf = piece("wall_shelf", 1.0, 0.25, 0.3, z=-2.875, y=1.5)  # same x/z region, above the desk
    lamp = piece("desk_lamp", 0.2, 0.2, 0.45, x=0.4, z=-2.7, y=0.75)  # on the desk
    assert check(desk, shelf, lamp).status == "valid"


def test_lamp_inside_the_table_volume_collides():
    table = piece("table", 1.2, 0.8, 0.75)
    lamp = piece("table_lamp", 0.3, 0.3, 0.5, y=0.3)  # half sunk into the table top
    assert "collision" in types(check(table, lamp))


def test_small_bounding_box_overlap_is_tolerated():
    a = piece("chair_a", 0.5, 0.5, 0.9)
    b = piece("chair_b", 0.5, 0.5, 0.9, x=0.492)  # 8 mm overlap of imperfect model bounds
    assert check(a, b).status == "valid"


def test_piece_above_the_ceiling_is_out_of_bounds():
    shelf = piece("wall_shelf", 1.0, 0.25, 0.3, z=-2.875, y=2.7)
    assert types(check(shelf)) == ["out_of_bounds"]


# --- 7. compatibility ---------------------------------------------------------------------------------

def test_data_without_y_means_on_the_floor():
    old = Furniture.model_validate({"id": "a", "name": "Sofa", "width": 1, "depth": 1, "x": 0, "z": 0})
    assert old.y == 0.0
    assert Placement.model_validate({"id": "a", "x": 1, "z": 2, "rotation": 0}).y == 0.0


def test_pieces_without_height_keep_the_2d_collision_check():
    a = Furniture(id="a", name="A", width=1, depth=1, x=0, z=0)
    b = Furniture(id="b", name="B", width=1, depth=1, x=0.5, z=0, y=1.5)  # height unknown: blocks its footprint
    assert "collision" in types(check(a, b))


def test_room_without_walls_counts_every_side_as_wall():
    picture = piece("picture", 1.0, 0.03, 0.7, x=3.985, z=0, y=1.4, rotation_deg=90)
    assert check(picture, room=Room(width=8, depth=6, height=2.8)).status == "valid"


# --- 8. designer output with y -------------------------------------------------------------------------

def test_designer_schema_requires_y_for_every_position():
    schema = DesignerDecision.model_json_schema()["$defs"]
    assert "y" in schema["PlacementProposal"]["required"] and "y" in schema["NewPiece"]["required"]


def test_designer_places_a_tv_on_a_sideboard(fake_llm, monkeypatch):
    sideboard = Furniture(id="sb", name="Sideboard", assetId="sideboard-item", width=1.4, depth=0.45, height=0.75,
                          x=0, z=-2.775)
    fake = fake_llm(decide(add=[("tv-item", 0, -2.8, 0, 0.75)]))
    fake.inventory["tv-item"] = {"item_id": "tv-item", "name": "TV", "scale": 1.0, "width": 1.1, "depth": 0.2,
                                 "height": 0.65}

    async def categories():
        return {"tv-item": "decor", "sideboard-item": "sideboard"}
    monkeypatch.setattr(designer_module, "item_categories", categories)
    _, result = run_workflow("Put a TV on the sideboard.", add=True, furniture=[sideboard])

    context = fake.designer_calls[0][1]["content"]
    context = context if isinstance(context, str) else context[0]["text"]
    assert '"category": "sideboard"' in context and '"category": "decor"' in context
    assert result.result["validation"].status == "valid"
    tv = next(p for p in final(result)["placements"] if p["name"] == "TV")
    assert tv["y"] == 0.75
    # the working layout handed to the renderer for the designer carries the height as well
    designer_out = completed(result, "interior_designer")[0]["payload"]["output"]
    assert next(p for p in designer_out["proposed_layout"] if p["name"] == "TV")["y"] == 0.75


def test_layout_list_for_the_designer_shows_height_and_top(fake_llm):
    table = Furniture(id="t", name="Table", width=1.2, depth=0.8, height=0.75, x=0, z=0)
    fake = fake_llm(decide(P("t", 0.5, 0), finish=False), decide())  # validate, then a second step
    run_workflow("Shift the table a bit.", furniture=[table])
    text = fake.designer_calls[1][1]["content"][0]["text"]
    assert '"y": 0.0' in text and '"top": 0.75' in text


def test_render_config_carries_y(tmp_path, monkeypatch):
    import inventory
    folder = tmp_path / "inventory" / "lamp"
    folder.mkdir(parents=True)
    (folder / "lamp.glb").write_bytes(b"glb")
    monkeypatch.setattr(inventory, "ASSETS_DIR", tmp_path / "inventory")
    monkeypatch.setattr(inventory, "load_index", lambda: {"lamp": {"id": "lamp", "modelPath": "lamp.glb",
                                                                   "modelUrl": "/m/lamp.glb", "dimensions": None}})
    lamp = Furniture(id="l", name="Lamp", assetId="lamp", width=0.3, depth=0.3, height=0.5, scale=1, x=0, y=0.75, z=0)
    config = build_render_config(ROOM, [lamp])
    assert config["furniture"][0]["y"] == 0.75
