"""render_scene in the workflow: state updates, images reaching the agents, vision checks,
failures. The renderer itself is faked here; tests/test_rendering.py renders for real."""
import asyncio
import base64

import pytest

import design.render_scene as render_scene_module
import design.scene_images as scene_images_module
from design.runner import execute_design_run
from design.state import Furniture, Room
from observability.store import RunStore
from rendering.service import RenderError
from tests.conftest import P, decide

ROOM = Room(width=8, depth=6, height=2.8)
FURNITURE = [
    Furniture(id="a", name="Sofa", width=1, depth=1, x=-2, z=0),
    Furniture(id="b", name="Table", width=1, depth=1, x=2, z=0),
]
CLEAN = [P("a", -1, 0), P("b", 1, 0)]
OVERLAPPING = [P("a", 0, 0), P("b", 0.5, 0)]


def run():
    store = RunStore()
    run = store.create("design")
    inputs = {"request": "Make it cozy", "model": "fake", "room": ROOM, "furniture": FURNITURE}
    asyncio.run(execute_design_run(store, run, inputs))
    return run


def image_parts(messages):
    content = messages[1]["content"]
    return [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"] if isinstance(content, list) else []


def test_state_holds_only_image_references_and_the_layout_version(fake_llm):
    fake = fake_llm(decide(*CLEAN))
    result = run()
    rendered = next(e for e in result.events if e["type"] == "node.completed" and e["node"] == "render_scene")
    output = rendered["payload"]["output"]
    folder = f"renders/runs/{result.id}/01_v0_initial"  # the run's folder, first render, first look
    assert output["scene_images"] == {"top_view": f"{folder}/top.png", "perspective_view": f"{folder}/perspective.png"}
    assert output["rendered_layout_version"] == 0 and fake.render_triggers == ["initial"]
    assert "base64" not in str(output)
    room, furniture = fake.render_calls[0]
    assert room == ROOM and furniture == FURNITURE  # the renderer got the room from the state


def test_designer_gets_both_images_with_the_text(fake_llm):
    fake = fake_llm(decide(*CLEAN))
    run()
    parts = image_parts(fake.designer_calls[0])
    assert len(parts) == 2
    for part in parts:
        url = part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG")


def test_validation_retries_do_not_render_again(fake_llm):
    fake = fake_llm(decide(*OVERLAPPING), decide(*CLEAN))
    run()
    assert len(fake.render_calls) == 1
    assert len(fake.designer_calls) == 2
    assert len(image_parts(fake.designer_calls[1])) == 2  # the same screenshots are sent again


def test_model_without_image_input_fails_clearly(fake_llm):
    fake = fake_llm(decide(*CLEAN))
    fake.vision = False
    result = run()
    assert fake.designer_calls == []
    assert result.events[-1]["type"] == "workflow.failed"
    assert "does not accept image input" in result.events[-1]["payload"]["error"]
    assert result.result is None


def test_explicit_text_only_fallback(fake_llm, monkeypatch):
    fake = fake_llm(decide(*CLEAN))
    fake.vision = False
    monkeypatch.setattr(scene_images_module, "TEXT_ONLY_FALLBACK", True)
    result = run()
    assert result.events[-1]["type"] == "workflow.completed"
    assert isinstance(fake.designer_calls[0][1]["content"], str)  # no images sent
    steps = [e["payload"].get("stage") for e in result.events if e["type"] == "node.progress"]
    assert "text_only" in steps


@pytest.mark.parametrize("code", ["browser_start_failed", "timeout", "missing_asset", "empty_image"])
def test_render_failure_stops_the_workflow(fake_llm, monkeypatch, code):
    fake = fake_llm(decide(*CLEAN))

    async def failing(room, furniture, stage=None, **kwargs):
        raise RenderError(code, f"simulated {code}")
    monkeypatch.setattr(render_scene_module, "render_room", failing)
    result = run()
    assert fake.designer_calls == []
    last = result.events[-1]
    assert last["type"] == "workflow.failed"
    assert last["payload"]["errorCode"] == code
    assert f"simulated {code}" in last["payload"]["error"]
    failed = [e for e in result.events if e["type"] == "node.failed"]
    assert [e["node"] for e in failed] == ["render_scene"]


def test_missing_image_file_is_not_silently_skipped(fake_llm, monkeypatch):
    fake = fake_llm(decide(*CLEAN))

    async def render_then_delete(room, furniture, stage=None, **kwargs):
        result = await fake.render(room, furniture, stage, **kwargs)
        (fake.renders_dir.parent / result.images["top_view"]).unlink()
        return result
    monkeypatch.setattr(render_scene_module, "render_room", render_then_delete)
    result = run()
    assert result.events[-1]["type"] == "workflow.failed"
    assert "image_missing" in result.events[-1]["payload"]["error"]
    assert fake.designer_calls == []


def test_render_scene_reports_a_completion_step(fake_llm):
    fake_llm(decide(*CLEAN))
    result = run()
    steps = [e for e in result.events if e["type"] == "node.progress" and e["node"] == "render_scene"]
    assert steps[-1]["payload"]["stage"] == "done"
    assert steps[-1]["payload"]["data"]["images"]["top_view"].endswith("top.png")
    assert "for interior_designer" in steps[-1]["payload"]["label"]


def test_designer_is_told_how_far_the_screenshots_are_behind():
    from design.interior_designer import _render_text

    images = {"top_view": "renders/runs/_adhoc/01_v0_initial/top.png"}
    base = {"room": ROOM, "furniture": FURNITURE, "scene_images": images, "rendered_layout_version": 0}
    assert _render_text({"room": ROOM, "furniture": FURNITURE}) is None
    assert "show your current working layout" in _render_text(base)

    nudged = [FURNITURE[0].model_copy(update={"x": -1.98}), FURNITURE[1].model_copy(update={"x": 1.0})]
    small = _render_text({**base, "proposed_layout": nudged, "layout_version": 1})
    assert "moved/turned 1 (Table)" in small and "Small change" in small

    rebuilt = [FURNITURE[0].model_copy(update={"rotation": 1.57}),
               Furniture(id="new-1-1", name="Lamp", width=0.3, depth=0.3, x=0, z=0)]
    big = _render_text({**base, "proposed_layout": rebuilt, "layout_version": 2})
    assert "added 1 (Lamp)" in big and "removed 1 (Table)" in big and "Substantial change" in big
    assert "follow automatically" in big  # the designer cannot request them itself

    current = _render_text({**base, "proposed_layout": rebuilt, "layout_version": 2,
                            "rendered_layout": rebuilt, "rendered_layout_version": 2})
    assert "show your current working layout" in current
