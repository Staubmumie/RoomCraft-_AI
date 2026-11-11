"""Real rendering: a live FastAPI server serves the built renderer page and test assets, a headless
browser renders, and the PNGs are checked pixel by pixel.

Needs `npm run build` in frontend/ and a Chromium-based browser (Playwright's own, Edge or Chrome);
skipped otherwise. Test furniture are single-colour unlit boxes, so they are easy to find in images.
"""
import asyncio
import json
import math
import socket
import struct
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from PIL import Image

import inventory
import rendering.service as service
from design.runner import execute_design_run
from tests.conftest import P, decide
from design.state import Furniture, Room
from main import FRONTEND_DIST, app
from observability.store import RunStore
from rendering.service import RenderError, RenderSettings, render_room

pytestmark = pytest.mark.skipif(not (FRONTEND_DIST / "render.html").exists(),
                                reason="renderer not built (run npm run build in frontend/)")

ROOM = Room(width=8, depth=6, height=2.8)
BACKGROUND = (244, 244, 244)
HALF = max(ROOM.width, ROOM.depth) / 2 + 0.5  # top view half extent, see frontend/src/render/cameras.js
PX_PER_M = 512 / HALF
RED, BLUE = (1, 0, 0), (0, 0, 1)


# --- test assets ---------------------------------------------------------------------------

def box_gltf(width, height, depth, color) -> tuple[dict, bytes]:
    hw, hd = width / 2, depth / 2
    corners = [(-hw, 0, -hd), (hw, 0, -hd), (hw, height, -hd), (-hw, height, -hd),
               (-hw, 0, hd), (hw, 0, hd), (hw, height, hd), (-hw, height, hd)]
    faces = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3,
             1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2]
    data = struct.pack("<24f", *[c for corner in corners for c in corner]) + struct.pack("<36H", *faces)
    gltf = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [*color, 1]}, "doubleSided": True,
                       "extensions": {"KHR_materials_unlit": {}}}],
        "buffers": [{"byteLength": len(data)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 96, "target": 34962},
                        {"buffer": 0, "byteOffset": 96, "byteLength": 72, "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                       "min": [-hw, 0, -hd], "max": [hw, height, hd]},
                      {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"}],
    }
    return gltf, data


def glb(gltf: dict, data: bytes) -> bytes:
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(data), 0x004E4942) + data
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


@pytest.fixture
def assets(tmp_path, monkeypatch):
    """Temporary inventory and render store; returns add(name, size, color, fmt) -> inventory id."""
    monkeypatch.setattr(inventory, "DATA_DIR", tmp_path)
    monkeypatch.setattr(inventory, "ASSETS_DIR", tmp_path / "inventory")
    monkeypatch.setattr(inventory, "INDEX_FILE", tmp_path / "inventory.json")
    monkeypatch.setattr(service, "DATA_DIR", tmp_path)
    monkeypatch.setattr(service, "RENDERS_DIR", tmp_path / "renders")
    index = {}

    def add(name, size, color, fmt="glb"):
        item_id = f"test-{name}"
        folder = tmp_path / "inventory" / item_id
        folder.mkdir(parents=True)
        gltf, data = box_gltf(*size, color)
        if fmt == "glb":
            model = "box.glb"
            (folder / model).write_bytes(glb(gltf, data))
        else:
            model = "box.gltf"
            gltf["buffers"][0]["uri"] = "box.bin"
            (folder / "box.bin").write_bytes(data)
            (folder / model).write_text(json.dumps(gltf))
        index[item_id] = {"id": item_id, "assetId": f"test:{name}", "name": name, "format": fmt,
                          "modelPath": model, "modelUrl": f"/api/inventory/{item_id}/files/{model}",
                          "thumbnailUrl": None, "dimensions": dict(zip(("width", "height", "depth"), size)),
                          "sizeBytes": 1, "downloadedAt": "2026-01-01T00:00:00+00:00"}
        inventory.save_index(index)
        return item_id
    return add


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


@pytest.fixture
def settings(server):
    return RenderSettings(page_url=f"{server}/renderer/render.html", timeout_s=60)


def furniture(item_id, name, x, z, rotation_deg=0.0, size=(1, 1, 1)):
    return Furniture(id=name, name=name, assetId=item_id, width=size[0], height=size[1], depth=size[2],
                     scale=1, x=x, z=z, rotation=math.radians(rotation_deg))


def render(furn, settings, room=ROOM):
    try:
        return asyncio.run(render_room(room, furn, settings))
    except RenderError as exc:
        if exc.code == "browser_start_failed":
            pytest.skip(str(exc))
        raise


# --- image helpers ---------------------------------------------------------------------------

def load(ref) -> Image.Image:
    return Image.open(service.DATA_DIR / ref).convert("RGB")


def pixels_where(image, predicate):
    w, h = image.size
    data = image.load()
    return [(x, y) for y in range(0, h, 2) for x in range(0, w, 2) if predicate(data[x, y])]


def bbox(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def is_red(p): return p[0] > 200 and p[1] < 60 and p[2] < 60
def is_blue(p): return p[2] > 200 and p[0] < 60 and p[1] < 60
def not_background(p): return max(abs(a - b) for a, b in zip(p, BACKGROUND)) > 6


def to_px(x, z):
    return 512 + x * PX_PER_M, 512 + z * PX_PER_M


# --- tests ---------------------------------------------------------------------------------------

def test_empty_room_top_view_is_centred_and_framed(assets, settings):
    result = render([], settings)
    assert set(result.images) == {"top_view", "perspective_view"}
    top = load(result.images["top_view"])
    assert top.size == (1024, 1024)
    left, upper, right, lower = bbox(pixels_where(top, not_background))
    # floor spans x -4..4, z -3..3; walls are 0.1 m thick, the back wall overlaps both side ends
    assert left == pytest.approx(to_px(-4.1, 0)[0], abs=4)
    assert right == pytest.approx(to_px(4.1, 0)[0], abs=4)
    assert upper == pytest.approx(to_px(0, -3.1)[1], abs=4)
    assert lower == pytest.approx(to_px(0, 3)[1], abs=4)


def test_perspective_view_shows_whole_room_without_clipping(assets, settings):
    result = render([], settings)
    image = load(result.images["perspective_view"])
    left, upper, right, lower = bbox(pixels_where(image, not_background))
    assert left > 5 and upper > 5 and right < 1018 and lower < 1018  # nothing cut off
    assert right - left > 600  # and the room fills the frame


def test_furniture_positions_rotations_and_both_formats(assets, settings):
    red = assets("red", (1.0, 0.8, 0.4), RED, fmt="glb")
    blue = assets("blue", (0.6, 0.5, 0.6), BLUE, fmt="gltf")
    furn = [furniture(red, "bench", -2, 1, rotation_deg=90, size=(1.0, 0.8, 0.4)),
            furniture(blue, "cube", 2, -1, size=(0.6, 0.5, 0.6))]
    result = render(furn, settings)
    top = load(result.images["top_view"])

    left, upper, right, lower = bbox(pixels_where(top, is_red))
    centre = ((left + right) / 2, (upper + lower) / 2)
    assert centre == pytest.approx(to_px(-2, 1), abs=6)
    # 1.0 × 0.4 m turned by 90°: now 0.4 m wide and 1.0 m deep in the image
    assert right - left == pytest.approx(0.4 * PX_PER_M, abs=8)
    assert lower - upper == pytest.approx(1.0 * PX_PER_M, abs=8)

    left, upper, right, lower = bbox(pixels_where(top, is_blue))
    assert ((left + right) / 2, (upper + lower) / 2) == pytest.approx(to_px(2, -1), abs=6)

    perspective = load(result.images["perspective_view"])
    assert len(pixels_where(perspective, is_red)) > 100
    assert len(pixels_where(perspective, is_blue)) > 100


def test_y_positions_floor_stacking_and_height(assets, settings):
    """y = bottom of the bounding box: a box at y=0 starts at the floor, a box at y=top stands on it."""
    blue = assets("blue", (1.0, 1.0, 1.0), BLUE)
    red = assets("red", (0.5, 0.5, 0.5), RED)
    on_floor = render([furniture(red, "cube", 0, 0, size=(0.5, 0.5, 0.5))], settings)
    lifted = render([furniture(red, "cube", 0, 0, size=(0.5, 0.5, 0.5)).model_copy(update={"y": 1.0})], settings)
    floor_box = bbox(pixels_where(load(on_floor.images["perspective_view"]), is_red))
    lifted_box = bbox(pixels_where(load(lifted.images["perspective_view"]), is_red))
    assert lifted_box[3] < floor_box[3] - 60  # the lifted cube is drawn clearly higher
    # top view: height does not change where a piece is seen from above
    centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    assert centre(bbox(pixels_where(load(lifted.images["top_view"]), is_red))) == pytest.approx(to_px(0, 0), abs=6)

    stacked = render([furniture(blue, "cabinet", 0, 0, size=(1.0, 1.0, 1.0)),
                      furniture(red, "cube", 0, 0, size=(0.5, 0.5, 0.5)).model_copy(update={"y": 1.0})], settings)
    top = load(stacked.images["top_view"])
    assert centre(bbox(pixels_where(top, is_red))) == pytest.approx(to_px(0, 0), abs=6)  # on top: visible from above
    assert len(pixels_where(top, is_blue)) > 100  # the cabinet around it is still visible
    perspective = load(stacked.images["perspective_view"])
    red_box, blue_box = bbox(pixels_where(perspective, is_red)), bbox(pixels_where(perspective, is_blue))
    assert red_box[3] <= blue_box[3] - 40 and red_box[1] < blue_box[1]  # the cube sits above the cabinet


def test_screenshots_are_cached_per_exact_configuration(assets, settings, monkeypatch):
    browser_runs = []
    real = service._render_in_browser

    async def counting(*args, **kwargs):
        browser_runs.append(1)
        return await real(*args, **kwargs)
    monkeypatch.setattr(service, "_render_in_browser", counting)

    red = assets("red", (1, 1, 1), RED)
    first = render([furniture(red, "a", 0, 0)], settings)
    assert not first.cached
    for ref in first.images.values():
        assert (service.DATA_DIR / ref).read_bytes().startswith(b"\x89PNG")
    again = render([furniture(red, "a", 0, 0)], settings)
    assert again.cached and again.key == first.key
    # every render gets its own folder, a cache hit is a copy of the same images
    assert again.folder != first.folder
    assert (service.DATA_DIR / again.images["top_view"]).read_bytes() == (service.DATA_DIR / first.images["top_view"]).read_bytes()
    moved = render([furniture(red, "a", 1, 0)], settings)
    assert moved.key != first.key and not moved.cached
    assert len(browser_runs) == 2  # the identical room was not rendered twice
    assert set(Path(first.images["top_view"]).parts) != set(Path(moved.images["top_view"]).parts)


def test_render_does_not_change_its_input(assets, settings):
    red = assets("red", (1, 1, 1), RED)
    furn = [furniture(red, "a", 0.5, -0.5, rotation_deg=45)]
    before = [f.model_copy() for f in furn]
    render(furn, settings)
    assert furn == before


def test_missing_asset_is_rejected_before_rendering(assets, settings):
    red = assets("red", (1, 1, 1), RED)
    (inventory.ASSETS_DIR / red / "box.glb").unlink()
    with pytest.raises(RenderError) as err:
        asyncio.run(render_room(ROOM, [furniture(red, "a", 0, 0)], settings))
    assert err.value.code == "missing_asset"


def test_broken_model_file_is_reported(assets, settings):
    red = assets("red", (1, 1, 1), RED)
    (inventory.ASSETS_DIR / red / "box.glb").write_bytes(b"not a model")
    with pytest.raises(RenderError) as err:
        render([furniture(red, "a", 0, 0)], settings)
    assert err.value.code == "asset_load_failed"


def test_timeout_while_starting(assets, server):
    tight = RenderSettings(page_url=f"{server}/renderer/render.html", timeout_s=0.2)
    with pytest.raises(RenderError) as err:
        asyncio.run(render_room(ROOM, [], tight))
    assert err.value.code == "timeout"


def test_timeout_when_the_capture_stalls(assets):
    # A renderer page that never answers: the render must give up and close the browser.
    stalled = ("data:text/html,<script>window.roomcraftReady=true;"
               "window.roomcraftRender=()=>new Promise(()=>{})</script>")
    started = time.monotonic()
    with pytest.raises(RenderError) as err:
        render([], RenderSettings(page_url=stalled, timeout_s=6))
    assert err.value.code == "timeout"
    assert time.monotonic() - started < 15


def test_browser_start_failure(assets, server):
    broken = RenderSettings(page_url=f"{server}/renderer/render.html", browser_channels=("no-such-browser",))
    with pytest.raises(RenderError) as err:
        asyncio.run(render_room(ROOM, [], broken))
    assert err.value.code == "browser_start_failed"


def test_unreachable_renderer_page(assets):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again: nothing listens there
    with pytest.raises(RenderError) as err:
        render([], RenderSettings(page_url=f"http://127.0.0.1:{port}/render.html", timeout_s=60))
    assert err.value.code == "renderer_unavailable"


def test_webgl_failure(assets, settings, monkeypatch):
    monkeypatch.setattr(service, "BROWSER_ARGS", ["--disable-webgl", "--disable-3d-apis"])
    with pytest.raises(RenderError) as err:
        render([], settings)
    assert err.value.code == "webgl_unavailable"


def test_measure_model_reports_the_real_bounding_box(assets, settings):
    from rendering.service import measure_model
    box = assets("tall", (80, 180, 40), RED)  # a model built in centimetres
    try:
        size = asyncio.run(measure_model(f"/api/inventory/{box}/files/box.glb", settings))
    except RenderError as exc:
        if exc.code == "browser_start_failed":
            pytest.skip(str(exc))
        raise
    assert size == pytest.approx({"width": 80, "height": 180, "depth": 40}, abs=1e-3)


def test_full_workflow_with_real_rendering(assets, settings, monkeypatch, fake_llm):
    """Orchestrator and designer are faked, the renderer is real."""
    import design.render_scene as node

    fake = fake_llm(decide(P("a", -1, 0)))
    # use the real renderer (the fixture replaced it) with the test server's page
    monkeypatch.setattr(node, "render_room",
                        lambda room, furn, stage=None, **kwargs: render_room(room, furn, settings, stage, **kwargs))
    monkeypatch.setattr(service, "DATA_DIR", inventory.DATA_DIR)
    monkeypatch.setattr(service, "RENDERS_DIR", inventory.DATA_DIR / "renders")
    red = assets("red", (1, 1, 1), RED)

    store = RunStore()
    run = store.create("design")
    inputs = {"request": "cozy", "model": "fake", "room": ROOM, "furniture": [furniture(red, "a", 1, 1)]}
    asyncio.run(execute_design_run(store, run, inputs))
    if run.events[-1]["payload"].get("errorCode") == "browser_start_failed":
        pytest.skip("no browser")
    assert run.events[-1]["type"] == "workflow.completed", run.events[-1]["payload"]

    stages = [e["payload"]["stage"] for e in run.events if e["type"] == "node.progress" and e["node"] == "render_scene"]
    assert ["prepare", "cache", "browser", "load", "capture", "store", "done"] == [
        s for i, s in enumerate(stages) if s not in stages[:i]]
    images = [p for p in fake.designer_calls[0][1]["content"] if p["type"] == "image_url"]
    assert len(images) == 2
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,iVBOR") for p in images)
