"""Where screenshots are stored: one folder per render inside the run's folder, a content-addressed
cache shared by all runs, a manifest per run, and an API that only serves valid run screenshots.
The browser is faked; tests/test_rendering.py renders for real."""
import asyncio
import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rendering.service as service
from design.state import Room
from rendering.api import router
from rendering.service import RenderSettings, render_room
from tests.conftest import png_bytes

ROOM = Room(width=4, depth=3, height=2.5)
SETTINGS = RenderSettings(image_size=8)
RUN_ID = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def renders(tmp_path, monkeypatch):
    """Temporary render store and a fake browser; returns the list of browser renders."""
    monkeypatch.setattr(service, "DATA_DIR", tmp_path)
    monkeypatch.setattr(service, "RENDERS_DIR", tmp_path / "renders")
    calls = []

    async def fake_browser(config, settings, stage):
        calls.append(config)
        url = "data:image/png;base64," + base64.b64encode(png_bytes(size=settings.image_size)).decode()
        return {"images": {"top": url, "perspective": url},
                "stats": {"top": {"contrast": 10}, "perspective": {"contrast": 10}}}
    monkeypatch.setattr(service, "_render_in_browser", fake_browser)
    return calls


def render(room=ROOM, **kwargs):
    return asyncio.run(render_room(room, [], SETTINGS, **kwargs))


def test_each_render_gets_a_folder_in_its_run(renders):
    first = render(run_id=RUN_ID, layout_version=0, trigger="initial")
    second = render(run_id=RUN_ID, layout_version=3, trigger="validation")  # identical room: cache hit
    run_dir = service.RENDERS_DIR / "runs" / RUN_ID
    assert first.folder == f"renders/runs/{RUN_ID}/01_v0_initial"
    assert second.folder == f"renders/runs/{RUN_ID}/02_v3_validation"
    assert first.images["top_view"] == f"{first.folder}/top.png"
    for result in (first, second):
        for ref in result.images.values():
            assert (service.DATA_DIR / ref).read_bytes().startswith(b"\x89PNG")
    assert not first.cached and second.cached and len(renders) == 1  # rendered once, copied into the run
    assert (service.RENDERS_DIR / "cache" / first.key / "top.png").is_file()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [(e["seq"], e["folder"], e["layoutVersion"], e["trigger"], e["cached"]) for e in manifest] == [
        (1, "01_v0_initial", 0, "initial", False), (2, "02_v3_validation", 3, "validation", True)]
    assert manifest[0]["cacheKey"] == first.key


def test_a_cache_hit_from_another_run_gets_its_own_folder(renders):
    render(run_id=RUN_ID, trigger="initial")
    other = render(run_id="f" * 32, trigger="initial")
    assert other.cached and other.folder == f"renders/runs/{'f' * 32}/01_v0_initial"
    assert len(renders) == 1


def test_renders_without_a_valid_run_go_to_the_adhoc_folder(renders):
    assert render().folder == "renders/runs/_adhoc/01_v0_adhoc"
    assert render(run_id="../../evil", trigger="nonsense").folder == "renders/runs/_adhoc/02_v0_adhoc"


def test_api_serves_run_screenshots_and_rejects_everything_else(renders):
    result = render(run_id=RUN_ID, trigger="initial")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    ok = client.get(f"/api/{result.images['top_view']}")  # the frontend uses /api/<ref>
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    assert ok.content.startswith(b"\x89PNG")

    (service.RENDERS_DIR / "runs" / RUN_ID / "01_v0_initial" / "secret.txt").write_text("x")
    for path in [
        f"/api/renders/runs/{RUN_ID}/01_v0_initial/secret.txt",       # not a view file
        f"/api/renders/runs/{RUN_ID}/01_v0_initial/missing.png",
        f"/api/renders/runs/{RUN_ID}/..%2F..%2Fcache/top.png",          # traversal in the folder
        f"/api/renders/runs/..%2F..%2F/01_v0_initial/top.png",         # traversal in the run id
        f"/api/renders/runs/{RUN_ID}/02_v0_initial/top.png",           # folder that does not exist
        f"/api/renders/runs/{RUN_ID}/manifest.json/top.png",
        f"/api/renders/cache/{result.key}/top.png",                     # the cache is not served
    ]:
        assert client.get(path).status_code == 404, path
