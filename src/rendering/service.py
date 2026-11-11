"""Renders a room configuration to a top-down and a perspective screenshot.

The scene is the frontend's own React Three Fiber code (frontend/src/render): a headless browser
(Playwright) opens the renderer page, hands it the configuration and gets PNGs back. It renders
exactly the configuration it is given, never the user's visible room.

Storage (data/renders/):
    cache/<key>/top.png, perspective.png      content-addressed: an identical room (same furniture,
                                              positions, assets, renderer version) is rendered once
    runs/<run_id>/<seq>_v<version>_<trigger>/ one folder per render of a workflow run (a cache hit
                                              is copied in), so a run can be followed render by render
    runs/<run_id>/manifest.json               seq, folder, layoutVersion, trigger, cacheKey, cached, ts
Callers only keep the relative paths of the run folder ("renders/runs/<run_id>/<folder>/top.png").
"""
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import inventory
from design.state import Furniture, Room

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RENDERS_DIR = DATA_DIR / "renders"
RENDERER_VERSION = 2  # bump when the rendered output changes; invalidates cached images (2: y positions)
VIEW_FILES = {"top_view": "top.png", "perspective_view": "perspective.png"}
ADHOC_RUN = "_adhoc"  # renders outside a workflow run (e.g. direct calls in tests)
TRIGGERS = ("initial", "validation", "critic", "adhoc")  # why a render was made; part of the folder name
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}|_adhoc")  # observability.store creates uuid4().hex ids
FOLDER_PATTERN = re.compile(r"(\d{2,})_v(\d+)_(" + "|".join(TRIGGERS) + r")")
PAGE_VIEWS = {"top_view": "top", "perspective_view": "perspective"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MIN_CONTRAST = 2.0  # luminance spread below this means an empty / single-colour image
BROWSER_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]  # WebGL without a GPU

StageFactory = Callable[[str, str], AbstractContextManager[dict]]


@dataclass(frozen=True)
class RenderSettings:
    page_url: str = field(default_factory=lambda: os.environ.get(
        "RENDER_PAGE_URL", "http://127.0.0.1:8000/renderer/render.html"))
    image_size: int = 1024
    timeout_s: float = 90
    # Playwright's own Chromium first, then installed browsers.
    browser_channels: tuple[str, ...] = ("chromium", "msedge", "chrome")


DEFAULT_SETTINGS = RenderSettings()


class RenderError(Exception):
    """code: invalid_room, missing_asset, browser_start_failed, renderer_unavailable,
    webgl_unavailable, asset_load_failed, capture_failed, timeout, empty_image,
    invalid_image, storage_failed, image_missing"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class RenderResult:
    key: str
    images: dict[str, str]  # view -> path relative to DATA_DIR (inside the run folder)
    cached: bool
    duration_ms: int
    folder: str = ""        # the run folder, relative to DATA_DIR


# --- configuration -----------------------------------------------------------------

def _finite(*values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def build_render_config(room: Room, furniture: list[Furniture], settings: RenderSettings = DEFAULT_SETTINGS) -> dict:
    """Validated description of what the renderer page should draw."""
    if not _finite(room.width, room.depth, room.height) or max(room.width, room.depth, room.height) > 100:
        raise RenderError("invalid_room", f"Invalid room dimensions: {room.width} × {room.depth} × {room.height} m.")
    index = inventory.load_index()
    items, problems = [], []
    for f in furniture:
        if not _finite(f.x, f.y, f.z, f.rotation):
            problems.append(f"'{f.name}' has an invalid position or rotation")
            continue
        item = index.get(f.assetId or "")
        if not item or not (inventory.ASSETS_DIR / item["id"] / item["modelPath"]).is_file():
            problems.append(f"model of '{f.name}' is missing from the inventory")
            continue
        scale = f.scale
        if not _finite(scale) or not scale or scale <= 0:
            dims = item.get("dimensions") or {}
            scale = f.width / dims["width"] if f.width and dims.get("width") else 1.0
        items.append({"id": f.id, "name": f.name, "modelUrl": item["modelUrl"],
                      "x": f.x, "y": f.y, "z": f.z, "rotation": f.rotation, "scale": round(scale, 6)})
    if problems:
        code = "missing_asset" if any("missing" in p for p in problems) else "invalid_room"
        raise RenderError(code, "Cannot render the room: " + "; ".join(problems) + ".")
    return {
        "room": {"width": room.width, "depth": room.depth, "height": room.height},
        "furniture": items,
        "imageSize": settings.image_size,
    }


def cache_key(config: dict) -> str:
    """Changes whenever anything visible changes: room, furniture, model files or renderer."""
    index = inventory.load_index()
    by_url = {item["modelUrl"]: item for item in index.values()}
    files = []
    for f in config["furniture"]:
        item = by_url[f["modelUrl"]]
        stat = (inventory.ASSETS_DIR / item["id"] / item["modelPath"]).stat()
        files.append([f["modelUrl"], stat.st_size, stat.st_mtime_ns])
    payload = json.dumps({"version": RENDERER_VERSION, "config": config, "files": files}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# --- images ----------------------------------------------------------------------------

def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise RenderError("invalid_image", "The renderer returned data that is not a PNG image.")
    return struct.unpack(">II", data[16:24])


def image_path(ref: str) -> Path:
    path = (DATA_DIR / ref).resolve()
    if not path.is_relative_to(RENDERS_DIR.resolve()):
        raise RenderError("image_missing", f"Invalid image reference: {ref}")
    return path


def image_data_url(ref: str) -> str:
    """Loads a stored screenshot as a data URL (for multimodal LLM input)."""
    try:
        data = image_path(ref).read_bytes()
    except OSError as exc:
        raise RenderError("image_missing", f"Screenshot {ref} could not be read: {exc}")
    _png_size(data)
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _refs(folder: str) -> dict[str, str]:
    """Image references of a folder relative to RENDERS_DIR (e.g. "cache/<key>")."""
    return {view: f"renders/{folder}/{name}" for view, name in VIEW_FILES.items()}


def run_image_path(run_id: str, folder: str, name: str) -> Path | None:
    """Path of a run screenshot for the API, or None for anything that is not a valid reference."""
    if not RUN_ID_PATTERN.fullmatch(run_id) or not FOLDER_PATTERN.fullmatch(folder) or name not in VIEW_FILES.values():
        return None
    try:
        return image_path(f"renders/runs/{run_id}/{folder}/{name}")
    except RenderError:
        return None


def _cached_images(key: str, size: int) -> dict[str, str] | None:
    refs = _refs(f"cache/{key}")
    try:
        if all(_png_size(image_path(ref).read_bytes()) == (size, size) for ref in refs.values()):
            return refs
    except (OSError, RenderError):
        pass
    return None


def _store(key: str, result: dict, size: int) -> dict[str, str]:
    images = {}
    for view, page_view in PAGE_VIEWS.items():
        data_url = (result.get("images") or {}).get(page_view, "")
        if not data_url.startswith("data:image/png;base64,"):
            raise RenderError("invalid_image", f"The renderer returned no {page_view} image.")
        data = base64.b64decode(data_url.split(",", 1)[1])
        if _png_size(data) != (size, size):
            raise RenderError("invalid_image", f"The {page_view} image has the wrong size.")
        contrast = (result.get("stats") or {}).get(page_view, {}).get("contrast", 0)
        if contrast < MIN_CONTRAST:
            raise RenderError("empty_image", f"The {page_view} image is empty (contrast {contrast:.1f}).")
        images[view] = data
    cache = RENDERS_DIR / "cache"
    target = cache / key
    try:
        cache.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="render-", dir=cache))
        for view, data in images.items():
            (staging / VIEW_FILES[view]).write_bytes(data)
        try:
            staging.rename(target)
        except OSError:
            if not target.exists():
                raise
            shutil.rmtree(staging, ignore_errors=True)  # a parallel run stored the same room first
    except OSError as exc:
        raise RenderError("storage_failed", f"Screenshots could not be stored: {exc}")
    return _refs(f"cache/{key}")


def _publish(key: str, run_id: str | None, layout_version: int, trigger: str, cached: bool) -> str:
    """Copies the cached screenshots into a new folder of the run and records it in the run's
    manifest. Returns the folder relative to RENDERS_DIR ("runs/<run_id>/<seq>_v<n>_<trigger>")."""
    run_id = run_id if run_id and RUN_ID_PATTERN.fullmatch(run_id) else ADHOC_RUN
    trigger = trigger if trigger in TRIGGERS else "adhoc"
    run_dir = RENDERS_DIR / "runs" / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="render-", dir=run_dir))
        for name in VIEW_FILES.values():
            shutil.copyfile(RENDERS_DIR / "cache" / key / name, staging / name)
        while True:  # the next free sequence number (another render of the run may have taken it)
            seq = 1 + max((int(m.group(1)) for p in run_dir.iterdir() if (m := FOLDER_PATTERN.fullmatch(p.name))),
                          default=0)
            folder = f"{seq:02d}_v{layout_version}_{trigger}"
            try:
                staging.rename(run_dir / folder)
                break
            except OSError:
                if not (run_dir / folder).exists():
                    raise
        manifest = run_dir / "manifest.json"
        entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []
        entries.append({"seq": seq, "folder": folder, "layoutVersion": layout_version, "trigger": trigger,
                        "cacheKey": key, "cached": cached, "ts": time.time()})
        manifest.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise RenderError("storage_failed", f"Screenshots could not be stored for the run: {exc}")
    return f"runs/{run_id}/{folder}"


# --- browser ---------------------------------------------------------------------------

class _Deadline:
    """One time budget for the whole render. Playwright calls get the remaining time as their own
    timeout instead of being cancelled from outside, which can leave Playwright hanging."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.end = time.monotonic() + seconds

    def remaining_ms(self) -> float:
        remaining = (self.end - time.monotonic()) * 1000
        if remaining <= 0:
            raise self.error()
        return remaining

    def error(self) -> RenderError:
        return RenderError("timeout", f"Rendering did not finish within {self.seconds:g} s.")


async def _launch(playwright, settings: RenderSettings, deadline: _Deadline):
    from playwright.async_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeout

    errors = []
    for channel in settings.browser_channels:
        try:
            return await playwright.chromium.launch(headless=True, channel=None if channel == "chromium" else channel,
                                                    args=BROWSER_ARGS, timeout=deadline.remaining_ms())
        except PlaywrightTimeout:
            raise deadline.error()
        except PlaywrightError as exc:
            errors.append(f"{channel}: {str(exc).splitlines()[0]}")
    raise RenderError("browser_start_failed", "No headless browser could be started ("
                      + "; ".join(errors) + "). Install one with `uv run playwright install chromium`.")


async def _open_renderer(page, settings: RenderSettings, deadline: _Deadline) -> None:
    from playwright.async_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeout

    try:
        await page.goto(settings.page_url, timeout=min(deadline.remaining_ms(), 20_000))
        await page.wait_for_function("window.roomcraftReady === true", timeout=min(deadline.remaining_ms(), 20_000))
    except PlaywrightTimeout as exc:
        if deadline.end <= time.monotonic():
            raise deadline.error()
        raise RenderError("renderer_unavailable", f"Renderer page {settings.page_url} did not load "
                          f"({str(exc).splitlines()[0]}).")
    except PlaywrightError as exc:
        raise RenderError("renderer_unavailable", f"Renderer page {settings.page_url} is not available "
                          f"({str(exc).splitlines()[0]}). Build the frontend (npm run build) "
                          "or point RENDER_PAGE_URL to a running dev server.")


class _RendererSession:
    """Headless browser with the renderer page loaded. Page calls are never cancelled from
    outside (that can leave Playwright hanging); on timeout the browser is closed instead."""

    def __init__(self, settings: RenderSettings, stage: StageFactory):
        self.settings, self.stage = settings, stage
        self.deadline = _Deadline(settings.timeout_s)
        self.pending: list[asyncio.Future] = []

    async def __aenter__(self):
        from playwright.async_api import async_playwright

        self._playwright_cm = async_playwright()
        playwright = await self._playwright_cm.__aenter__()
        try:
            with self.stage("browser", "Start headless browser") as details:
                self.browser = await _launch(playwright, self.settings, self.deadline)
                details.update(browser=self.browser.browser_type.name, version=self.browser.version)
            size = self.settings.image_size
            self.page = await self.browser.new_page(viewport={"width": size, "height": size})
            self.page_errors: list[str] = []
            self.page.on("pageerror", lambda exc: self.page_errors.append(str(exc)))
            with self.stage("load", "Load renderer") as details:
                await _open_renderer(self.page, self.settings, self.deadline)
                details.update(url=self.settings.page_url)
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def evaluate(self, expression: str, argument) -> dict:
        task = asyncio.ensure_future(self.page.evaluate(expression, argument))
        self.pending.append(task)
        done, _ = await asyncio.wait({task}, timeout=self.deadline.remaining_ms() / 1000)
        if not done:
            raise self.deadline.error()
        return task.result()

    async def __aexit__(self, *exc_info):
        if getattr(self, "browser", None):
            await self.browser.close()
        if self.pending:  # collect errors of evaluations aborted by closing the browser
            await asyncio.gather(*self.pending, return_exceptions=True)
        await self._playwright_cm.__aexit__(None, None, None)


def _subprocess_capable(func):
    """Playwright starts the browser as a subprocess. On Windows the SelectorEventLoop (which
    `uvicorn --reload` uses) cannot do that and raises NotImplementedError. There the browser work
    runs in a worker thread with its own ProactorEventLoop; asyncio.to_thread copies the context,
    so progress events (LangGraph's thread-safe stream writer) still arrive."""
    async def wrapper(*args, **kwargs):
        if sys.platform != "win32" or isinstance(asyncio.get_running_loop(), asyncio.ProactorEventLoop):
            return await func(*args, **kwargs)
        return await asyncio.to_thread(
            lambda: asyncio.run(func(*args, **kwargs), loop_factory=asyncio.ProactorEventLoop))
    return wrapper


@_subprocess_capable
async def _render_in_browser(config: dict, settings: RenderSettings, stage: StageFactory) -> dict:
    async with _RendererSession(settings, stage) as session:
        with stage("capture", "Render and capture scene") as details:
            result = await session.evaluate("config => window.roomcraftRender(config)", config)
            if result.get("error"):
                failures = "; ".join(f"{f['id']}: {f['error']}" for f in result.get("failures", []))
                raise RenderError(result.get("code", "capture_failed"),
                                  result["error"] + (f" ({failures})" if failures else ""))
            details.update(furniture=len(config["furniture"]), stats=result.get("stats"),
                           pageErrors=session.page_errors[:3])
        return result


@_subprocess_capable
async def measure_model(model_url: str, settings: RenderSettings = DEFAULT_SETTINGS,
                        stage: StageFactory | None = None) -> dict[str, float]:
    """Bounding box of a model (in its own units) as three.js loads it in the app:
    {"width": x, "height": y, "depth": z}."""
    stage = stage or (lambda name, label: nullcontext({}))
    async with _RendererSession(settings, stage) as session:
        with stage("measure", "Measure model") as details:
            result = await session.evaluate("url => window.roomcraftMeasure(url)", model_url)
            if result.get("error"):
                raise RenderError("asset_load_failed", f"Model could not be loaded for measuring: {result['error']}")
            size = result["size"]
            if not all(_finite(size.get(k)) and size[k] >= 0 for k in ("width", "height", "depth")) \
                    or max(size.values()) <= 0:
                raise RenderError("invalid_model", "The model has no measurable geometry.")
            details.update(size=size)
        return size


# --- entry point -------------------------------------------------------------------------

async def render_room(room: Room, furniture: list[Furniture], settings: RenderSettings = DEFAULT_SETTINGS,
                      stage: StageFactory | None = None, *, run_id: str | None = None, layout_version: int = 0,
                      trigger: str = "adhoc") -> RenderResult:
    """Renders `room` with `furniture` (or reuses cached images of the identical room) into a new
    folder of the run `run_id` (runs/_adhoc without one)."""
    stage = stage or (lambda name, label: nullcontext({}))
    started = time.perf_counter()
    elapsed = lambda: round((time.perf_counter() - started) * 1000)

    with stage("prepare", "Check room configuration") as details:
        config = build_render_config(room, furniture, settings)
        key = cache_key(config)
        details.update(furniture=len(config["furniture"]), key=key[:12], imageSize=settings.image_size)

    with stage("cache", "Check cache") as details:
        cached = _cached_images(key, settings.image_size) is not None
        details.update(hit=cached)
    if not cached:
        result = await _render_in_browser(config, settings, stage)

    with stage("store", "Store screenshots") as details:
        if not cached:
            _store(key, result, settings.image_size)
        folder = _publish(key, run_id, layout_version, trigger, cached)
        images = _refs(folder)
        details.update(folder=folder, images=list(images.values()))
    return RenderResult(key=key, images=images, cached=cached, duration_ms=elapsed(), folder=f"renders/{folder}")
