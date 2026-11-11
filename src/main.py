"""RoomCraft API: start with `uv run uvicorn main:app --reload --port 8000` from src/."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from asset_api import router as asset_router
from design_api import router as design_router
from furniture.api import router as furniture_router
from inventory_api import router as inventory_router
from rendering.api import router as render_router

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app = FastAPI(title="RoomCraft API")
app.include_router(asset_router)
app.include_router(inventory_router)
app.include_router(design_router)
app.include_router(render_router)
app.include_router(furniture_router)
# Built frontend incl. the headless renderer page (renderer/render.html) used by render_scene.
app.mount("/renderer", StaticFiles(directory=FRONTEND_DIST, html=True, check_dir=False), name="renderer")
