from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from rendering.service import run_image_path

router = APIRouter(prefix="/api/renders", tags=["renders"])


@router.get("/runs/{run_id}/{folder}/{name}")
async def get_render(run_id: str, folder: str, name: str):
    """Serves a screenshot of a workflow run, e.g. for the execution inspector. Every path segment
    is checked against its strict format, so nothing outside the run folders can be reached."""
    path = run_image_path(run_id, folder, name)
    if path is None or not path.is_file():
        raise HTTPException(404, "Screenshot not found.")
    return FileResponse(path, media_type="image/png")
