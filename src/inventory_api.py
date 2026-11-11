"""HTTP routes for the furniture inventory. Downloads go through the shared furniture pipeline,
the same one the research agent uses."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import inventory
from furniture.pipeline import process_asset, remove

router = APIRouter(prefix="/api/inventory", tags=["inventory"])
MEDIA_TYPES = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json"}


class DownloadRequest(BaseModel):
    assetId: str


class Dimensions(BaseModel):
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    depth: float = Field(gt=0)


@router.get("")
async def list_inventory():
    return inventory.list_items()


@router.post("", status_code=201)
async def add_to_inventory(request: DownloadRequest):
    """Download, measure, describe and index an asset. `processing` tells which steps succeeded."""
    if any(item["assetId"] == request.assetId for item in inventory.load_index().values()):
        raise HTTPException(409, "This asset is already in the inventory.")
    result = await process_asset(request.assetId)
    return {**result.item, "processing": result.as_dict()}


@router.delete("/{item_id}")
async def delete_from_inventory(item_id: str):
    await inventory.delete_item(item_id)
    try:
        await remove(item_id)
    except Exception:
        pass  # a stale search record is removed by the next reindex; the files are gone either way
    return {"deleted": item_id}


@router.put("/{item_id}/dimensions")
async def set_dimensions(item_id: str, dimensions: Dimensions):
    return await inventory.update_item(item_id, dimensions=dimensions.model_dump())


@router.get("/{item_id}/files/{file_path:path}")
async def get_file(item_id: str, file_path: str):
    path = inventory.item_file(item_id, file_path)
    if path is None:
        raise HTTPException(404, "File not found.")
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower()))
