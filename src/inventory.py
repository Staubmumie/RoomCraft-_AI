"""Furniture inventory: downloaded 3D models in data/inventory/<item id>/, registered in
data/inventory.json. Downloads are verified before they are registered, so the index only ever
contains complete, loadable assets. HTTP routes live in inventory_api.py; the furniture pipeline
(furniture/pipeline.py) builds on acquire_asset.
"""
import asyncio
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse

import httpx
from fastapi import HTTPException

from asset_api import asset_path, fetch_json, select_model_file
from url_safety import public_http_client

DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = DATA_DIR / "inventory"
INDEX_FILE = DATA_DIR / "inventory.json"
MAX_FILE_BYTES = 200 * 1024 * 1024
THUMBNAIL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

_index_lock = asyncio.Lock()
_downloading: set[str] = set()


# --- storage -------------------------------------------------------------

def load_index() -> dict:
    if not INDEX_FILE.exists():
        return {}
    return json.loads(INDEX_FILE.read_text("utf-8"))


def save_index(index: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(INDEX_FILE)


def item_id_for(asset_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", asset_id.lower()).strip("-")[:50]
    return f"{slug}-{hashlib.sha1(asset_id.encode()).hexdigest()[:8]}"


def file_url(item_id: str, relative_path: str) -> str:
    return f"/api/inventory/{item_id}/files/{quote(relative_path)}"


def with_status(item: dict) -> dict:
    model = ASSETS_DIR / item["id"] / item["modelPath"]
    return {**item, "missing": not model.is_file()}


def safe_relative(path: str) -> str:
    """Rejects absolute paths and '..' so downloads stay inside the asset folder."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if not parts or parts[0] == "/" or any(p in ("..", "") for p in parts) or ":" in parts[0]:
        raise ValueError(f"Unsafe file path in asset: {path!r}")
    return "/".join(parts)


# --- download & verification ---------------------------------------------

async def download_file(client: httpx.AsyncClient, url: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with target.open("wb") as out:
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise ValueError(f"{target.name} is larger than {MAX_FILE_BYTES // 2**20} MB.")
                out.write(chunk)
    if size == 0:
        raise ValueError(f"{target.name} was downloaded empty.")
    return size


async def download_thumbnail(client: httpx.AsyncClient, url: str | None, folder: Path) -> str | None:
    if not url:
        return None
    suffix = Path(urlparse(url).path).suffix.lower()
    name = "thumbnail" + (suffix if suffix in THUMBNAIL_SUFFIXES else ".jpg")
    try:
        await download_file(client, url, folder / name)
        return name
    except (httpx.HTTPError, ValueError):
        return None  # a thumbnail is nice to have, never a reason to fail


def verify_model(folder: Path, model_path: str) -> str:
    """Checks the file really is GLB/glTF and that every referenced file exists. Returns the format."""
    model = folder / model_path
    with model.open("rb") as f:
        header = f.read(4)
    if header == b"glTF":
        return "glb"
    try:
        gltf = json.loads(model.read_text("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("The downloaded file is neither GLB nor glTF.")
    if not isinstance(gltf, dict) or "asset" not in gltf:
        raise ValueError("The downloaded file is not a valid glTF document.")
    root = folder.resolve()
    for entry in gltf.get("buffers", []) + gltf.get("images", []):
        uri = entry.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        dependency = (model.parent / unquote(uri)).resolve()
        if not dependency.is_relative_to(root) or not dependency.is_file():
            raise ValueError(f"The glTF references a missing file: {uri}")
    return "gltf"


async def download_asset(asset_id: str) -> dict:
    details = await fetch_json(asset_path(asset_id), "Asset lookup")
    if not details.get("downloadable"):
        raise HTTPException(422, "This asset cannot be downloaded directly. Use its source page instead.")
    model_file, license_ = await select_model_file(asset_id)
    if not model_file:
        raise HTTPException(422, "No freely downloadable GLB/glTF file is available for this asset.")

    item_id = item_id_for(asset_id)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="download-", dir=DATA_DIR))
    try:
        model_path = safe_relative(model_file["filename"])
        # Every request (and redirect hop) must go to a public host; UnsafeURLError is a ValueError -> 422.
        async with public_http_client(timeout=120) as client:
            total = await download_file(client, model_file["url"], staging / model_path)
            for include in model_file.get("includes") or []:
                path = safe_relative(include["path"])
                if path != model_path:
                    total += await download_file(client, include["url"], staging / path)
            thumbnail = await download_thumbnail(client, details.get("thumbnailUrl"), staging)
        fmt = verify_model(staging, model_path)

        target = ASSETS_DIR / item_id
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)  # leftover from an earlier, unregistered attempt
        staging.rename(target)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Download failed: {exc}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return {
        "id": item_id,
        "assetId": asset_id,
        "name": details.get("title") or asset_id,
        "provider": details.get("provider"),
        "author": details.get("author"),
        "sourceUrl": details.get("url"),
        "license": license_ or details.get("license"),
        "format": fmt,
        "modelPath": model_path,
        "modelUrl": file_url(item_id, model_path),
        "thumbnailUrl": file_url(item_id, thumbnail) if thumbnail else None,
        # Source metadata for the furniture search (furniture.pipeline); only what the source states.
        "description": (details.get("description") or "")[:1000] or None,
        "tags": (details.get("tags") or [])[:20],
        "price": 0.0 if (details.get("price") or {}).get("free") else (details.get("price") or {}).get("amount"),
        "currency": (details.get("price") or {}).get("currency"),
        "dimensions": None,
        "unitScale": 1.0,  # scale that turns model units into metres (see furniture.pipeline)
        "sizeBytes": total,
        "downloadedAt": datetime.now(timezone.utc).isoformat(),
    }


# --- service (used by the routes and by the asset_download graph node) ----

async def acquire_asset(asset_id: str) -> tuple[dict, bool]:
    """Downloads, verifies and registers an asset. Returns (item, created); an asset that is
    already in the inventory is returned as is (created=False) and not downloaded again.
    Raises HTTPException with a readable detail on failure; the inventory stays unchanged then."""
    async with _index_lock:
        existing = next((item for item in load_index().values() if item["assetId"] == asset_id), None)
        if existing:
            return with_status(existing), False
        if asset_id in _downloading:
            raise HTTPException(409, "This asset is already being downloaded.")
        _downloading.add(asset_id)
    try:
        item = await download_asset(asset_id)
        async with _index_lock:
            index = load_index()
            index[item["id"]] = item
            save_index(index)
        return with_status(item), True
    finally:
        _downloading.discard(asset_id)


async def update_item(item_id: str, **fields) -> dict:
    async with _index_lock:
        index = load_index()
        if item_id not in index:
            raise HTTPException(404, "Inventory item not found.")
        index[item_id].update(fields)
        save_index(index)
    return with_status(index[item_id])


async def delete_item(item_id: str) -> None:
    """Removes an item from the index and deletes its files."""
    async with _index_lock:
        index = load_index()
        if item_id not in index:
            raise HTTPException(404, "Inventory item not found.")
        del index[item_id]
        save_index(index)
    shutil.rmtree(ASSETS_DIR / item_id, ignore_errors=True)


def list_items() -> list[dict]:
    """All items, newest first, each with a `missing` flag for lost model files."""
    items = sorted(load_index().values(), key=lambda i: i["downloadedAt"], reverse=True)
    return [with_status(item) for item in items]


def item_file(item_id: str, relative_path: str) -> Path | None:
    """Path of one of an item's files, or None if the item or file does not exist or the path
    would leave the item's folder."""
    if item_id not in load_index():
        return None
    folder = (ASSETS_DIR / item_id).resolve()
    target = (folder / relative_path).resolve()
    return target if target.is_relative_to(folder) and target.is_file() else None
