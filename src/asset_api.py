from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/assets", tags=["assets"])

ASSET_API = "https://3d.shep.bot"
LOADABLE_FORMATS = {"glb", "gltf"}


def raise_upstream_error(exc: httpx.HTTPError, action: str):
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        try:
            message = exc.response.json().get("error", "")
        except ValueError:
            message = ""
        if status == 429:
            raise HTTPException(429, f"Asset server rate limit reached. {message}".strip())
        if status == 404:
            raise HTTPException(404, message or "Asset not found.")
        raise HTTPException(502, f"{action} failed ({status}). {message}".strip())
    raise HTTPException(502, f"{action} failed: 3D asset server unavailable ({exc}).")


async def fetch_json(path: str, action: str, params: dict | None = None, timeout: float = 30):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{ASSET_API}{path}", params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise_upstream_error(exc, action)


def asset_path(asset_id: str) -> str:
    return f"/v1/assets/{quote(asset_id, safe=':')}"


def summarize(asset: dict) -> dict:
    formats = [f.lower() for f in asset.get("formats") or []]
    downloadable = bool(asset.get("downloadable"))
    return {
        "id": asset["id"],
        "name": asset.get("title") or asset["id"],
        "provider": asset.get("provider"),
        "author": asset.get("author"),
        "thumbnailUrl": asset.get("thumbnailUrl"),
        "sourceUrl": asset.get("url"),
        "license": asset.get("license"),
        "free": (asset.get("price") or {}).get("free"),
        "formats": formats,
        "downloadable": downloadable,
        # Assets without format info are allowed; the download step verifies the actual files.
        "loadable": downloadable and (not formats or bool(LOADABLE_FORMATS & set(formats))),
    }


async def select_model_file(asset_id: str) -> tuple[dict | None, dict | None]:
    """Returns the first GLB/glTF file (with its companion files) that needs no login."""
    for fmt in ("glb", "gltf"):
        selection = await fetch_json(
            f"{asset_path(asset_id)}/files", "File selection",
            params={"format": fmt, "resolution": "1k"},
        )
        for file in selection.get("files", []):
            name = file.get("filename", "").lower()
            is_model = file.get("format", "").lower() in LOADABLE_FORMATS or name.endswith((".glb", ".gltf"))
            if is_model and not file.get("requiresAuth"):
                return file, selection.get("license")
    return None, None


@router.get("/health")
async def check_connection():
    await fetch_json("/health", "Health check", timeout=10)
    return {"connected": True}


@router.get("/search")
async def search_assets(query: str):
    data = await fetch_json(
        "/v1/search", "Asset search",
        params={"q": query, "type": "model", "free": "true", "downloadable": "true", "limit": 24},
    )
    results = [summarize(asset) for asset in data.get("results", [])]
    # Known GLB/glTF assets first, keep the server's ranking otherwise.
    results.sort(key=lambda a: (not a["loadable"], not a["formats"]))
    return {"query": query, "results": results}


@router.get("/details")
async def asset_details(id: str):
    asset = await fetch_json(asset_path(id), "Asset lookup")
    return {
        **summarize(asset),
        "description": asset.get("description"),
        "tags": asset.get("tags", []),
        "polyCount": asset.get("polyCount"),
        "animated": asset.get("animated"),
        "files": [
            {
                "filename": f.get("filename"),
                "format": f.get("format"),
                "sizeBytes": f.get("sizeBytes"),
                "requiresAuth": bool(f.get("requiresAuth")),
            }
            for f in asset.get("files", [])
        ],
    }
