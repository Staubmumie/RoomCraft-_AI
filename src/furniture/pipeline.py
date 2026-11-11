"""Shared asset pipeline, used by the frontend API (POST /api/inventory) and by the LangGraph
asset_download node:

    Asset Download -> Measure -> Add Description -> Save Furniture (LanceDB)

1. acquire_asset (inventory): download, verify and register the files; existing assets are reused.
2. measure_model (renderer): real bounding box -> dimensions and unit scale.
3. generate_furniture_description: structured description from metadata + preview image.
4. FurnitureStore.upsert: embed and save the searchable record (id = inventory item id).

The asset is usable as soon as step 1 succeeded. Later steps degrade instead of failing: without a
size the app measures on first placement, without an LLM a metadata-only description is stored, and
an item that could not be indexed is reported (stored=False) and picked up again by reindex().
"""
import hashlib
import json
from dataclasses import dataclass, field
from urllib.parse import unquote

import inventory
from furniture.enrichment import generate_furniture_description, image_data_url
from furniture.store import get_store
from rendering.service import RenderError, measure_model

ENRICHMENT_VERSION = 1  # bump to regenerate all descriptions


def unit_scale(size: dict[str, float]) -> float:
    """Scale from model units to metres, judged from the largest side of a piece of furniture."""
    largest = max(size["width"], size["height"], size["depth"])
    if largest > 500:
        return 0.001  # millimetres
    if largest > 30:
        return 0.01   # centimetres
    return 1.0


@dataclass
class ProcessResult:
    item: dict
    created: bool              # downloaded now (False: it was already in the inventory)
    measured: bool = False
    enrichment: str | None = None  # vision | metadata | fallback | unchanged
    stored: bool = False       # searchable in the furniture database
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"furnitureId": self.item["id"], "created": self.created, "measured": self.measured,
                "enrichment": self.enrichment, "stored": self.stored, "warnings": self.warnings}


def item_metadata(item: dict, extra: dict | None = None) -> dict:
    """Verified facts about an inventory item (plus extra source metadata, e.g. from MCP)."""
    extra = extra or {}
    scale = item.get("unitScale") or 1.0
    dims = item.get("dimensions")
    license_ = item.get("license") or extra.get("license")
    return {
        "name": item.get("name"),
        "provider": item.get("provider"),
        "author": item.get("author"),
        "license": license_.get("name") if isinstance(license_, dict) else license_,
        "description": item.get("description") or extra.get("description"),
        "tags": item.get("tags") or extra.get("tags") or [],
        "price": item.get("price"),
        "currency": item.get("currency"),
        "formats": [item.get("format")],
        "dimensions_m": {k: round(v * scale, 3) for k, v in dims.items()} if dims else None,
    }


def _preview_path(item: dict):
    url = item.get("thumbnailUrl") or ""
    if "/files/" not in url:
        return None
    return inventory.ASSETS_DIR / item["id"] / unquote(url.split("/files/", 1)[1])


def _source_hash(metadata: dict, item: dict) -> str:
    payload = json.dumps({"v": ENRICHMENT_VERSION, "meta": metadata, "thumb": item.get("thumbnailUrl")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


async def index_item(item: dict, extra: dict | None = None, force: bool = False) -> tuple[str, list[str]]:
    """Add Description + Save Furniture for one inventory item. Returns (enrichment source, warnings).
    Skips the LLM and the embedding when the inputs did not change since the last run."""
    store = get_store()
    metadata = item_metadata(item, extra)
    source_hash = _source_hash(metadata, item)
    existing = await store.get(item["id"])
    if existing and existing["source_hash"] == source_hash and not force:
        return "unchanged", []

    enrichment = await generate_furniture_description(metadata, image_data_url(_preview_path(item)))
    dims = metadata["dimensions_m"] or {}
    record = {
        "id": item["id"], "asset_id": item["assetId"], "name": item["name"],
        "category": enrichment.category, "description": enrichment.description, "style": enrichment.style,
        "colors": enrichment.colors, "materials": enrichment.materials, "features": enrichment.features,
        "tags": metadata["tags"][:20], "width": dims.get("width"), "height": dims.get("height"), "depth": dims.get("depth"),
        "price": metadata["price"], "currency": metadata["currency"], "provider": metadata["provider"],
        "author": metadata["author"], "license": metadata["license"], "source_url": item.get("sourceUrl"),
        "model_url": item["modelUrl"], "model_path": item["modelPath"], "format": item["format"],
        "thumbnail_url": item.get("thumbnailUrl"), "enrichment": enrichment.source, "observed": enrichment.observed,
        "source_hash": source_hash,
    }
    await store.upsert(record)
    return enrichment.source, [enrichment.warning] if enrichment.warning else []


async def _measure(item: dict, result: ProcessResult) -> dict:
    try:
        size = await measure_model(item["modelUrl"])
    except RenderError as exc:
        result.warnings.append(f"Size could not be determined ({exc.code}); the app measures it on first placement.")
        return item
    result.measured = True
    return await inventory.update_item(item["id"], dimensions=size, unitScale=unit_scale(size))


async def process_asset(asset_id: str, extra: dict | None = None) -> ProcessResult:
    """The whole pipeline for one asset. Raises HTTPException only if the download itself failed
    (then nothing was registered)."""
    item, created = await inventory.acquire_asset(asset_id)
    result = ProcessResult(item=item, created=created)
    if created or not item.get("dimensions"):
        item = result.item = await _measure(item, result)
    try:
        result.enrichment, warnings = await index_item(item, extra)
        result.warnings += warnings
        result.stored = True
    except Exception as exc:  # database or embedding problems must not lose the downloaded asset
        result.warnings.append(f"Not added to the furniture search ({type(exc).__name__}: {exc}); run a reindex later.")
    return result


async def reindex(force: bool = False) -> dict:
    """Brings the furniture database in line with the inventory: indexes missing or changed items
    and removes records whose files are gone."""
    store = get_store()
    await store.ensure_model()
    index = inventory.load_index()
    indexed, unchanged, failed = [], [], []
    for item in index.values():
        if not (inventory.ASSETS_DIR / item["id"] / item["modelPath"]).is_file():
            continue
        try:
            source, _ = await index_item(item, force=force)
            (unchanged if source == "unchanged" else indexed).append(item["id"])
        except Exception as exc:
            failed.append({"id": item["id"], "error": f"{type(exc).__name__}: {exc}"})
    stale = sorted(await store.ids() - {i for i, item in index.items()
                                        if (inventory.ASSETS_DIR / i / item["modelPath"]).is_file()})
    for record_id in stale:
        await store.delete(record_id)
    return {"indexed": indexed, "unchanged": unchanged, "failed": failed, "removed": stale}


async def remove(item_id: str) -> None:
    await get_store().delete(item_id)
