from fastapi import APIRouter, HTTPException, Query

from furniture.pipeline import reindex
from furniture.store import CATEGORIES, compact, get_store
from llm import LLMError

router = APIRouter(prefix="/api/furniture", tags=["furniture"])


@router.get("/search")
async def search_furniture(q: str = Query(min_length=1, max_length=300), category: str | None = None,
                           max_width: float | None = None, max_depth: float | None = None, limit: int = Query(8, le=30)):
    """Semantic search over the furniture database with optional metadata filters."""
    if category and category not in CATEGORIES:
        raise HTTPException(422, f"Unknown category; use one of: {', '.join(CATEGORIES)}")
    try:
        hits = await get_store().search(q, category=category, max_width=max_width, max_depth=max_depth, limit=limit)
    except LLMError as exc:
        raise HTTPException(502, str(exc))
    return {"query": q, "results": [compact(h) for h in hits]}


@router.get("/{item_id}")
async def get_furniture(item_id: str):
    record = await get_store().get(item_id)
    if not record:
        raise HTTPException(404, "Not in the furniture database.")
    record.pop("vector", None)
    return record


@router.post("/reindex")
async def reindex_furniture(force: bool = False):
    """Indexes inventory items that are missing or changed (e.g. downloaded before the database existed)."""
    return await reindex(force=force)
