"""LanceDB index of the furniture inventory: searchable metadata, AI descriptions and embeddings.

The files themselves (3D models, previews) stay in the inventory's file storage
(data/inventory/<id>/, registered in inventory.json); a record only holds references to them.
Record id = inventory item id, so the same asset is never stored twice (upsert).
"""
import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector

from furniture.embeddings import OpenRouterEmbedder

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "furniture_db"
TABLE = "furniture"
CATEGORIES = ("chair", "armchair", "sofa", "table", "coffee_table", "desk", "bed", "cabinet", "wardrobe", "shelf",
              "sideboard", "lamp", "rug", "decor", "plant", "storage", "other")


def record_model(dim: int) -> type[LanceModel]:
    class FurnitureRecord(LanceModel):
        id: str                                  # inventory item id
        asset_id: str                            # source id, e.g. "polyhaven:ArmChair_01"
        name: str
        category: str                            # one of CATEGORIES
        description: str                         # short, retrieval-oriented
        style: str | None = None
        colors: list[str] = []
        materials: list[str] = []
        features: list[str] = []                 # shape and design characteristics
        tags: list[str] = []
        width: float | None = None               # metres (model size x unit scale)
        height: float | None = None
        depth: float | None = None
        price: float | None = None
        currency: str | None = None
        provider: str | None = None
        author: str | None = None
        license: str | None = None
        source_url: str | None = None
        model_url: str                           # file references, served by the inventory API
        model_path: str
        format: str
        thumbnail_url: str | None = None
        enrichment: str                          # vision | metadata | fallback
        observed: list[str] = []                 # attributes that come from image analysis (unverified)
        embedding_model: str
        source_hash: str                         # inputs of the enrichment; unchanged -> nothing is regenerated
        text: str                                # what was embedded
        updated_at: str
        vector: Vector(dim)
    return FurnitureRecord


def embedding_text(record: dict) -> str:
    parts = [record["name"], record["category"].replace("_", " ")]
    if record.get("style"):
        parts.append(f"style: {record['style']}")
    for key in ("colors", "materials", "features"):
        if record.get(key):
            parts.append(f"{key}: {', '.join(record[key])}")
    parts.append(record["description"])
    if record.get("tags"):
        parts.append(f"tags: {', '.join(record['tags'][:10])}")
    return ". ".join(parts)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class FurnitureStore:
    """Async wrapper around one LanceDB table (LanceDB calls run in a worker thread)."""

    def __init__(self, path: Path = DB_DIR, embedder=None):
        self.path = Path(path)
        self.embedder = embedder or OpenRouterEmbedder()
        self.model = record_model(self.embedder.dim)
        self._table = None
        self._lock = asyncio.Lock()

    def _open(self):
        if self._table is None:
            self.path.mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(self.path)
            self._table = db.create_table(TABLE, schema=self.model, exist_ok=True)
        return self._table

    async def _run(self, fn):
        return await asyncio.to_thread(fn)

    async def ensure_model(self) -> int:
        """Re-embeds rows written with another embedding model (vectors must match the query model)."""
        def stale_rows():
            table = self._open()
            if table.schema.field("vector").type.list_size != self.embedder.dim:
                raise RuntimeError("The furniture index uses another vector size; delete data/furniture_db and reindex.")
            return table.search().where(f"embedding_model != {_sql_string(self.embedder.model)}").limit(10_000).to_list()
        rows = await self._run(stale_rows)
        if rows:
            vectors = await self.embedder.embed([row["text"] for row in rows])
            for row, vector in zip(rows, vectors):
                row.update(vector=vector, embedding_model=self.embedder.model)
                row.pop("_distance", None)
            await self._upsert_rows(rows)
        return len(rows)

    async def _upsert_rows(self, rows: list[dict]):
        def write():
            self._open().merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
                [self.model(**row).model_dump() for row in rows])
        async with self._lock:
            await self._run(write)

    async def upsert(self, record: dict) -> dict:
        """Creates or replaces the record with this id; embeds its text."""
        record = {**record, "text": embedding_text(record), "embedding_model": self.embedder.model,
                  "updated_at": datetime.now(timezone.utc).isoformat()}
        record["vector"] = (await self.embedder.embed([record["text"]]))[0]
        await self._upsert_rows([record])
        return record

    async def get(self, record_id: str) -> dict | None:
        def read():
            rows = self._open().search().where(f"id = {_sql_string(record_id)}").limit(1).to_list()
            return rows[0] if rows else None
        return await self._run(read)

    async def ids(self) -> set[str]:
        return await self._run(lambda: set(self._open().to_arrow().column("id").to_pylist()))

    async def categories(self) -> dict[str, str]:
        """Record id (= inventory item id) -> category, for all records."""
        def read():
            columns = self._open().to_arrow().select(["id", "category"]).to_pydict()
            return dict(zip(columns["id"], columns["category"]))
        return await self._run(read)

    async def count(self) -> int:
        return await self._run(lambda: self._open().count_rows())

    async def delete(self, record_id: str) -> None:
        async with self._lock:
            await self._run(lambda: self._open().delete(f"id = {_sql_string(record_id)}"))

    async def search(self, query: str, category: str | None = None, max_width: float | None = None,
                     max_depth: float | None = None, max_height: float | None = None, exclude: list[str] | None = None,
                     limit: int = 8, min_score: float = 0.0) -> list[dict]:
        """Metadata pre-filter + vector similarity. score = cosine similarity (1 is identical)."""
        filters = []
        if category:
            filters.append(f"category = {_sql_string(category)}")
        for column, value in (("width", max_width), ("depth", max_depth), ("height", max_height)):
            if value is not None:
                filters.append(f"({column} IS NULL OR {column} <= {float(value)})")
        if exclude:
            filters.append("id NOT IN (" + ", ".join(_sql_string(i) for i in exclude) + ")")
        vector = (await self.embedder.embed([query]))[0]

        def run():
            search = self._open().search(vector).metric("cosine").limit(limit)
            if filters:
                search = search.where(" AND ".join(filters), prefilter=True)
            return search.to_list()
        rows = await self._run(run)
        hits = []
        for row in rows:
            score = 1 - row.pop("_distance")
            if score >= min_score:
                row.pop("vector", None)
                hits.append({**row, "score": round(score, 3)})
        return hits


_store: FurnitureStore | None = None


def get_store() -> FurnitureStore:
    """The application's store (tests replace it with set_store)."""
    global _store
    if _store is None:
        _store = FurnitureStore()
    return _store


def set_store(store: FurnitureStore | None) -> None:
    global _store
    _store = store


def compact(row: dict) -> dict:
    """Search hit as the agents and the API see it (no vector, no internal fields)."""
    keys = ("id", "name", "category", "description", "style", "colors", "materials", "features", "width", "depth",
            "height", "price", "currency", "provider", "license", "thumbnail_url", "model_url", "enrichment",
            "observed", "score")
    return {k: row.get(k) for k in keys if row.get(k) not in (None, [], "")}


def category_of(value: str | None) -> str | None:
    value = re.sub(r"[^a-z_]", "", (value or "").lower().replace(" ", "_"))
    return value if value in CATEGORIES else None
