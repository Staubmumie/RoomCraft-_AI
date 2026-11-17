# Furniture database, enrichment and search

## Storage

| What | Where | Notes |
|---|---|---|
| 3D models, preview images | `src/data/inventory/<item id>/` | unchanged file storage, served by `/api/inventory/<id>/files/...` |
| Inventory registry | `src/data/inventory.json` | unchanged; used by the frontend, the renderer and the designer |
| Searchable furniture records | `src/data/furniture_db/` (LanceDB, table `furniture`) | references to the files, metadata, description, embedding |

A record's id is the inventory item id (derived from the source asset id), so an asset can never be
stored twice; writes are upserts (`merge_insert` on `id`).

### Record schema (`src/furniture/store.py`)

| Field | Content |
|---|---|
| `id`, `asset_id` | inventory item id, source id (`provider:id`) |
| `name`, `category` | category from a fixed list (chair, armchair, sofa, table, coffee_table, desk, bed, cabinet, wardrobe, shelf, sideboard, lamp, rug, decor, plant, storage, other) |
| `description` | short, retrieval-oriented text (AI or fallback) |
| `style`, `colors`, `materials`, `features` | from image analysis or metadata; `observed` lists which of them come from the image (unverified) |
| `tags` | source tags |
| `width`, `height`, `depth` | metres, measured from the model (`unitScale` applied); null if unknown |
| `price`, `currency` | only if the source states them (free assets: 0) |
| `provider`, `author`, `license`, `source_url` | verified source metadata |
| `model_url`, `model_path`, `format`, `thumbnail_url` | file references |
| `enrichment` | `vision` (image + metadata), `metadata` (no usable image) or `fallback` (LLM failed) |
| `embedding_model`, `text`, `vector` | what was embedded, with which model |
| `source_hash`, `updated_at` | inputs of the enrichment; unchanged inputs are not regenerated |

## Shared asset pipeline (`src/furniture/pipeline.py`)

```
process_asset(asset_id, extra_metadata)
  1. inventory.acquire_asset   download, verify, register files (existing assets are reused)
  2. rendering.measure_model   real bounding box -> dimensions, unit scale
  3. generate_furniture_description   vision LLM with structured output (furniture/enrichment.py)
  4. FurnitureStore.upsert     embed (OpenRouter, openai/text-embedding-3-small) and save
```

Used by:
- **Frontend:** `POST /api/inventory` returns the inventory item plus
  `processing: {furnitureId, created, measured, enrichment, stored, warnings}`.
- **LangGraph:** the `asset_download` node calls `process_asset` for every approved asset (no extra graph nodes).

Failures degrade instead of producing broken records: a failed download registers nothing; a failed
measurement leaves the size to the app; a failed LLM call stores a metadata-only description; a
database error keeps the downloaded asset and reports `stored: false`. `POST /api/furniture/reindex`
indexes missing or changed inventory items and removes records whose files are gone; deleting an
inventory item also deletes its record.

## Retrieval

`GET /api/furniture/search?q=...&category=...&max_width=...&max_depth=...` and the research tool
`search_local_furniture` use the same `FurnitureStore.search`:

1. Hard constraints as a LanceDB pre-filter (category, maximum width/depth/height; unknown sizes pass).
2. Cosine similarity between the query embedding and the record embeddings (name, category, style,
   colours, materials, features, description, tags).
3. Results below a minimum score (0.25 for the research agent) are dropped; the agent sees the scores.

For direct user searches Furniture Research is local-first, per item of the furniture list: it searches
the database with the item's description, then an LLM checks the best matches against the item's
essential requirements (`choose_local_item`, may refine once with filters); a match is only taken when
every listed requirement is met, a category match alone is not enough. Only items without a local match
are searched online (MCP), if the online search is enabled. Local finds need no approval, no download
and no online budget; external assets go through Human Approval (or the automatic choice) -> Asset
Download. Requests of the interior designer skip the local check (it already sees the inventory).
External results that are already in the inventory are treated as local.

Embedding model: `EMBEDDING_MODEL` (default `openai/text-embedding-3-small`, 1536 dimensions). Rows written
with another model are re-embedded by `FurnitureStore.ensure_model()` (called by reindex); a different vector
size needs a rebuild (delete `src/data/furniture_db` and reindex).
