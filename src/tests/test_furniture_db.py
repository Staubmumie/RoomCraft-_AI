"""Furniture database (LanceDB), enrichment and the shared asset pipeline.

Real LanceDB in a temp dir (see the furniture_store fixture) with a deterministic hash embedder;
the download, the renderer and the LLM are faked."""
import asyncio

import pytest

import design.asset_download as download_module
import furniture.enrichment as enrichment_module
import furniture.pipeline as pipeline
import inventory
import inventory_api
from design.human_approval import validate_decision
from design.runner import execute_design_run, resume_design_run
from design.state import Room
from furniture.enrichment import FurnitureEnrichment, generate_furniture_description
from furniture.store import FurnitureStore, set_store
from llm import CallInfo, LLMError, StructuredResult
from observability.store import RunStore
from tests.conftest import HashEmbedder, png_bytes
from tests.research_fakes import FakeMCP, call, install_research

SOURCES = {
    "polyhaven:oak_chair": {"title": "Oak Dining Chair", "description": "A minimalist wooden chair with a slim backrest.",
                            "tags": ["chair", "wood", "minimalist"], "thumbnail": True},
    "grab3d:leather_sofa": {"title": "Chesterfield Sofa", "description": "Classic three-seat leather sofa.",
                            "tags": ["sofa", "leather"], "thumbnail": True},
    "kenney:reading_armchair": {"title": "Reading Armchair", "description": "Comfortable armchair for reading.",
                                "tags": ["armchair", "reading", "comfortable"], "thumbnail": False},
}
ENRICHED = {
    "Oak Dining Chair": ("chair", "scandinavian", ["light oak"], ["wood"], "Minimalist light oak wooden chair, slim backrest."),
    "Chesterfield Sofa": ("sofa", "classic", ["brown"], ["leather"], "Classic brown leather chesterfield sofa with buttons."),
    "Reading Armchair": ("armchair", "cozy", ["green"], ["fabric"], "Comfortable upholstered armchair for reading."),
}


@pytest.fixture
def assets(tmp_path, monkeypatch):
    """Temporary inventory; download_asset, the renderer and the enrichment LLM are faked."""
    monkeypatch.setattr(inventory, "DATA_DIR", tmp_path)
    monkeypatch.setattr(inventory, "ASSETS_DIR", tmp_path / "inventory")
    monkeypatch.setattr(inventory, "INDEX_FILE", tmp_path / "inventory.json")
    ns = type("Recorders", (), {"downloads": [], "llm_calls": [], "llm_fails": False, "vision": True})()

    async def fake_download(asset_id):
        ns.downloads.append(asset_id)
        source = SOURCES[asset_id]
        item_id = inventory.item_id_for(asset_id)
        folder = inventory.ASSETS_DIR / item_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.glb").write_bytes(b"glTF....")
        thumbnail = None
        if source["thumbnail"]:
            (folder / "thumbnail.png").write_bytes(png_bytes())
            thumbnail = inventory.file_url(item_id, "thumbnail.png")
        return {"id": item_id, "assetId": asset_id, "name": source["title"], "provider": asset_id.split(":")[0],
                "author": None, "sourceUrl": None, "license": {"name": "CC0"}, "format": "glb", "modelPath": "model.glb",
                "modelUrl": inventory.file_url(item_id, "model.glb"), "thumbnailUrl": thumbnail,
                "description": source["description"], "tags": source["tags"], "price": 0.0, "currency": None,
                "dimensions": None, "unitScale": 1.0, "sizeBytes": 8, "downloadedAt": "2026-01-01T00:00:00+00:00"}

    async def fake_measure(url):
        return {"width": 50.0, "height": 90.0, "depth": 55.0}  # centimetres

    async def fake_llm(model, messages, schema):
        assert schema is FurnitureEnrichment
        ns.llm_calls.append(messages)
        if ns.llm_fails:
            raise LLMError("OpenRouter is not reachable.")
        text = messages[1]["content"] if isinstance(messages[1]["content"], str) else messages[1]["content"][0]["text"]
        name = next(n for n in ENRICHED if n in text)
        category, style, colors, materials, description = ENRICHED[name]
        return StructuredResult(parsed=FurnitureEnrichment(
            category=category, style=style, colors=colors, materials=materials, features=["simple lines"],
            description=description, matches_metadata=True), info=CallInfo(model=model))

    async def fake_vision(model):
        return ns.vision

    monkeypatch.setattr(inventory, "download_asset", fake_download)
    monkeypatch.setattr(pipeline, "measure_model", fake_measure)
    monkeypatch.setattr(enrichment_module, "structured_completion", fake_llm)
    monkeypatch.setattr(enrichment_module, "supports_image_input", fake_vision)
    return ns


def process(asset_id, extra=None):
    return asyncio.run(pipeline.process_asset(asset_id, extra))


def has_image(messages) -> bool:
    content = messages[1]["content"]
    return isinstance(content, list) and any(p.get("type") == "image_url" for p in content)


# --- 1. frontend download ------------------------------------------------------------------------

def test_1_frontend_download_is_enriched_and_stored(assets, furniture_store):
    response = asyncio.run(inventory_api.add_to_inventory(inventory_api.DownloadRequest(assetId="polyhaven:oak_chair")))
    processing = response["processing"]
    assert processing == {"furnitureId": response["id"], "created": True, "measured": True,
                          "enrichment": "vision", "stored": True, "warnings": []}
    record = asyncio.run(furniture_store.get(response["id"]))
    assert record["category"] == "chair" and record["materials"] == ["wood"] and record["style"] == "scandinavian"
    assert record["width"] == pytest.approx(0.5) and record["height"] == pytest.approx(0.9)  # cm -> m via unit scale
    assert record["model_url"] == response["modelUrl"] and record["thumbnail_url"] == response["thumbnailUrl"]
    assert record["observed"] == ["style", "colors", "materials", "features"]  # marked as image observations
    assert record["license"] == "CC0" and record["price"] == 0.0
    assert has_image(assets.llm_calls[0])


def test_frontend_rejects_assets_that_are_already_in_the_inventory(assets):
    asyncio.run(inventory_api.add_to_inventory(inventory_api.DownloadRequest(assetId="polyhaven:oak_chair")))
    with pytest.raises(inventory_api.HTTPException) as err:
        asyncio.run(inventory_api.add_to_inventory(inventory_api.DownloadRequest(assetId="polyhaven:oak_chair")))
    assert err.value.status_code == 409


# --- 2. research download --------------------------------------------------------------------------

def test_2_research_download_is_enriched_and_stored(assets, furniture_store, fake_llm, monkeypatch):
    fake_llm(intent="SEARCH_FURNITURE")
    research = install_research(monkeypatch, [call("search_assets", query="oak chair")],
                                [call("submit_candidates", summary="An oak chair.", local_assessment="Nothing local.",
                                      candidates=[{"asset_id": "polyhaven:oak_chair", "reason": "Oak"}])])
    monkeypatch.setattr(download_module, "process_asset", pipeline.process_asset)  # the real pipeline
    monkeypatch.setattr(FakeMCP, "search", _search_sources)
    store = RunStore()
    run = store.create("design")
    asyncio.run(execute_design_run(store, run, {"request": "Find an oak chair", "model": "fake",
                                                "room": Room(width=8, depth=6, height=2.8), "furniture": []}))
    asyncio.run(resume_design_run(store, run, validate_decision(
        run.pending_approval["candidates"], {"action": "approve", "selected": ["polyhaven:oak_chair"]})))
    added = run.result["downloads"]["added"][0]
    assert added["searchable"] is True and added["enrichment"] == "vision"
    record = asyncio.run(furniture_store.get(added["itemId"]))
    assert record and record["category"] == "chair"
    assert research.downloads == []  # the fake pipeline was not used


async def _search_sources(self, query, limit=12):
    words = query.lower().split()
    return {"results": [{"id": i, "type": "model", "title": s["title"], "formats": ["glb"], "downloadable": True}
                        for i, s in SOURCES.items() if any(w in s["title"].lower() for w in words)]}


# --- 3.-5. retrieval ---------------------------------------------------------------------------------

@pytest.fixture
def indexed(assets, furniture_store):
    for asset_id in SOURCES:
        process(asset_id)
    return furniture_store


def test_3_existing_asset_is_found_by_local_search(indexed):
    hits = asyncio.run(indexed.search("oak dining chair"))
    assert hits[0]["name"] == "Oak Dining Chair" and hits[0]["score"] > 0.4


def test_4_metadata_filters(indexed):
    chairs = asyncio.run(indexed.search("furniture to sit on", category="chair"))
    assert [h["category"] for h in chairs] == ["chair"]
    small = asyncio.run(indexed.search("furniture", max_width=0.4))
    assert small == []  # every test piece is 0.5 m wide


def test_5_semantic_search_ranks_by_meaning(indexed):
    hits = asyncio.run(indexed.search("comfortable chair for reading"))
    assert hits[0]["name"] == "Reading Armchair"
    sofa = next(h for h in hits if h["name"] == "Chesterfield Sofa")
    assert sofa["score"] < hits[0]["score"]


def test_search_api(indexed):
    from furniture.api import search_furniture
    result = asyncio.run(search_furniture(q="leather sofa", category="sofa", max_width=None, max_depth=None, limit=5))
    assert [r["name"] for r in result["results"]] == ["Chesterfield Sofa"]
    assert "vector" not in result["results"][0]


# --- 6./7. research: local first, MCP as fallback ------------------------------------------------------

def _start_research(request):
    store = RunStore()
    run = store.create("design")
    asyncio.run(execute_design_run(store, run, {"request": request, "model": "fake",
                                                "room": Room(width=8, depth=6, height=2.8), "furniture": []}))
    return run


def test_6_research_uses_local_results_without_mcp(indexed, fake_llm, monkeypatch):
    fake_llm(intent="SEARCH_FURNITURE")
    item_id = inventory.item_id_for("kenney:reading_armchair")
    research = install_research(monkeypatch, [call("submit_candidates", summary="You already have a reading armchair.",
                                                   local_assessment="The local armchair fits the request.",
                                                   candidates=[{"asset_id": item_id, "reason": "Made for reading"}])])
    run = _start_research("Find a comfortable armchair for reading")
    first_prompt = research.agent.requests[0][1]["content"]
    assert "Reading Armchair" in first_prompt  # local matches were searched before the first LLM call
    assert FakeMCP.instances == []  # no MCP session was opened
    assert "human_approval" not in [e["node"] for e in run.events if e["type"] == "node.started"]
    result = run.result["downloads"]
    assert result["status"] == "found_locally" and result["local"][0]["itemId"] == item_id
    assert result["local_assessment"] == "The local armchair fits the request."


def test_7_research_falls_back_to_mcp_when_local_is_not_enough(indexed, fake_llm, monkeypatch):
    fake_llm(intent="SEARCH_FURNITURE")
    install_research(monkeypatch, [call("search_assets", query="wooden cabinet")],
                     [call("submit_candidates", summary="Cabinets found externally.",
                           local_assessment="No cabinet in the database.",
                           candidates=[{"asset_id": "polyhaven:oak_cabinet", "reason": "Oak cabinet"}])])
    run = _start_research("Find a wooden cabinet")
    assert len(FakeMCP.instances) == 1  # MCP opened only for the external search
    assert run.status == "waiting_for_approval"
    assert [c["assetId"] for c in run.pending_approval["candidates"]] == ["polyhaven:oak_cabinet"]


def test_external_asset_already_in_inventory_counts_as_local(indexed, fake_llm, monkeypatch):
    fake_llm(intent="SEARCH_FURNITURE")
    install_research(monkeypatch, [call("submit_candidates", summary="s", local_assessment="a",
                                        candidates=[{"asset_id": "grab3d:leather_sofa", "reason": "Leather"}])])
    import research.agent as agent
    monkeypatch.setattr(agent, "inventory_asset_ids",
                        lambda: {"grab3d:leather_sofa": inventory.item_id_for("grab3d:leather_sofa")})
    run = _start_research("Find a leather sofa")
    assert run.result["downloads"]["status"] == "found_locally"  # no download of something we already have


# --- 8.-10. consistency -----------------------------------------------------------------------------------

def test_8_processing_again_creates_no_duplicates_and_no_new_llm_calls(assets, furniture_store):
    first = process("polyhaven:oak_chair")
    second = process("polyhaven:oak_chair")
    assert first.created and not second.created
    assert second.enrichment == "unchanged" and second.stored
    assert assets.downloads == ["polyhaven:oak_chair"] and len(assets.llm_calls) == 1
    assert asyncio.run(furniture_store.count()) == 1


def test_9_missing_image_and_metadata_do_not_break_the_pipeline(assets, furniture_store):
    result = process("kenney:reading_armchair")  # source without preview image
    assert result.stored and result.enrichment == "metadata"
    assert not has_image(assets.llm_calls[0])


def test_9_llm_failure_falls_back_to_metadata_description(assets, furniture_store):
    assets.llm_fails = True
    result = process("polyhaven:oak_chair")
    assert result.stored and result.enrichment == "fallback"
    assert "Description generation failed" in result.warnings[0]
    record = asyncio.run(furniture_store.get(result.item["id"]))
    assert record["category"] == "chair"  # guessed from name and tags
    assert record["observed"] == [] and record["style"] is None  # nothing invented


def test_9_database_failure_keeps_the_asset_and_reports_it(assets, furniture_store, monkeypatch):
    async def broken(record):
        raise OSError("disk full")
    monkeypatch.setattr(furniture_store, "upsert", broken)
    result = process("polyhaven:oak_chair")
    assert result.created and not result.stored
    assert "reindex" in result.warnings[0]
    assert inventory.load_index()[result.item["id"]]  # the download is not lost


def test_10_records_survive_a_restart(assets, furniture_store):
    result = process("polyhaven:oak_chair")
    reopened = FurnitureStore(furniture_store.path, embedder=HashEmbedder())  # new process, same files
    set_store(reopened)
    record = asyncio.run(reopened.get(result.item["id"]))
    assert record["name"] == "Oak Dining Chair"
    assert asyncio.run(reopened.search("oak chair"))[0]["id"] == result.item["id"]


def test_reindex_adds_missing_and_removes_stale_records(assets, furniture_store):
    process("polyhaven:oak_chair")
    asyncio.run(inventory.acquire_asset("grab3d:leather_sofa"))  # in the inventory, not yet indexed
    asyncio.run(furniture_store.upsert({**asyncio.run(furniture_store.get(inventory.item_id_for("polyhaven:oak_chair"))),
                                        "id": "ghost", "vector": None}))
    summary = asyncio.run(pipeline.reindex())
    assert summary["indexed"] == [inventory.item_id_for("grab3d:leather_sofa")]
    assert summary["unchanged"] == [inventory.item_id_for("polyhaven:oak_chair")]
    assert summary["removed"] == ["ghost"]


def test_deleting_an_inventory_item_removes_its_record(assets, furniture_store):
    result = process("polyhaven:oak_chair")
    asyncio.run(inventory_api.delete_from_inventory(result.item["id"]))
    assert asyncio.run(furniture_store.get(result.item["id"])) is None


def test_changed_embedding_model_is_re_embedded(assets, furniture_store):
    process("polyhaven:oak_chair")

    class OtherModel(HashEmbedder):
        model = "another-model"
    switched = FurnitureStore(furniture_store.path, embedder=OtherModel())
    assert asyncio.run(switched.ensure_model()) == 1
    assert asyncio.run(switched.ensure_model()) == 0


def test_description_function_is_usable_on_its_own(assets):
    result = asyncio.run(generate_furniture_description({"name": "Oak Dining Chair", "tags": ["chair"]}))
    assert result.category == "chair" and result.source == "metadata"
