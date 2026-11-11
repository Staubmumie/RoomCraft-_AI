"""Fakes for the furniture research path: MCP server, the agent's tool-calling LLM, preview images
and downloads. Shared by the research tests and the designer -> research workflow tests."""
import json

from fastapi import HTTPException

import design.asset_download as download_module
import research.agent as agent
from furniture.pipeline import ProcessResult
from llm import CallInfo
from research.mcp_client import AssetMCPError

CATALOG = {
    "polyhaven:oak_cabinet": {"type": "model", "title": "Oak Cabinet", "source": "polyhaven", "formats": ["gltf", "blend"],
                              "license": "CC0", "thumbnail": "https://cdn.example.com/oak.png", "downloadable": True,
                              "tags": ["cabinet", "oak"], "description": "A modern oak cabinet with two doors."},
    "blenderkit:walnut_cabinet": {"type": "model", "title": "Walnut Cabinet", "source": "blenderkit", "formats": ["glb"],
                                  "license": "Royalty Free", "thumbnail": "https://cdn.example.com/walnut.jpg",
                                  "downloadable": True, "description": "Walnut sideboard."},
    "grab3d:dark_cabinet": {"type": "model", "title": "Dark Cabinet", "source": "grab3d", "formats": ["glb"],
                            "license": "CC0", "downloadable": True},  # no preview image
    "opengameart:cabinet_scene": {"type": "other", "title": "Cabinet scene", "source": "opengameart", "downloadable": True},
    "polyhaven:fbx_cabinet": {"type": "model", "title": "FBX Cabinet", "source": "polyhaven", "formats": ["fbx"],
                              "downloadable": True},
    "grab3d:black_cabinet": {"type": "model", "title": "Black Cabinet", "source": "grab3d", "formats": ["glb"],
                             "license": "CC0", "thumbnail": "https://cdn.example.com/black.png", "downloadable": True},
    "polyfork:pine_cabinet": {"type": "model", "title": "Pine Cabinet", "source": "polyfork", "formats": ["glb"],
                              "license": "Royalty Free", "downloadable": True},
}


class FakeMCP:
    instances: list["FakeMCP"] = []
    fail_on_connect = False
    fail_search = False

    def __init__(self):
        self.calls = []
        FakeMCP.instances.append(self)

    async def __aenter__(self):
        if FakeMCP.fail_on_connect:
            raise AssetMCPError("MCP server https://3d.shep.bot/mcp is not reachable (ConnectError).")
        return self

    async def __aexit__(self, *exc):
        return False

    async def search(self, query, limit=12):
        self.calls.append(("search", query))
        if FakeMCP.fail_search:
            raise AssetMCPError("MCP call search_assets failed (timeout).")
        words = query.lower().split()
        results = [{"id": i, **a} for i, a in CATALOG.items() if any(w in a["title"].lower() for w in words)]
        return {"query": query, "results": results[:limit]}

    async def get_asset(self, asset_id):
        self.calls.append(("get_asset", asset_id))
        if asset_id not in CATALOG:
            raise AssetMCPError(f"Unknown asset {asset_id}")
        return {"id": asset_id, **CATALOG[asset_id],
                "selection": {"files": [{"filename": "model.glb", "format": "glb", "size": "1 MB"}]}}


def call(name, **arguments):
    call.counter += 1
    return {"id": f"call_{call.counter}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


call.counter = 0


class ScriptedAgent:
    """Plays a list of assistant turns; each turn is a list of tool calls."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests: list[list[dict]] = []

    async def __call__(self, model, messages, tools, tool_choice="auto"):
        self.requests.append(json.loads(json.dumps(messages)))  # snapshot
        calls = self.turns.pop(0)
        return {"role": "assistant", "content": "", "tool_calls": calls}, CallInfo(model=model)


ROUND_1 = [
    [call("search_assets", query="wooden cabinet")],
    [call("get_asset", asset_id="polyhaven:oak_cabinet"),
     call("view_preview", asset_id="polyhaven:oak_cabinet"),
     call("view_preview", asset_id="grab3d:dark_cabinet")],
    [call("submit_candidates", summary="Two wooden cabinets.", candidates=[
        {"asset_id": "polyhaven:oak_cabinet", "reason": "Light oak, modern doors.", "visual_check": "Looks like a cabinet."},
        {"asset_id": "blenderkit:walnut_cabinet", "reason": "Warm walnut."},
        {"asset_id": "opengameart:cabinet_scene", "reason": "filtered: not a model"},
        {"asset_id": "ghost:asset", "reason": "never found"},
    ])],
]
ROUND_2 = [
    [call("search_assets", query="dark cabinet")],
    [call("submit_candidates", summary="Darker options.", candidates=[
        {"asset_id": "polyhaven:oak_cabinet", "reason": "excluded now"},
        {"asset_id": "grab3d:dark_cabinet", "reason": "Dark wood."},
        {"asset_id": "grab3d:black_cabinet", "reason": "Black lacquer."},
    ])],
]
ROUND_3 = [
    [call("search_assets", query="pine cabinet")],
    [call("submit_candidates", summary="Pine.", candidates=[{"asset_id": "polyfork:pine_cabinet", "reason": "Pine."}])],
]


def install_research(monkeypatch, *turns):
    """Installs fakes for MCP, LLM tool calls, previews and downloads; returns a namespace of recorders."""
    FakeMCP.instances, FakeMCP.fail_on_connect, FakeMCP.fail_search = [], False, False
    ns = type("Recorders", (), {})()
    ns.agent = ScriptedAgent(*(turns or (*ROUND_1, *ROUND_2, *ROUND_3)))
    ns.previews, ns.downloads, ns.updates = [], [], []
    ns.vision = True
    ns.download_errors = {}

    async def fake_preview(url):
        ns.previews.append(url)
        return "data:image/png;base64,iVBORw0KGgo="

    async def fake_vision(model):
        return ns.vision

    async def fake_process(asset_id, extra=None):
        """Stands in for furniture.pipeline.process_asset (tested on its own in test_furniture_db.py)."""
        ns.downloads.append(asset_id)
        ns.updates.append((asset_id, extra))
        if asset_id in ns.download_errors:
            raise HTTPException(422, ns.download_errors[asset_id])
        item_id = asset_id.replace(":", "-")
        item = {"id": item_id, "assetId": asset_id, "name": CATALOG[asset_id]["title"], "format": "glb",
                "modelUrl": f"/api/inventory/{item_id}/files/model.glb",
                "dimensions": {"width": 80.0, "height": 180.0, "depth": 40.0}, "unitScale": 0.01}  # centimetres
        return ProcessResult(item=item, created=True, measured=True, enrichment="vision", stored=True)

    monkeypatch.setattr(agent, "AssetMCP", FakeMCP)
    monkeypatch.setattr(agent, "tool_completion", ns.agent)
    monkeypatch.setattr(agent, "fetch_preview", fake_preview)
    monkeypatch.setattr(agent, "supports_image_input", fake_vision)
    monkeypatch.setattr(agent, "inventory_asset_ids", lambda: {})  # never the user's real inventory
    monkeypatch.setattr(download_module, "process_asset", fake_process)
    return ns
