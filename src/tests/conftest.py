import io
import uuid

import pytest
from PIL import Image

import design.interior_critic as critic_module
import design.interior_designer as designer_module
import design.orchestrator as orchestrator_module
import design.render_scene as render_scene_module
import design.scene_images as scene_images_module
import rendering.service as render_service
from design.interior_critic import Critique
from design.interior_designer import (DesignerDecision, DesignPlan, FurnitureGap, FurnitureRequest, NewPiece,
                                      PlacementProposal)
from design.orchestrator import IntentDecision
from llm import CallInfo, StructuredResult
from rendering.service import RenderResult


class HashEmbedder:
    """Deterministic stand-in for the embedding API: hashed bag of words, L2-normalised.
    Texts that share words get a high cosine similarity, unrelated texts a low one."""
    model = "test-hash-embedder"
    dim = 256

    async def embed(self, texts):
        import hashlib
        import math
        import re
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in re.findall(r"[a-z]+", text.lower()):
                word = word[:-1] if word.endswith("s") and len(word) > 3 else word  # chairs -> chair
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


@pytest.fixture(autouse=True)
def furniture_store(tmp_path):
    """Every test gets its own furniture database (real LanceDB in a temp dir, fake embeddings)."""
    from furniture.store import FurnitureStore, set_store
    store = FurnitureStore(tmp_path / "furniture_db", embedder=HashEmbedder())
    set_store(store)
    yield store
    set_store(None)


def thread() -> dict:
    """LangGraph config with a fresh thread id (the graph has a checkpointer)."""
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def png_bytes(color=(200, 180, 150), size=8) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def P(id, x, z, rotation_deg=0.0, y=0.0) -> PlacementProposal:  # noqa: N802 - short helper for test scripts
    return PlacementProposal(id=id, x=x, y=y, z=z, rotation_deg=rotation_deg)


def decide(*placements, add=(), remove=(), action="validate", finish=True, request=None,
           explanation="test", plan=None, gaps=(), strategy="none") -> DesignerDecision:
    """One scripted designer step. `add` takes (item_id, x, z[, rotation_deg[, y]]) tuples; `plan` a DesignPlan
    dict; `gaps` (item, essential) tuples for the completeness check."""
    furniture_request = FurnitureRequest(description=request, style=None, size=None, colors_materials=None,
                                         budget=None) if request else None
    new = [NewPiece(item_id=a[0], x=a[1], z=a[2], rotation_deg=a[3] if len(a) > 3 else 0.0,
                    y=a[4] if len(a) > 4 else 0.0) for a in add]
    return DesignerDecision(plan=DesignPlan.model_validate(plan) if plan else None, explanation=explanation,
                            placements=list(placements), add=new, remove=list(remove),
                            furniture_gaps=[FurnitureGap(item=item, essential=essential, reason="requested")
                                            for item, essential in gaps],
                            repair_strategy=strategy, next_action=action, finish_when_valid=finish,
                            furniture_request=furniture_request)


# Placeable inventory items the fake designer sees (sizes in metres, like available_items()).
INVENTORY = {
    "lamp-item": {"item_id": "lamp-item", "name": "Floor Lamp", "scale": 1.0, "width": 0.4, "depth": 0.4, "height": 1.6},
    "chair-item": {"item_id": "chair-item", "name": "Armchair", "scale": 0.01, "width": 0.8, "depth": 0.9, "height": 1.0},
}


def critique(summary="Looks balanced.", issues=0) -> Critique:
    return Critique.model_validate({
        "summary": summary,
        "scores": {"functionality": 4, "aesthetics": 4, "space_utilization": 3, "preference_fit": 4},
        "strengths": ["Clear walkway"],
        "issues": [{"severity": "medium", "description": f"Issue {i}", "furniture_ids": []} for i in range(issues)],
        "suggestions": ["Add a rug"],
    })


class FakeLLM:
    """Scripted replacement for every structured LLM call (orchestrator, designer, critic) and for
    the renderer, which stores two small PNGs."""

    def __init__(self, intent: str, steps: tuple, renders_dir, suspicious=False):
        self.intent = intent
        self.suspicious = suspicious
        self.steps = list(steps)
        self.critiques: list[Critique] = []
        self.renders_dir = renders_dir
        self.orchestrator_calls: list[list[dict]] = []
        self.designer_calls: list[list[dict]] = []
        self.critic_calls: list[list[dict]] = []
        self.render_calls: list[tuple] = []
        self.render_triggers: list[str] = []  # initial | validation | critic per render
        self.models_used: list[tuple[str, str]] = []  # (schema name, model) per LLM call
        self.vision = True
        self.inventory = dict(INVENTORY)

    async def complete(self, model, messages, schema):
        self.models_used.append((schema.__name__, model))
        if schema is IntentDecision:
            self.orchestrator_calls.append(messages)
            parsed = IntentDecision(intent=self.intent, suspicious=self.suspicious, reason="test")
        elif schema is DesignerDecision:
            self.designer_calls.append(messages)
            parsed = self.steps[min(len(self.designer_calls), len(self.steps)) - 1]
        elif schema is Critique:
            self.critic_calls.append(messages)
            parsed = (self.critiques or [critique()])[min(len(self.critic_calls), len(self.critiques) or 1) - 1]
        else:
            raise AssertionError(f"unexpected schema {schema}")
        return StructuredResult(parsed=parsed, info=CallInfo(model=model))

    async def render(self, room, furniture, stage=None, *, run_id=None, layout_version=0, trigger="adhoc"):
        """Stores two small PNGs in a run folder laid out like rendering.service does."""
        self.render_calls.append((room, [f.model_copy() for f in furniture]))
        self.render_triggers.append(trigger)
        key = f"fake{len(self.render_calls)}"
        folder = f"runs/{run_id or '_adhoc'}/{len(self.render_calls):02d}_v{layout_version}_{trigger}"
        target = self.renders_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / "top.png").write_bytes(png_bytes((10, 20, 30)))
        (target / "perspective.png").write_bytes(png_bytes((40, 50, 60)))
        return RenderResult(key=key, images={"top_view": f"renders/{folder}/top.png",
                                             "perspective_view": f"renders/{folder}/perspective.png"},
                            cached=False, duration_ms=1, folder=f"renders/{folder}")

    async def supports_image_input(self, model):
        return self.vision


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """fake_llm(*designer_steps, intent="DESIGN_ROOM") installs the fakes and returns them."""
    monkeypatch.setattr(render_service, "DATA_DIR", tmp_path)
    monkeypatch.setattr(render_service, "RENDERS_DIR", tmp_path / "renders")

    def install(*steps, intent="DESIGN_ROOM", suspicious=False):
        fake = FakeLLM(intent, steps, tmp_path / "renders", suspicious)
        for module in (orchestrator_module, designer_module, critic_module):
            monkeypatch.setattr(module, "structured_completion", fake.complete)
        monkeypatch.setattr(scene_images_module, "supports_image_input", fake.supports_image_input)
        monkeypatch.setattr(render_scene_module, "render_room", fake.render)
        monkeypatch.setattr(designer_module, "available_items", lambda: fake.inventory)
        _block_research_network(monkeypatch)
        return fake
    return install


def _block_research_network(monkeypatch):
    """Research tests install their own fakes; anything else must never reach the network."""
    import research.agent as agent

    async def no_network(*args, **kwargs):
        raise AssertionError("unexpected network call in a test")

    class NoMCP:
        async def __aenter__(self):
            raise AssertionError("unexpected MCP connection in a test")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(agent, "tool_completion", no_network)
    monkeypatch.setattr(agent, "supports_image_input", no_network)
    monkeypatch.setattr(agent, "fetch_preview", no_network)
    monkeypatch.setattr(agent, "AssetMCP", NoMCP)
