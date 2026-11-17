"""AI metadata enrichment for furniture: a short, search-oriented description plus category, style,
colours, materials and design features. Usable on its own, from the pipeline or from agents.

Verified data (name, provider, licence, measured size, price, source description and tags) is kept
as it is. What the model reads from the preview image is stored as observation (`observed`) and
never includes measurements or prices. Without an image or a vision model, the description is based
on the metadata alone; if the LLM fails, a deterministic fallback text is used.
"""
import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from furniture.store import CATEGORIES, category_of
from llm import DEFAULT_MODEL, LLMError, structured_completion, supports_image_input
from prompts import load_prompt

ENRICHMENT_MODEL = os.environ.get("ENRICHMENT_MODEL", DEFAULT_MODEL)
MAX_IMAGE_BYTES = 3 * 1024 * 1024
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}

Category = Literal[CATEGORIES]  # type: ignore[valid-type]


class FurnitureEnrichment(BaseModel):
    category: Category
    style: str | None = Field(description="e.g. modern, scandinavian, industrial, rustic; null if unclear")
    colors: list[str] = Field(description="Main visible colours, e.g. 'light oak', 'black'")
    materials: list[str] = Field(description="Materials as they appear, e.g. 'wood', 'leather', 'metal'; empty if unclear")
    features: list[str] = Field(description="Shape and design characteristics, e.g. 'curved backrest', 'tapered legs'")
    description: str = Field(description="One or two plain sentences for search, max. 40 words, no measurements or prices")
    matches_metadata: bool = Field(description="False if the image clearly shows something other than the metadata says")


SYSTEM_PROMPT = load_prompt("furniture_enrichment_system", categories=", ".join(CATEGORIES))


@dataclass
class EnrichmentResult:
    category: str
    description: str
    style: str | None = None
    colors: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    source: str = "fallback"          # vision | metadata | fallback
    observed: list[str] = field(default_factory=list)
    warning: str | None = None


def image_data_url(path: Path | None) -> str | None:
    """A local preview image as data URL, or None if it is missing, too large or not an image."""
    if not path or not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
        return None
    mime = IMAGE_TYPES.get(path.suffix.lower())  # explicit: mimetypes does not know .webp on every platform
    if not mime:
        return None
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _metadata_text(metadata: dict) -> str:
    keys = ("name", "provider", "author", "license", "description", "tags", "dimensions_m", "price", "formats")
    return "\n".join(f"{k}: {metadata[k]}" for k in keys if metadata.get(k) not in (None, "", []))


def fallback_description(metadata: dict, warning: str | None = None) -> EnrichmentResult:
    """No LLM: name, category guess from name/tags, and the source description."""
    words = " ".join([metadata.get("name") or "", *(metadata.get("tags") or [])]).lower()
    category = next((c for c in CATEGORIES if c.replace("_", " ") in words), None) \
        or next((c for c in CATEGORIES if c.rstrip("s") in words), "other")
    text = (metadata.get("description") or "").strip()
    description = f"{metadata.get('name', 'Furniture')}. {text}"[:300].strip()
    return EnrichmentResult(category=category, description=description, source="fallback", warning=warning)


async def generate_furniture_description(metadata: dict, image: str | None = None,
                                         model: str = ENRICHMENT_MODEL) -> EnrichmentResult:
    """Structured description from metadata (+ preview image as data URL, if any)."""
    vision = image is not None and await supports_image_input(model) is not False
    text = f"Metadata (verified):\n{_metadata_text(metadata)}"
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image}}] if vision else text
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    try:
        response = await structured_completion(model, messages, FurnitureEnrichment)
    except LLMError as exc:
        return fallback_description(metadata, warning=f"Description generation failed: {exc}")
    parsed = response.parsed
    result = EnrichmentResult(
        category=category_of(parsed.category) or "other",
        description=parsed.description.strip(),
        style=parsed.style or None,
        colors=parsed.colors[:6], materials=parsed.materials[:6], features=parsed.features[:8],
        source="vision" if vision else "metadata",
    )
    if vision:
        result.observed = [k for k in ("style", "colors", "materials", "features") if getattr(result, k)]
        if not parsed.matches_metadata:
            result.warning = "The preview image does not seem to match the metadata."
    elif image is not None:
        result.warning = f"Model {model} cannot read images; description from metadata only."
    return result
