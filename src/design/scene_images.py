"""Loads the latest render_scene screenshots for an agent's multimodal LLM call."""
import os

from design.state import DesignState
from llm import LLMError, supports_image_input
from observability import progress
from rendering.service import image_data_url

# Explicit opt-in: let models without image input work from the structured data alone.
TEXT_ONLY_FALLBACK = os.environ.get("DESIGNER_TEXT_ONLY_FALLBACK") == "1"
VIEW_LABELS = {"top_view": "Top-down view", "perspective_view": "Perspective view"}


async def load_scene_images(state: DesignState, agent: str, model: str) -> list[tuple[str, str]]:
    """(view, data URL) pairs for `agent` running on `model`. A model without image input is an error
    unless the text-only fallback is enabled explicitly; it never switches to another model."""
    refs = state.get("scene_images") or {}
    if not refs:
        return []
    if await supports_image_input(model) is False:
        if TEXT_ONLY_FALLBACK:
            progress.step("text_only", "Model without image input: structured data only", model=model)
            return []
        raise LLMError(f"Model '{model}' does not accept image input, which the {agent} needs "
                       "for the room screenshots. Choose a vision-capable model "
                       "(or set DESIGNER_TEXT_ONLY_FALLBACK=1 to work from room data only).", status_code=400)
    with progress.stage("images", "Load screenshots") as details:
        images = [(view, image_data_url(ref)) for view, ref in refs.items()]
        details.update(images=list(refs), layoutVersion=state.get("rendered_layout_version", 0))
    return images


def image_content(text: str, images: list[tuple[str, str]], version: int) -> str | list[dict]:
    """One multimodal user message: the text followed by the labelled screenshots."""
    if not images:
        return text
    content = [{"type": "text", "text": text}]
    for view, data_url in images:
        label = f"{VIEW_LABELS.get(view, view)} of layout v{version}{' (the confirmed room)' if version == 0 else ''}:"
        content += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": data_url}}]
    return content
