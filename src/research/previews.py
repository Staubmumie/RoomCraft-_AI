"""Fetches asset preview images so a vision model can look at them. Images are kept in memory
for the LLM call only and never stored in the graph state."""
import base64

import httpx

from url_safety import UnsafeURLError, public_http_client

MAX_PREVIEW_BYTES = 3 * 1024 * 1024
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


class PreviewError(Exception):
    pass


async def fetch_preview(url: str) -> str:
    """Returns the image as a data URL, or raises PreviewError."""
    try:
        async with public_http_client(timeout=15) as client:
            response = await client.get(url)
            response.raise_for_status()
    except UnsafeURLError as exc:
        raise PreviewError(str(exc))
    except httpx.HTTPError as exc:
        raise PreviewError(f"Preview could not be loaded ({type(exc).__name__}).")
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in IMAGE_TYPES:
        raise PreviewError(f"Preview is not a supported image ({content_type or 'unknown type'}).")
    if not response.content or len(response.content) > MAX_PREVIEW_BYTES:
        raise PreviewError("Preview image is empty or too large.")
    return f"data:{content_type};base64,{base64.b64encode(response.content).decode()}"
