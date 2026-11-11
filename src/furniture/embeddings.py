"""Text embeddings for furniture search, through the existing OpenRouter client.

Stored vectors and query vectors must come from the same model; the store records the model per
row and re-embeds everything when it changes (see FurnitureStore.ensure_model)."""
import os

import openai

from llm import LLMError, get_client

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))


class OpenRouterEmbedder:
    def __init__(self, model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM):
        self.model = model
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await get_client().embeddings.create(model=self.model, input=texts)
        except openai.APIError as exc:
            raise LLMError(f"Embedding request failed ({self.model}): {exc}")
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        if len(vectors) != len(texts) or any(len(v) != self.dim for v in vectors):
            raise LLMError(f"Embedding model {self.model} returned unexpected vectors (expected {self.dim} dimensions).")
        return vectors
