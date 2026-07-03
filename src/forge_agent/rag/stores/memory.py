"""In-memory vector store for deterministic local RAG."""

from __future__ import annotations

import math

from forge_agent.rag.embeddings.base import EmbeddingVector
from forge_agent.rag.stores.base import (
    VectorSearchResult,
    VectorStoreItem,
)


class InMemoryVectorStore:
    """A minimal vector store backed by a Python list."""

    def __init__(self) -> None:
        self._items: list[VectorStoreItem] = []

    def add(self, items: list[VectorStoreItem]) -> None:
        """Add vectorized chunks to the store."""
        self._items.extend(items)

    def similarity_search(
        self,
        query_vector: EmbeddingVector,
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Return top-k chunks by cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        scored: list[VectorSearchResult] = []
        for item in self._items:
            score = _cosine_similarity(query_vector, item.vector)
            if score > 0:
                scored.append(VectorSearchResult(chunk=item.chunk, score=score))

        scored.sort(
            key=lambda result: (
                -result.score,
                result.chunk.metadata.relative_path,
                result.chunk.metadata.ordinal,
            )
        )
        return scored[:top_k]


def _cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)
