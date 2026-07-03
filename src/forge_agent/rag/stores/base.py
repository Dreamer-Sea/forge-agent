"""Vector store abstractions for RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from forge_agent.rag.chunker import Chunk
from forge_agent.rag.embeddings.base import EmbeddingVector


@dataclass(frozen=True, slots=True)
class VectorStoreItem:
    """A chunk and its vector representation."""

    chunk: Chunk
    vector: EmbeddingVector


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """A vector similarity search result."""

    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """Store and search vectorized chunks."""

    def add(self, items: list[VectorStoreItem]) -> None:
        """Add vectorized chunks."""

    def similarity_search(
        self,
        query_vector: EmbeddingVector,
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Return top-k most similar chunks."""
