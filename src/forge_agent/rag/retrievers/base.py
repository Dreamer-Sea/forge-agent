"""Retriever abstractions for RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from forge_agent.rag.chunker import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A ranked retrieval result."""

    chunk: Chunk
    score: float
    rank: int


class Retriever(Protocol):
    """Search interface for local or remote retrieval backends."""

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return ranked chunks for a query."""
