"""Reranker abstractions for RAG pipelines."""

from __future__ import annotations

from typing import Protocol

from forge_agent.rag.retrievers.base import SearchResult


class Reranker(Protocol):
    """Reorder first-stage retrieval results."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return reranked retrieval results."""
