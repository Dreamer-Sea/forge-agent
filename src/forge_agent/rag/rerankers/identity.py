"""Identity reranker for pipeline composition tests."""

from __future__ import annotations

from forge_agent.rag.retrievers.base import SearchResult


class IdentityReranker:
    """Return retrieval results without changing their order."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return the input results unchanged, optionally truncated."""
        del query
        if top_k is None:
            return results
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        return results[:top_k]
