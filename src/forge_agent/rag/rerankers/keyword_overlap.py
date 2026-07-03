"""Keyword-overlap reranker baseline for RAG pipelines."""

from __future__ import annotations

from forge_agent.rag.retrievers import SearchResult
from forge_agent.rag.retrievers.keyword import expand_query_tokens, tokenize


class KeywordOverlapReranker:
    """Rerank retrieval results by deterministic query-token overlap.

    This is a test-friendly baseline reranker. It is intentionally simple and
    deterministic so retrieval and reranking can be evaluated independently.
    It is not intended to replace production cross-encoder or hosted rerankers.
    """

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results sorted by keyword overlap with the query."""
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = set(expand_query_tokens(query))
        limit = len(results) if top_k is None else top_k

        if not query_tokens:
            return self._with_updated_ranks(results[:limit])

        ranked = sorted(
            results,
            key=lambda result: (
                -self._overlap_score(query_tokens, result),
                result.rank,
                result.chunk.metadata.relative_path,
                result.chunk.metadata.ordinal,
                result.chunk.metadata.chunk_id,
            ),
        )

        return self._with_updated_ranks(ranked[:limit])

    def _overlap_score(
        self,
        query_tokens: set[str],
        result: SearchResult,
    ) -> float:
        candidate_tokens = set(tokenize(self._candidate_text(result)))
        if not candidate_tokens:
            return 0.0

        matched_tokens = query_tokens & candidate_tokens
        return len(matched_tokens) / len(query_tokens)

    def _candidate_text(self, result: SearchResult) -> str:
        chunk = result.chunk
        metadata = chunk.metadata
        return "\n".join(
            [
                metadata.title,
                metadata.relative_path,
                " ".join(metadata.heading_path),
                chunk.content,
            ]
        )

    def _with_updated_ranks(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk=result.chunk,
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(results, start=1)
        ]
