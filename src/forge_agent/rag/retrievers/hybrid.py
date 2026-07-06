"""Hybrid retriever for combining keyword and vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from forge_agent.rag.chunker import Chunk
from forge_agent.rag.retrievers.base import SearchResult
from forge_agent.rag.retrievers.keyword import KeywordRetriever
from forge_agent.rag.retrievers.vector import VectorRetriever


@dataclass(frozen=True, slots=True)
class _HybridCandidate:
    chunk: Chunk
    score: float


class HybridRetriever:
    """Combine keyword and vector retrieval with reciprocal-rank fusion.

    The implementation is intentionally deterministic and lightweight. Keyword
    retrieval gives precise lexical matches, while vector retrieval gives a
    second semantic signal. Results are merged by chunk id and re-ranked with a
    weighted reciprocal-rank score.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        keyword_retriever: KeywordRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        default_top_k: int = 5,
        keyword_weight: float = 1.0,
        vector_weight: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than 0")
        if keyword_weight <= 0:
            raise ValueError("keyword_weight must be greater than 0")
        if vector_weight <= 0:
            raise ValueError("vector_weight must be greater than 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        self._default_top_k = default_top_k
        self._keyword_retriever = keyword_retriever or KeywordRetriever(
            chunks,
            default_top_k=default_top_k,
        )
        self._vector_retriever = vector_retriever or VectorRetriever(
            chunks,
            default_top_k=default_top_k,
        )
        self._keyword_weight = keyword_weight
        self._vector_weight = vector_weight
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search chunks with both keyword and vector retrievers."""
        limit = self._default_top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than 0")

        candidate_depth = max(limit * 2, self._default_top_k)

        keyword_results = self._keyword_retriever.search(
            query,
            top_k=candidate_depth,
        )
        vector_results = self._vector_retriever.search(
            query,
            top_k=candidate_depth,
        )

        candidates: dict[str, _HybridCandidate] = {}

        self._merge_results(
            candidates=candidates,
            results=keyword_results,
            weight=self._keyword_weight,
        )
        self._merge_results(
            candidates=candidates,
            results=vector_results,
            weight=self._vector_weight,
        )

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.chunk.metadata.relative_path,
                candidate.chunk.metadata.ordinal,
            ),
        )

        return [
            SearchResult(
                chunk=candidate.chunk,
                score=candidate.score,
                rank=rank,
            )
            for rank, candidate in enumerate(ranked_candidates[:limit], start=1)
        ]

    def _merge_results(
        self,
        *,
        candidates: dict[str, _HybridCandidate],
        results: list[SearchResult],
        weight: float,
    ) -> None:
        for result in results:
            chunk_id = result.chunk.metadata.chunk_id
            score = weight / (self._rrf_k + result.rank)

            existing = candidates.get(chunk_id)
            if existing is None:
                candidates[chunk_id] = _HybridCandidate(
                    chunk=result.chunk,
                    score=score,
                )
                continue

            candidates[chunk_id] = _HybridCandidate(
                chunk=existing.chunk,
                score=existing.score + score,
            )
