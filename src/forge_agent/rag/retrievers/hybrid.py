"""Hybrid retriever with Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from forge_agent.rag.chunker import Chunk
from forge_agent.rag.retrievers.base import Retriever, SearchResult


@dataclass(slots=True)
class _FusedCandidate:
    chunk: Chunk
    score: float
    best_rank: int


class HybridRetriever:
    """Combine multiple retrievers with rank-based fusion.

    Hybrid retrieval is useful when different retrievers have incompatible
    score scales. Keyword retrieval may use BM25-style scores, while vector
    retrieval may use cosine similarity or dot product. Reciprocal Rank Fusion
    avoids mixing raw scores directly and only depends on each result's rank.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        default_top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("retrievers must not be empty")
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        self._retrievers = tuple(retrievers)
        self._default_top_k = default_top_k
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search all child retrievers and return fused ranked results."""
        limit = self._default_top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than 0")

        candidates: dict[str, _FusedCandidate] = {}

        for retriever in self._retrievers:
            for result in retriever.search(query, top_k=limit):
                chunk_id = result.chunk.metadata.chunk_id
                rrf_score = 1.0 / (self._rrf_k + result.rank)

                candidate = candidates.get(chunk_id)
                if candidate is None:
                    candidates[chunk_id] = _FusedCandidate(
                        chunk=result.chunk,
                        score=rrf_score,
                        best_rank=result.rank,
                    )
                    continue

                candidate.score += rrf_score
                candidate.best_rank = min(candidate.best_rank, result.rank)

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.best_rank,
                candidate.chunk.metadata.relative_path,
                candidate.chunk.metadata.ordinal,
                candidate.chunk.metadata.chunk_id,
            ),
        )

        return [
            SearchResult(chunk=candidate.chunk, score=candidate.score, rank=rank)
            for rank, candidate in enumerate(ranked_candidates[:limit], start=1)
        ]
