from __future__ import annotations

from forge_agent.memory.models import MemoryScope, MemorySearchResult, MemoryType
from forge_agent.memory.store import MemoryStore


class MemoryRetriever:
    """Policy-aware retrieval facade for long-term memory."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        scope_filter: MemoryScope | None = None,
        scope_id: str | None = None,
        type_filter: MemoryType | None = None,
        min_score: float = 0.0,
    ) -> list[MemorySearchResult]:
        """Retrieve relevant memories with optional scope, type, and score filters."""

        if top_k <= 0:
            return []

        results = self._store.search(
            query,
            scope=scope_filter,
            top_k=top_k,
            scope_id=scope_id,
            type_filter=type_filter,
        )

        filtered = [result for result in results if result.score >= min_score]

        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(filtered[:top_k], start=1)
        ]
