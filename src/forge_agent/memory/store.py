from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from forge_agent.memory.models import MemoryRecord, MemoryScope, MemorySearchResult, MemoryType


@runtime_checkable
class MemoryStore(Protocol):
    """Storage boundary for long-term memory backends."""

    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Persist one memory record and return the stored record."""

        ...

    def search(
        self,
        query: str,
        scope: MemoryScope | None = None,
        top_k: int = 5,
        *,
        scope_id: str | None = None,
        type_filter: MemoryType | None = None,
    ) -> list[MemorySearchResult]:
        """Search memory records by query and optional filters."""

        ...

    def list(
        self,
        scope: MemoryScope | None = None,
        *,
        scope_id: str | None = None,
        type_filter: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        """List memory records, optionally filtered by scope and type."""

        ...

    def update(self, record_id: str, patch: dict[str, Any]) -> MemoryRecord:
        """Patch one memory record and return the updated record."""

        ...

    def delete(self, record_id: str) -> None:
        """Delete one memory record by id."""

        ...
