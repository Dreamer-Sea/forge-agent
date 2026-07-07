from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from forge_agent.memory import (
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryStore,
    MemoryType,
)


class FakeMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.id] = record
        return record

    def search(
        self,
        query: str,
        scope: MemoryScope | None = None,
        top_k: int = 5,
    ) -> list[MemorySearchResult]:
        normalized_query = query.lower()
        results: list[MemorySearchResult] = []

        for record in self._records.values():
            if scope is not None and record.scope != scope:
                continue

            if normalized_query not in record.content.lower():
                continue

            results.append(
                MemorySearchResult(
                    record=record,
                    score=1.0,
                    reason="deterministic fake store substring match",
                    rank=len(results) + 1,
                )
            )

        return results[:top_k]

    def list(self, scope: MemoryScope | None = None) -> list[MemoryRecord]:
        records = list(self._records.values())
        if scope is None:
            return records
        return [record for record in records if record.scope == scope]

    def update(self, record_id: str, patch: dict[str, Any]) -> MemoryRecord:
        record = self._records[record_id]
        updated_patch = dict(patch)
        updated_patch["updated_at"] = datetime.now(UTC)
        updated = record.model_copy(update=updated_patch)
        self._records[record_id] = updated
        return updated

    def delete(self, record_id: str) -> None:
        del self._records[record_id]


def test_fake_memory_store_matches_protocol_at_runtime() -> None:
    store = FakeMemoryStore()

    assert isinstance(store, MemoryStore)


def test_memory_store_protocol_supports_add_list_search_update_and_delete() -> None:
    store: MemoryStore = FakeMemoryStore()
    record = MemoryRecord(
        type=MemoryType.SEMANTIC,
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
        content="项目默认使用 Python 3.13 和 uv",
        source="test",
    )

    stored = store.add(record)

    assert stored == record
    assert store.list() == [record]
    assert store.list(scope=MemoryScope.USER) == []

    results = store.search("Python 3.13", scope=MemoryScope.PROJECT, top_k=5)
    assert len(results) == 1
    assert results[0].record == record
    assert results[0].score == 1.0

    updated = store.update(record.id, {"content": "项目默认使用 Python 3.13、uv 和 Typer"})
    assert updated.content == "项目默认使用 Python 3.13、uv 和 Typer"
    assert updated.updated_at >= updated.created_at

    store.delete(record.id)
    assert store.list() == []


def test_memory_store_protocol_raises_for_unknown_update_id() -> None:
    store: MemoryStore = FakeMemoryStore()

    with pytest.raises(KeyError):
        store.update("missing", {"content": "new content"})
