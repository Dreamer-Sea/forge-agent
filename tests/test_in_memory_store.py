from __future__ import annotations

import pytest

from forge_agent.memory import InMemoryStore, MemoryRecord, MemoryScope, MemoryType


def _record(
    content: str,
    *,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    scope: MemoryScope = MemoryScope.PROJECT,
    scope_id: str | None = "forge-agent",
    importance: float = 0.5,
    confidence: float = 0.5,
) -> MemoryRecord:
    return MemoryRecord(
        type=memory_type,
        scope=scope,
        scope_id=scope_id,
        content=content,
        source="test",
        importance=importance,
        confidence=confidence,
    )


def test_in_memory_store_can_add_and_list_records() -> None:
    store = InMemoryStore()
    record = _record("项目默认使用 Python 3.13 和 uv")

    stored = store.add(record)

    assert stored == record
    assert store.list() == [record]
    assert store.list(scope=MemoryScope.PROJECT) == [record]
    assert store.list(scope=MemoryScope.USER) == []


def test_in_memory_store_search_returns_ranked_top_k_results() -> None:
    store = InMemoryStore()
    less_important = store.add(
        _record(
            "Python 3.13 是项目默认版本",
            importance=0.4,
            confidence=0.7,
        )
    )
    more_important = store.add(
        _record(
            "项目默认使用 Python 3.13 和 uv",
            importance=0.9,
            confidence=0.9,
        )
    )
    store.add(_record("workspace guard 会阻止路径逃逸"))

    results = store.search("Python 3.13 uv", scope=MemoryScope.PROJECT, top_k=2)

    assert [result.record for result in results] == [more_important, less_important]
    assert results[0].rank == 1
    assert results[0].score > 0
    assert "keyword overlap matched" in results[0].reason


def test_in_memory_store_can_filter_by_scope_id_and_type() -> None:
    store = InMemoryStore()
    session_a = store.add(
        _record(
            "demo-a session 使用 Python 3.13",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.SESSION,
            scope_id="demo-a",
        )
    )
    store.add(
        _record(
            "demo-b session 使用 Python 3.12",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.SESSION,
            scope_id="demo-b",
        )
    )
    store.add(
        _record(
            "demo-a 任务摘要",
            memory_type=MemoryType.SUMMARY,
            scope=MemoryScope.SESSION,
            scope_id="demo-a",
        )
    )

    results = store.search(
        "Python",
        scope=MemoryScope.SESSION,
        scope_id="demo-a",
        type_filter=MemoryType.SEMANTIC,
    )

    assert [result.record for result in results] == [session_a]


def test_in_memory_store_deduplicates_same_type_scope_scope_id_and_content() -> None:
    store = InMemoryStore()
    first = _record("项目默认使用 Python 3.13 和 uv")
    duplicate = _record("  项目默认使用 Python 3.13 和 uv  ")

    stored_first = store.add(first)
    stored_duplicate = store.add(duplicate)

    assert stored_duplicate == stored_first
    assert store.list() == [stored_first]


def test_in_memory_store_allows_same_content_in_different_scope_ids() -> None:
    store = InMemoryStore()
    session_a = store.add(
        _record(
            "用户偏好使用 uv",
            scope=MemoryScope.SESSION,
            scope_id="session-a",
        )
    )
    session_b = store.add(
        _record(
            "用户偏好使用 uv",
            scope=MemoryScope.SESSION,
            scope_id="session-b",
        )
    )

    assert session_a.id != session_b.id
    assert len(store.list(scope=MemoryScope.SESSION)) == 2
    assert store.list(scope=MemoryScope.SESSION, scope_id="session-a") == [session_a]
    assert store.list(scope=MemoryScope.SESSION, scope_id="session-b") == [session_b]


def test_in_memory_store_can_update_record() -> None:
    store = InMemoryStore()
    record = store.add(_record("项目默认使用 Python 3.12"))

    updated = store.update(record.id, {"content": "项目默认使用 Python 3.13 和 uv"})

    assert updated.id == record.id
    assert updated.content == "项目默认使用 Python 3.13 和 uv"
    assert updated.updated_at >= record.updated_at
    assert store.list() == [updated]


def test_in_memory_store_rejects_update_that_creates_duplicate() -> None:
    store = InMemoryStore()
    store.add(_record("项目默认使用 Python 3.13 和 uv"))
    second = store.add(_record("项目默认使用 Python 3.12"))

    with pytest.raises(ValueError, match="duplicates existing record"):
        store.update(second.id, {"content": "项目默认使用 Python 3.13 和 uv"})


def test_in_memory_store_can_delete_record() -> None:
    store = InMemoryStore()
    record = store.add(_record("项目默认使用 Python 3.13 和 uv"))

    store.delete(record.id)

    assert store.list() == []


def test_in_memory_store_raises_for_missing_delete_id() -> None:
    store = InMemoryStore()

    with pytest.raises(KeyError, match="memory record not found"):
        store.delete("missing")
