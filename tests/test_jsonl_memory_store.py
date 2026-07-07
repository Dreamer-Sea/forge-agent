from __future__ import annotations

import pytest

from forge_agent.memory import JsonlMemoryStore, MemoryRecord, MemoryScope, MemoryType


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


def test_jsonl_memory_store_can_persist_and_reload_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)
    record = store.add(_record("项目默认使用 Python 3.13 和 uv"))

    reloaded_store = JsonlMemoryStore(tmp_path)

    assert reloaded_store.list() == [record]
    assert (tmp_path / "semantic.jsonl").exists()


def test_jsonl_memory_store_uses_type_specific_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)

    store.add(
        _record(
            "项目默认使用 Python 3.13 和 uv",
            memory_type=MemoryType.SEMANTIC,
        )
    )
    store.add(
        _record(
            "workspace guard 测试曾因 symlink case 缺失失败",
            memory_type=MemoryType.EPISODIC,
        )
    )
    store.add(
        _record(
            "本次任务完成了 memory store 实现",
            memory_type=MemoryType.SUMMARY,
        )
    )

    assert (tmp_path / "semantic.jsonl").exists()
    assert (tmp_path / "episodic.jsonl").exists()
    assert (tmp_path / "summary.jsonl").exists()


def test_jsonl_memory_store_search_returns_top_k_results(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)
    more_important = store.add(
        _record(
            "项目默认使用 Python 3.13 和 uv",
            importance=0.9,
            confidence=0.9,
        )
    )
    store.add(
        _record(
            "Python 3.13 是项目默认版本",
            importance=0.4,
            confidence=0.7,
        )
    )
    store.add(_record("workspace guard 会阻止路径逃逸"))

    results = store.search("Python 3.13 uv", scope=MemoryScope.PROJECT, top_k=1)

    assert len(results) == 1
    assert results[0].record == more_important


def test_jsonl_memory_store_isolates_session_scope_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)
    session_a = store.add(
        _record(
            "demo-a session 使用 Python 3.13",
            scope=MemoryScope.SESSION,
            scope_id="demo-a",
        )
    )
    store.add(
        _record(
            "demo-b session 使用 Python 3.12",
            scope=MemoryScope.SESSION,
            scope_id="demo-b",
        )
    )

    reloaded_store = JsonlMemoryStore(tmp_path)
    results = reloaded_store.search(
        "Python",
        scope=MemoryScope.SESSION,
        scope_id="demo-a",
    )

    assert [result.record for result in results] == [session_a]


def test_jsonl_memory_store_deduplicates_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)
    first = _record("项目默认使用 Python 3.13 和 uv")
    duplicate = _record("  项目默认使用 Python 3.13 和 uv  ")

    stored_first = store.add(first)
    stored_duplicate = store.add(duplicate)

    reloaded_store = JsonlMemoryStore(tmp_path)

    assert stored_duplicate == stored_first
    assert reloaded_store.list() == [stored_first]
    assert len((tmp_path / "semantic.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_memory_store_can_update_record_and_rewrite_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)
    record = store.add(_record("项目默认使用 Python 3.12"))

    updated = store.update(record.id, {"content": "项目默认使用 Python 3.13 和 uv"})

    reloaded_store = JsonlMemoryStore(tmp_path)

    assert updated.content == "项目默认使用 Python 3.13 和 uv"
    assert reloaded_store.list() == [updated]
    assert "Python 3.12" not in (tmp_path / "semantic.jsonl").read_text(encoding="utf-8")


def test_jsonl_memory_store_can_move_record_between_type_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)
    record = store.add(
        _record(
            "本次任务完成了 memory store 实现",
            memory_type=MemoryType.SUMMARY,
        )
    )

    updated = store.update(record.id, {"type": MemoryType.EPISODIC})

    reloaded_store = JsonlMemoryStore(tmp_path)

    assert updated.type == MemoryType.EPISODIC
    assert reloaded_store.list(type_filter=MemoryType.EPISODIC) == [updated]
    assert not (tmp_path / "summary.jsonl").exists()
    assert (tmp_path / "episodic.jsonl").exists()


def test_jsonl_memory_store_can_delete_record_and_rewrite_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)
    record = store.add(_record("项目默认使用 Python 3.13 和 uv"))

    store.delete(record.id)

    reloaded_store = JsonlMemoryStore(tmp_path)

    assert reloaded_store.list() == []
    assert not (tmp_path / "semantic.jsonl").exists()


def test_jsonl_memory_store_raises_for_missing_update_id(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = JsonlMemoryStore(tmp_path)

    with pytest.raises(KeyError, match="memory record not found"):
        store.update("missing", {"content": "new content"})
