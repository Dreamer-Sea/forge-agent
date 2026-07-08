from __future__ import annotations

from forge_agent.memory import (
    InMemoryStore,
    MemoryScope,
    MemoryType,
    MemoryWriter,
)


def test_memory_writer_persists_explicit_user_memory() -> None:
    store = InMemoryStore()
    writer = MemoryWriter(store)

    decision = writer.maybe_write_from_user_input(
        "记住：我的项目默认使用 Python 3.13 和 uv",
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
    )

    records = store.list(scope=MemoryScope.PROJECT, scope_id="forge-agent")

    assert decision.should_write is True
    assert len(records) == 1
    assert records[0].type == MemoryType.SEMANTIC
    assert records[0].content == "我的项目默认使用 Python 3.13 和 uv"
    assert decision.metadata["record_id"] == records[0].id


def test_memory_writer_skips_temporary_input() -> None:
    store = InMemoryStore()
    writer = MemoryWriter(store)

    decision = writer.maybe_write_from_user_input("帮我继续下个阶段")

    assert decision.should_write is False
    assert store.list() == []


def test_memory_writer_persists_verification_failed_memory() -> None:
    store = InMemoryStore()
    writer = MemoryWriter(store)

    decision = writer.maybe_write_failure_memory(
        "JsonlMemoryStore 的 list 方法名遮蔽了内置 list 类型，导致 mypy valid-type 错误",
        failure_type="verification_failed",
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
    )

    records = store.list(type_filter=MemoryType.EPISODIC)

    assert decision.should_write is True
    assert len(records) == 1
    assert records[0].source == "verification_failed"
    assert "mypy valid-type" in records[0].content


def test_memory_writer_persists_successful_agent_summary() -> None:
    store = InMemoryStore()
    writer = MemoryWriter(store)

    decision = writer.maybe_write_from_agent_result(
        "完成阶段 2：实现 InMemoryStore 和 JsonlMemoryStore，并通过 pytest、mypy、ruff。",
        succeeded=True,
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
    )

    records = store.list(type_filter=MemoryType.SUMMARY)

    assert decision.should_write is True
    assert len(records) == 1
    assert records[0].type == MemoryType.SUMMARY
    assert records[0].source == "agent_result"


def test_memory_writer_skips_duplicate_memory() -> None:
    store = InMemoryStore()
    writer = MemoryWriter(store)

    first = writer.maybe_write_from_user_input(
        "记住：我的项目默认使用 Python 3.13 和 uv",
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
    )
    second = writer.maybe_write_from_user_input(
        "记住： 我的项目默认使用 Python 3.13 和 uv ",
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
    )

    assert first.should_write is True
    assert second.should_write is False
    assert second.reason.startswith("duplicate memory skipped")
    assert len(store.list()) == 1


def test_memory_writer_allows_same_content_in_different_scope_ids() -> None:
    store = InMemoryStore()
    writer = MemoryWriter(store)

    first = writer.maybe_write_from_user_input(
        "记住：我的项目默认使用 Python 3.13 和 uv",
        scope=MemoryScope.SESSION,
        scope_id="demo-a",
    )
    second = writer.maybe_write_from_user_input(
        "记住：我的项目默认使用 Python 3.13 和 uv",
        scope=MemoryScope.SESSION,
        scope_id="demo-b",
    )

    assert first.should_write is True
    assert second.should_write is True
    assert len(store.list(scope=MemoryScope.SESSION)) == 2
