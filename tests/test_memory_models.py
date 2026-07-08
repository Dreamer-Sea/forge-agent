from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge_agent.memory import (
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
    MemoryWriteDecision,
)


def test_memory_record_requires_structured_fields() -> None:
    record = MemoryRecord(
        type=MemoryType.SEMANTIC,
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
        content="项目默认使用 Python 3.13 和 uv",
        source="user_input",
        importance=0.8,
        confidence=0.9,
        metadata={"topic": "runtime"},
    )

    assert record.id.startswith("mem_")
    assert record.type == MemoryType.SEMANTIC
    assert record.scope == MemoryScope.PROJECT
    assert record.scope_id == "forge-agent"
    assert record.content == "项目默认使用 Python 3.13 和 uv"
    assert record.created_at.tzinfo is not None
    assert record.updated_at.tzinfo is not None
    assert record.belongs_to_scope(scope=MemoryScope.PROJECT, scope_id="forge-agent")


def test_memory_record_can_round_trip_as_json() -> None:
    record = MemoryRecord(
        type=MemoryType.EPISODIC,
        scope=MemoryScope.SESSION,
        scope_id="demo",
        content="workspace guard 测试曾因 symlink case 缺失失败",
        source="verification_failed",
    )

    loaded = MemoryRecord.model_validate_json(record.model_dump_json())

    assert loaded == record


def test_memory_record_rejects_blank_content() -> None:
    with pytest.raises(ValidationError, match="memory content must not be blank"):
        MemoryRecord(
            type=MemoryType.SUMMARY,
            scope=MemoryScope.SESSION,
            content="   ",
            source="test",
        )


def test_memory_record_rejects_invalid_scores() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(
            type=MemoryType.SEMANTIC,
            scope=MemoryScope.PROJECT,
            content="valid content",
            source="test",
            importance=1.2,
        )


def test_memory_search_result_records_score_rank_and_reason() -> None:
    record = MemoryRecord(
        type=MemoryType.SEMANTIC,
        scope=MemoryScope.GLOBAL,
        content="全局公共记忆",
        source="test",
    )

    result = MemorySearchResult(
        record=record,
        score=0.75,
        rank=1,
        reason="keyword overlap matched: 全局",
    )

    assert result.record == record
    assert result.score == 0.75
    assert result.rank == 1
    assert "keyword overlap" in result.reason


def test_write_decision_can_create_memory_record() -> None:
    decision = MemoryWriteDecision(
        should_write=True,
        reason="explicit remember request",
        content="项目默认使用 Python 3.13 和 uv",
        memory_type=MemoryType.SEMANTIC,
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
        importance=0.9,
        confidence=0.95,
        source="user_input",
    )

    record = decision.to_record()

    assert record.type == MemoryType.SEMANTIC
    assert record.scope == MemoryScope.PROJECT
    assert record.scope_id == "forge-agent"
    assert record.content == "项目默认使用 Python 3.13 和 uv"
    assert record.importance == 0.9
    assert record.confidence == 0.95


def test_skipped_write_decision_cannot_create_record() -> None:
    decision = MemoryWriteDecision(
        should_write=False,
        reason="temporary context should not be stored",
    )

    with pytest.raises(ValueError, match="cannot convert a skipped"):
        decision.to_record()
