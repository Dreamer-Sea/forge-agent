from __future__ import annotations

import pytest

from forge_agent.context import ContextComposer, enforce_context_budget
from forge_agent.memory import MemoryRecord, MemoryScope, MemorySearchResult, MemoryType


def _memory_result(
    content: str,
    *,
    rank: int = 1,
    score: float = 1.0,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    scope: MemoryScope = MemoryScope.PROJECT,
    scope_id: str | None = "forge-agent",
) -> MemorySearchResult:
    return MemorySearchResult(
        record=MemoryRecord(
            type=memory_type,
            scope=scope,
            scope_id=scope_id,
            content=content,
            source="test",
            importance=0.8,
            confidence=0.9,
        ),
        score=score,
        rank=rank,
        reason="test memory",
    )


def test_enforce_context_budget_keeps_short_text() -> None:
    result = enforce_context_budget("short context", max_chars=100)

    assert result.text == "short context"
    assert result.original_chars == len("short context")
    assert result.final_chars == len("short context")
    assert result.truncated is False


def test_enforce_context_budget_truncates_long_text() -> None:
    result = enforce_context_budget("a" * 100, max_chars=40)

    assert result.truncated is True
    assert result.original_chars == 100
    assert result.final_chars <= 40
    assert result.text.endswith("[context truncated]")


def test_enforce_context_budget_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="max_chars must be greater than 0"):
        enforce_context_budget("context", max_chars=0)


def test_context_composer_combines_memory_rag_and_user_input() -> None:
    composer = ContextComposer(max_context_chars=1_000)
    memory = [
        _memory_result(
            "项目默认使用 Python 3.13 和 uv",
            rank=1,
            score=0.9,
        )
    ]

    result = composer.compose(
        user_input="我的项目默认使用什么 Python 版本？",
        memory_results=memory,
        rag_context="forge-agent 是一个 Agent Platform runtime。",
    )

    assert len(result.messages) == 2
    assert result.messages[0].role == "system"
    assert "## Long-term Memory" in result.messages[0].content
    assert "项目默认使用 Python 3.13 和 uv" in result.messages[0].content
    assert "## Knowledge Base Context" in result.messages[0].content
    assert "Agent Platform runtime" in result.messages[0].content
    assert result.messages[1].role == "user"
    assert result.messages[1].content == "我的项目默认使用什么 Python 版本？"
    assert result.memory_count == 1
    assert result.truncated is False


def test_context_composer_can_emit_only_user_message_without_context() -> None:
    composer = ContextComposer()

    result = composer.compose(user_input="继续下个阶段")

    assert len(result.messages) == 1
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "继续下个阶段"
    assert result.memory_context == ""
    assert result.rag_context is None
    assert result.memory_count == 0


def test_context_composer_deduplicates_memory_context() -> None:
    composer = ContextComposer()
    memory = [
        _memory_result("项目默认使用 Python 3.13 和 uv", rank=1),
        _memory_result(" 项目默认使用 Python 3.13 和 uv ", rank=2),
    ]

    result = composer.compose(
        user_input="我的项目默认使用什么 Python 版本？",
        memory_results=memory,
    )

    assert result.memory_count == 1
    assert result.messages[0].content.count("项目默认使用 Python 3.13 和 uv") == 1


def test_context_composer_applies_context_budget() -> None:
    composer = ContextComposer(max_context_chars=120)
    memory = [
        _memory_result(
            "项目默认使用 Python 3.13 和 uv。" * 20,
            rank=1,
            score=0.9,
        )
    ]

    result = composer.compose(
        user_input="我的项目默认使用什么 Python 版本？",
        memory_results=memory,
    )

    assert result.truncated is True
    assert result.final_context_chars <= 120
    assert "[context truncated]" in result.messages[0].content


def test_context_composer_rejects_blank_user_input() -> None:
    composer = ContextComposer()

    with pytest.raises(ValueError, match="user_input must not be blank"):
        composer.compose(user_input="   ")


def test_context_composer_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="max_context_chars must be greater than 0"):
        ContextComposer(max_context_chars=0)
