from __future__ import annotations

from typing import Any

from forge_agent.memory import InMemoryStore, MemoryRecord, MemoryScope, MemoryType
from forge_agent.providers.base import ModelMessage, ProviderResponse
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.tools.registry import ToolRegistry


class FinalAnswerProvider:
    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        return ProviderResponse(final_answer="done")


def test_runtime_records_memory_recall_trace_events() -> None:
    store = InMemoryStore()
    store.add(
        MemoryRecord(
            type=MemoryType.SEMANTIC,
            scope=MemoryScope.PROJECT,
            scope_id="forge-agent",
            content="项目默认使用 Python 3.13 和 uv",
            source="test",
        )
    )
    runtime = NativeAgentRuntime(
        provider=FinalAnswerProvider(),
        tool_registry=ToolRegistry(),
        memory_store=store,
        memory_scope=MemoryScope.PROJECT,
        memory_scope_id="forge-agent",
    )

    result = runtime.run("我的项目默认使用什么 Python 版本？")
    event_types = [event.event_type for event in result.trace_events]

    assert "memory_recall_started" in event_types
    assert "memory_recall_result" in event_types

    recall_result = next(
        event for event in result.trace_events if event.event_type == "memory_recall_result"
    )
    assert recall_result.data["result_count"] == 1
    assert recall_result.data["memory_count"] == 1
    assert recall_result.data["records"][0]["scope_id"] == "forge-agent"


def test_runtime_records_memory_write_completed_for_explicit_memory() -> None:
    store = InMemoryStore()
    runtime = NativeAgentRuntime(
        provider=FinalAnswerProvider(),
        tool_registry=ToolRegistry(),
        memory_store=store,
        memory_scope=MemoryScope.PROJECT,
        memory_scope_id="forge-agent",
    )

    result = runtime.run("记住：我的项目默认使用 Python 3.13 和 uv")
    event_types = [event.event_type for event in result.trace_events]

    assert "memory_write_attempted" in event_types
    assert "memory_write_completed" in event_types

    completed = next(
        event
        for event in result.trace_events
        if event.event_type == "memory_write_completed"
    )
    assert completed.data["source"] == "user_input"
    assert completed.data["metadata"]["record_id"]


def test_runtime_records_memory_write_skipped_for_temporary_input() -> None:
    runtime = NativeAgentRuntime(
        provider=FinalAnswerProvider(),
        tool_registry=ToolRegistry(),
        memory_store=InMemoryStore(),
        memory_scope=MemoryScope.PROJECT,
        memory_scope_id="forge-agent",
    )

    result = runtime.run("帮我继续下个阶段")
    event_types = [event.event_type for event in result.trace_events]

    assert "memory_write_attempted" in event_types
    assert "memory_write_skipped" in event_types

    skipped = next(
        event
        for event in result.trace_events
        if event.event_type == "memory_write_skipped"
    )
    assert skipped.data["source"] == "user_input"
    assert skipped.data["reason"] == (
        "user input does not contain an explicit memory request"
    )
