from __future__ import annotations

from typing import Any

from forge_agent.memory import InMemoryStore, MemoryRecord, MemoryScope, MemoryType
from forge_agent.providers.base import ModelMessage, ProviderResponse
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.tools.registry import ToolRegistry


class RecordingProvider:
    def __init__(self, final_answer: str = "runtime completed") -> None:
        self.final_answer = final_answer
        self.messages: list[ModelMessage] = []

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        self.messages = list(messages)
        return ProviderResponse(final_answer=self.final_answer)


def test_runtime_recalls_memory_and_injects_context_into_model_messages() -> None:
    store = InMemoryStore()
    store.add(
        MemoryRecord(
            type=MemoryType.SEMANTIC,
            scope=MemoryScope.PROJECT,
            scope_id="forge-agent",
            content="项目默认使用 Python 3.13 和 uv",
            source="test",
            importance=0.9,
            confidence=0.9,
        )
    )
    provider = RecordingProvider()
    runtime = NativeAgentRuntime(
        provider=provider,
        tool_registry=ToolRegistry(),
        memory_store=store,
        memory_scope=MemoryScope.PROJECT,
        memory_scope_id="forge-agent",
    )

    result = runtime.run("我的项目默认使用什么 Python 版本？")

    assert result.stopped_reason == "completed"
    assert provider.messages[0].role == "system"
    assert "## Long-term Memory" in provider.messages[0].content
    assert "项目默认使用 Python 3.13 和 uv" in provider.messages[0].content
    assert provider.messages[1].role == "user"
    assert provider.messages[1].content == "我的项目默认使用什么 Python 版本？"


def test_runtime_writes_explicit_user_memory_after_run() -> None:
    store = InMemoryStore()
    runtime = NativeAgentRuntime(
        provider=RecordingProvider(),
        tool_registry=ToolRegistry(),
        memory_store=store,
        memory_scope=MemoryScope.PROJECT,
        memory_scope_id="forge-agent",
    )

    result = runtime.run("记住：我的项目默认使用 Python 3.13 和 uv")

    records = store.list(
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
        type_filter=MemoryType.SEMANTIC,
    )

    assert result.stopped_reason == "completed"
    assert len(records) == 1
    assert records[0].content == "我的项目默认使用 Python 3.13 和 uv"
    assert records[0].source == "user_input"


def test_runtime_without_memory_store_keeps_plain_user_message_flow() -> None:
    provider = RecordingProvider()
    runtime = NativeAgentRuntime(
        provider=provider,
        tool_registry=ToolRegistry(),
    )

    result = runtime.run("hello")

    assert result.stopped_reason == "completed"
    assert len(provider.messages) == 1
    assert provider.messages[0] == ModelMessage(role="user", content="hello")
    assert not any(
        event.event_type.startswith("memory_")
        for event in result.trace_events
    )


def test_runtime_respects_memory_scope_id_isolation() -> None:
    store = InMemoryStore()
    store.add(
        MemoryRecord(
            type=MemoryType.SEMANTIC,
            scope=MemoryScope.SESSION,
            scope_id="session-a",
            content="session-a 默认使用 Python 3.13",
            source="test",
        )
    )
    store.add(
        MemoryRecord(
            type=MemoryType.SEMANTIC,
            scope=MemoryScope.SESSION,
            scope_id="session-b",
            content="session-b 默认使用 Python 3.12",
            source="test",
        )
    )
    provider = RecordingProvider()
    runtime = NativeAgentRuntime(
        provider=provider,
        tool_registry=ToolRegistry(),
        memory_store=store,
        memory_scope=MemoryScope.SESSION,
        memory_scope_id="session-a",
    )

    runtime.run("默认使用什么 Python 版本？")

    assert "session-a 默认使用 Python 3.13" in provider.messages[0].content
    assert "session-b 默认使用 Python 3.12" not in provider.messages[0].content
