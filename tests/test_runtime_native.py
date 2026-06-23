from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_agent.providers.base import ModelMessage, ProviderResponse, ToolCall
from forge_agent.providers.fake import FakeProvider
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.tools.echo import EchoTextTool
from forge_agent.tools.file_tools import ListFilesTool, ReadFileTool
from forge_agent.tools.registry import ToolRegistry


class FinalAnswerProvider:
    def complete(
            self,
            messages: list[ModelMessage],
            tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        return ProviderResponse(final_answer="Done.")


class LoopingProvider:
    def complete(
            self,
            messages: list[ModelMessage],
            tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        return ProviderResponse(
            content="Need another tool call.",
            tool_calls=[
                ToolCall(
                    id="call_list_files",
                    name="list_files",
                    arguments={},
                )
            ],
        )


def test_agent_runtime_executes_tool_call(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# forge-agent", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    registry = ToolRegistry()
    registry.register(ListFilesTool())
    registry.register(ReadFileTool())

    runtime = NativeAgentRuntime(
        provider=FakeProvider(),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("list files and read README")

    assert result.stopped_reason == "completed"
    assert result.final_answer is not None
    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "list_files",
        "read_file",
    ]


def test_agent_runtime_stops_when_completed() -> None:
    registry = ToolRegistry()
    runtime = NativeAgentRuntime(
        provider=FinalAnswerProvider(),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("say done")

    assert result.stopped_reason == "completed"
    assert result.final_answer == "Done."
    assert result.steps == 1


def test_agent_runtime_stops_at_max_steps() -> None:
    registry = ToolRegistry()
    registry.register(ListFilesTool())

    runtime = NativeAgentRuntime(
        provider=LoopingProvider(),
        tool_registry=registry,
        max_steps=1,
    )

    result = runtime.run("keep looping")

    assert result.stopped_reason == "max_steps"
    assert result.final_answer is None
    assert result.steps == 1


def test_agent_runtime_handles_tool_error() -> None:
    registry = ToolRegistry()
    registry.register(ListFilesTool())

    runtime = NativeAgentRuntime(
        provider=FakeProvider(),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("list files and read README")

    assert result.stopped_reason == "completed"
    assert any(
        tool_result.error_code == "unknown_tool"
        for tool_result in result.tool_results
    )


def test_agent_runtime_returns_structured_result() -> None:
    registry = ToolRegistry()
    runtime = NativeAgentRuntime(
        provider=FinalAnswerProvider(),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("say done")

    assert result.final_answer == "Done."
    assert result.stopped_reason == "completed"
    assert result.tool_results == []
    assert len(result.trace_events) >= 3


def test_trace_records_model_call_and_tool_result(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# forge-agent", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    registry = ToolRegistry()
    registry.register(ListFilesTool())
    registry.register(ReadFileTool())

    runtime = NativeAgentRuntime(
        provider=FakeProvider(),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("list files and read README")
    event_types = [event.event_type for event in result.trace_events]

    assert "model_call" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "final_answer" in event_types


class EchoProvider:
    def complete(
            self,
            messages: list[ModelMessage],
            tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        tool_observations = [message for message in messages if message.role == "tool"]

        if not tool_observations:
            return ProviderResponse(
                content="I need to echo text.",
                tool_calls=[
                    ToolCall(
                        id="call_echo",
                        name="echo_text",
                        arguments={"text": "hello runtime"},
                    )
                ],
            )

        return ProviderResponse(final_answer="Echo completed.")


def test_agent_runtime_executes_echo_text_tool() -> None:
    registry = ToolRegistry()
    registry.register(EchoTextTool())

    runtime = NativeAgentRuntime(
        provider=EchoProvider(),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("echo hello runtime")

    assert result.stopped_reason == "completed"
    assert result.final_answer == "Echo completed."
    assert len(result.tool_results) == 1
    assert result.tool_results[0].tool_name == "echo_text"
    assert result.tool_results[0].success is True
    assert result.tool_results[0].payload["text"] == "hello runtime"
