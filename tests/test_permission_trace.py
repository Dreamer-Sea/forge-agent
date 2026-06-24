from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge_agent.providers.base import (
    ModelMessage,
    ProviderResponse,
    ToolCall,
)
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.security import (
    PATH_OUTSIDE_WORKSPACE,
    PATH_TRAVERSAL_BLOCKED,
    PERMISSION_DENIED,
    Workspace,
)
from forge_agent.tools.file_tools import ReadFileTool, WriteFileTool
from forge_agent.tools.registry import ToolRegistry


class OneToolCallProvider:
    """Provider that emits one tool call and then returns a final answer."""

    def __init__(self, *, tool_name: str, arguments: dict[str, Any]) -> None:
        self._tool_name = tool_name
        self._arguments = arguments

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        if any(message.role == "tool" for message in messages):
            return ProviderResponse(final_answer="done")

        return ProviderResponse(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name=self._tool_name,
                    arguments=self._arguments,
                )
            ]
        )


def test_permission_check_is_recorded_in_trace(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# forge-agent\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace=Workspace(tmp_path)))

    runtime = NativeAgentRuntime(
        provider=OneToolCallProvider(
            tool_name="read_file",
            arguments={"path": "README.md"},
        ),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("read README")

    permission_check_events = [
        event
        for event in result.trace_events
        if event.event_type == "permission_check"
    ]

    assert permission_check_events
    assert permission_check_events[0].data == {
        "tool_call_id": "call_1",
        "tool_name": "read_file",
        "arguments": {"path": "README.md"},
    }


def test_permission_denied_is_recorded_in_trace(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(WriteFileTool(workspace=Workspace(tmp_path)))

    runtime = NativeAgentRuntime(
        provider=OneToolCallProvider(
            tool_name="write_file",
            arguments={
                "path": "notes.md",
                "content": "hello\n",
            },
        ),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("write notes")

    permission_denied_events = [
        event
        for event in result.trace_events
        if event.event_type == "permission_denied"
    ]

    assert permission_denied_events
    assert permission_denied_events[0].data == {
        "tool_call_id": "call_1",
        "tool_name": "write_file",
        "error_code": PERMISSION_DENIED,
        "reason": "Write actions require explicit permission.",
        "safe_detail": {"path": "notes.md"},
    }


def test_path_traversal_denial_is_recorded_in_trace(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace=Workspace(tmp_path)))

    runtime = NativeAgentRuntime(
        provider=OneToolCallProvider(
            tool_name="read_file",
            arguments={"path": "../secret.txt"},
        ),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("try to read ../secret.txt")

    permission_denied_events = [
        event
        for event in result.trace_events
        if event.event_type == "permission_denied"
    ]

    assert permission_denied_events
    assert permission_denied_events[0].data["tool_name"] == "read_file"
    assert permission_denied_events[0].data["error_code"] == PATH_TRAVERSAL_BLOCKED
    assert permission_denied_events[0].data["safe_detail"] == {
        "path": "../secret.txt"
    }


def test_trace_does_not_expose_sensitive_absolute_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace=Workspace(workspace_root)))

    runtime = NativeAgentRuntime(
        provider=OneToolCallProvider(
            tool_name="read_file",
            arguments={"path": str(outside_file)},
        ),
        tool_registry=registry,
        max_steps=3,
    )

    result = runtime.run("try to read outside file")

    trace_json = json.dumps(
        [event.model_dump(mode="json") for event in result.trace_events],
        ensure_ascii=False,
    )

    assert str(outside_file) not in trace_json
    assert "<redacted-path>" in trace_json
    assert "<outside-workspace>" in trace_json

    permission_denied_events = [
        event
        for event in result.trace_events
        if event.event_type == "permission_denied"
    ]

    assert permission_denied_events
    assert permission_denied_events[0].data["error_code"] == PATH_OUTSIDE_WORKSPACE
    assert permission_denied_events[0].data["safe_detail"] == {
        "path": "<outside-workspace>"
    }