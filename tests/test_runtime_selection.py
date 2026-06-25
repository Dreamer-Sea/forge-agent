from __future__ import annotations

from forge_agent.providers.fake import FakeProvider
from forge_agent.runtime import (
    AgentRuntime,
    NativeAgentRuntime,
    RunConfig,
    RunResult,
    select_runtime_name,
)
from forge_agent.tools.registry import ToolRegistry


def test_native_runtime_implements_agent_runtime_protocol() -> None:
    runtime = NativeAgentRuntime(
        provider=FakeProvider(),
        tool_registry=ToolRegistry(),
    )

    assert isinstance(runtime, AgentRuntime)


def test_run_result_has_stable_fields() -> None:
    result = RunResult(
        final_answer="done",
        stopped_reason="completed",
        steps=1,
    )

    assert result.final_answer == "done"
    assert result.stopped_reason == "completed"
    assert result.steps == 1
    assert result.tool_results == []
    assert result.tool_calls == []
    assert result.trace_events == []
    assert result.error_message is None


def test_runtime_selection_defaults_to_native() -> None:
    config = RunConfig()

    assert config.runtime_name == "native"
    assert select_runtime_name() == "native"
    assert select_runtime_name("native") == "native"


def test_native_runtime_accepts_run_config_max_steps() -> None:
    runtime = NativeAgentRuntime(
        provider=FakeProvider(),
        tool_registry=ToolRegistry(),
        max_steps=5,
    )

    result = runtime.run(
        "Please answer directly",
        config=RunConfig(max_steps=1),
    )

    assert result.steps <= 1
