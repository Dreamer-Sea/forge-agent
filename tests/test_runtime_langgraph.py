from __future__ import annotations

from forge_agent.integrations.langgraph import LangGraphAgentRuntime
from forge_agent.runtime import AgentRuntime, RunConfig, RunResult


def test_langgraph_runtime_runs_simple_workflow() -> None:
    runtime = LangGraphAgentRuntime()

    result = runtime.run(
        "hello langgraph",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert result.final_answer == "LangGraph runtime completed."
    assert result.stopped_reason == "completed"
    assert result.steps == 1


def test_langgraph_runtime_returns_run_result() -> None:
    runtime = LangGraphAgentRuntime()

    result = runtime.run(
        "hello langgraph",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert isinstance(result, RunResult)


def test_langgraph_runtime_implements_agent_runtime_protocol() -> None:
    runtime = LangGraphAgentRuntime()

    assert isinstance(runtime, AgentRuntime)


def test_langgraph_runtime_emits_minimal_trace_event() -> None:
    runtime = LangGraphAgentRuntime()

    result = runtime.run(
        "hello langgraph",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert [event.event_type for event in result.trace_events] == ["runtime_stop"]
    assert result.trace_events[0].data["runtime"] == "langgraph"
    assert result.trace_events[0].data["node"] == "mock_node"
