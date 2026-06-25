from __future__ import annotations

from forge_agent.integrations.langgraph import LangGraphAgentRuntime
from forge_agent.runtime import AgentRuntime, RunConfig, RunResult


def test_langgraph_runtime_routes_to_rag_for_chinese_trigger() -> None:
    runtime = LangGraphAgentRuntime()

    result = runtime.run(
        "根据知识库回答 Permission system 如何工作",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert result.stopped_reason == "completed"
    assert result.final_answer is not None
    assert result.final_answer.startswith("Answered with context:")
    assert _trace_node_names(result) == [
        "classify_task",
        "retrieve",
        "answer_with_context",
    ]


def test_langgraph_runtime_routes_to_rag_for_english_trigger() -> None:
    runtime = LangGraphAgentRuntime()

    result = runtime.run(
        "according to docs, explain the permission system",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert result.final_answer is not None
    assert result.final_answer.startswith("Answered with context:")
    assert "retrieve" in _trace_node_names(result)


def test_langgraph_runtime_routes_to_direct_agent() -> None:
    runtime = LangGraphAgentRuntime()

    result = runtime.run(
        "echo hello",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert result.stopped_reason == "completed"
    assert result.final_answer == "Direct answer: echo hello"
    assert _trace_node_names(result) == [
        "classify_task",
        "direct_tool_agent",
    ]


def test_langgraph_runtime_preserves_state() -> None:
    runtime = LangGraphAgentRuntime()

    result = runtime.run(
        "knowledge base: explain ToolRegistry",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert result.steps == 3
    assert result.trace_events[0].data["node"] == "classify_task"
    assert result.trace_events[1].event_type == "workflow_route"
    assert result.trace_events[1].data["route"] == "rag"


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


def _trace_node_names(result: RunResult) -> list[str]:
    return [
        str(event.data["node"])
        for event in result.trace_events
        if event.event_type == "workflow_node"
    ]
