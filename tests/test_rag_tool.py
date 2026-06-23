from __future__ import annotations

from pathlib import Path

from forge_agent.providers.fake import FakeProvider
from forge_agent.rag.knowledge_base import KnowledgeBase
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.tools.defaults import create_default_tool_registry
from forge_agent.tools.rag_tool import SearchKnowledgeBaseTool


def write_knowledge_base(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    (path / "agent-runtime.md").write_text(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "Agent Runtime includes Agent Loop, Model Provider, Tool Registry, "
        "Tool Executor, Trace Recorder, and Settings.\n",
        encoding="utf-8",
    )

    (path / "security.md").write_text(
        "# Security\n\n"
        "## Permission System\n\n"
        "Permission System checks tool name, input arguments, workspace path, "
        "user approval, and runtime policy.\n",
        encoding="utf-8",
    )


def test_rag_tool_returns_grounded_results(tmp_path: Path) -> None:
    write_knowledge_base(tmp_path)

    knowledge_base = KnowledgeBase.from_directory(tmp_path)
    tool = SearchKnowledgeBaseTool(knowledge_base)

    result = tool.execute(
        arguments={"query": "Agent Runtime core modules", "top_k": 3},
        tool_call_id="call_search_knowledge_base",
    )

    assert result.success
    assert result.tool_name == "search_knowledge_base"
    assert "Agent Runtime includes Agent Loop" in result.payload["context"]
    assert result.payload["chunks"][0]["source"] == "agent-runtime.md"


def test_rag_tool_returns_citations(tmp_path: Path) -> None:
    write_knowledge_base(tmp_path)

    knowledge_base = KnowledgeBase.from_directory(tmp_path)
    tool = SearchKnowledgeBaseTool(knowledge_base)

    result = tool.execute(
        arguments={"query": "Permission System runtime policy", "top_k": 3},
        tool_call_id="call_search_knowledge_base",
    )

    assert result.success
    assert result.payload["citations"]
    assert result.payload["citations"][0]["source"] == "security.md"
    assert "[source: security.md#Security > Permission System" in (
        result.payload["citations"][0]["citation"]
    )


def test_agent_runtime_can_call_rag_tool(tmp_path: Path) -> None:
    write_knowledge_base(tmp_path)

    knowledge_base = KnowledgeBase.from_directory(tmp_path)
    registry = create_default_tool_registry(knowledge_base=knowledge_base)
    runtime = NativeAgentRuntime(
        provider=FakeProvider(),
        tool_registry=registry,
        max_steps=5,
    )

    result = runtime.run("根据知识库回答：Agent Runtime 有哪些核心模块？")

    assert result.stopped_reason == "completed"
    assert result.tool_results
    assert result.tool_results[0].tool_name == "search_knowledge_base"
    assert result.tool_results[0].success
    assert result.final_answer is not None
    assert "Agent Runtime includes Agent Loop" in result.final_answer
    assert "Citations:" in result.final_answer


def test_rag_trace_contains_query_and_sources(tmp_path: Path) -> None:
    write_knowledge_base(tmp_path)

    knowledge_base = KnowledgeBase.from_directory(tmp_path)
    registry = create_default_tool_registry(knowledge_base=knowledge_base)
    runtime = NativeAgentRuntime(
        provider=FakeProvider(),
        tool_registry=registry,
        max_steps=5,
    )

    result = runtime.run("根据知识库回答：Agent Runtime 有哪些核心模块？")

    tool_call_events = [
        event
        for event in result.trace_events
        if event.event_type == "tool_call"
        and event.data["tool_name"] == "search_knowledge_base"
    ]
    tool_result_events = [
        event
        for event in result.trace_events
        if event.event_type == "tool_result"
        and event.data["tool_name"] == "search_knowledge_base"
    ]

    assert tool_call_events
    assert tool_call_events[0].data["arguments"]["query"] == (
        "Agent Runtime 有哪些核心模块？"
    )

    assert tool_result_events
    payload = tool_result_events[0].data["payload"]
    assert payload["citations"][0]["source"] == "agent-runtime.md"
