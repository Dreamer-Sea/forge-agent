from __future__ import annotations

from pathlib import Path

from forge_agent.integrations.langgraph import LangGraphAgentRuntime
from forge_agent.rag.knowledge_base import KnowledgeBase
from forge_agent.runtime import RunConfig
from forge_agent.security import PermissionAction, PermissionDecision, PermissionPolicy
from forge_agent.tools.defaults import create_default_tool_registry


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


def test_langgraph_runtime_uses_existing_tool_registry(tmp_path: Path) -> None:
    write_knowledge_base(tmp_path)
    registry = create_default_tool_registry(
        knowledge_base=KnowledgeBase.from_directory(tmp_path),
        safe_knowledge_base_path="knowledge_base",
    )
    runtime = LangGraphAgentRuntime(tool_registry=registry)

    result = runtime.run(
        "根据知识库回答：Agent Runtime 有哪些核心模块？",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert result.stopped_reason == "completed"
    assert result.tool_results
    assert result.tool_results[0].tool_name == "search_knowledge_base"
    assert result.tool_results[0].success is True
    assert result.final_answer is not None
    assert "Agent Runtime includes Agent Loop" in result.final_answer


def test_langgraph_runtime_applies_permission_policy(tmp_path: Path) -> None:
    class DenySearchPolicy(PermissionPolicy):
        def check(self, action: PermissionAction) -> PermissionDecision:
            if action is PermissionAction.SEARCH_KB:
                return PermissionDecision.deny(
                    action=action,
                    reason="Knowledge base search is disabled for this test.",
                    policy_name="deny-search-test",
                )
            return super().check(action)

    write_knowledge_base(tmp_path)
    registry = create_default_tool_registry(
        knowledge_base=KnowledgeBase.from_directory(tmp_path),
        permission_policy=DenySearchPolicy(),
        safe_knowledge_base_path="knowledge_base",
    )
    runtime = LangGraphAgentRuntime(tool_registry=registry)

    result = runtime.run(
        "根据知识库回答：Permission System 如何工作？",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert result.tool_results
    assert result.tool_results[0].success is False
    assert result.tool_results[0].error_code == "PERMISSION_DENIED"
    assert result.tool_results[0].safe_detail == {
        "action": "search_kb",
        "index_path": "knowledge_base",
    }

    permission_denied_events = [
        event
        for event in result.trace_events
        if event.event_type == "permission_denied"
    ]
    assert permission_denied_events
    assert permission_denied_events[0].data["reason"] == (
        "Knowledge base search is disabled for this test."
    )


def test_langgraph_runtime_emits_trace_events(tmp_path: Path) -> None:
    write_knowledge_base(tmp_path)
    registry = create_default_tool_registry(
        knowledge_base=KnowledgeBase.from_directory(tmp_path),
        safe_knowledge_base_path="knowledge_base",
    )
    runtime = LangGraphAgentRuntime(tool_registry=registry)

    result = runtime.run(
        "knowledge base: Permission System runtime policy",
        config=RunConfig(runtime_name="langgraph"),
    )

    event_types = [event.event_type for event in result.trace_events]

    assert "workflow_node" in event_types
    assert "workflow_route" in event_types
    assert "tool_call" in event_types
    assert "permission_check" in event_types
    assert "tool_result" in event_types
    assert "final_answer" in event_types


def test_langgraph_runtime_returns_same_result_shape_as_native(
    tmp_path: Path,
) -> None:
    write_knowledge_base(tmp_path)
    registry = create_default_tool_registry(
        knowledge_base=KnowledgeBase.from_directory(tmp_path),
        safe_knowledge_base_path="knowledge_base",
    )
    runtime = LangGraphAgentRuntime(tool_registry=registry)

    result = runtime.run(
        "according to docs, explain the permission system",
        config=RunConfig(runtime_name="langgraph"),
    )

    assert hasattr(result, "final_answer")
    assert hasattr(result, "stopped_reason")
    assert hasattr(result, "steps")
    assert hasattr(result, "tool_results")
    assert hasattr(result, "tool_calls")
    assert hasattr(result, "trace_events")
    assert result.tool_calls == result.tool_results
