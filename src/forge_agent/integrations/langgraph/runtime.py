"""LangGraph runtime adapter."""

from __future__ import annotations

from forge_agent.integrations.langgraph.workflows import build_rag_routing_workflow
from forge_agent.runtime.base import RunConfig, RunResult
from forge_agent.tools.registry import ToolRegistry


class LangGraphAgentRuntime:
    """Runtime adapter backed by a LangGraph workflow."""

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._workflow = build_rag_routing_workflow(tool_registry=tool_registry)

    def run(
        self,
        user_input: str,
        config: RunConfig | None = None,
    ) -> RunResult:
        runtime_config = config or RunConfig(runtime_name="langgraph")
        state = self._workflow.invoke(
            {
                "user_input": user_input,
                "final_answer": None,
                "stopped_reason": "completed",
                "steps": 0,
                "tool_results": [],
                "trace_events": [],
            }
        )

        return RunResult(
            final_answer=state.get("final_answer"),
            stopped_reason=state.get("stopped_reason", "completed"),
            steps=state.get("steps", runtime_config.max_steps),
            tool_results=state.get("tool_results", []),
            trace_events=state.get("trace_events", []),
        )
