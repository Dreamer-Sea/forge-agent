"""LangGraph workflows for the runtime adapter."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from forge_agent.integrations.langgraph.state import LangGraphState
from forge_agent.runtime.events import TraceEvent


def mock_node(state: LangGraphState) -> LangGraphState:
    """Return a deterministic response from a minimal LangGraph node."""

    trace_events = list(state.get("trace_events", []))
    trace_events.append(
        TraceEvent(
            event_type="runtime_stop",
            step=0,
            data={
                "runtime": "langgraph",
                "node": "mock_node",
                "reason": "completed",
            },
        )
    )

    return {
        **state,
        "final_answer": "LangGraph runtime completed.",
        "stopped_reason": "completed",
        "steps": 1,
        "trace_events": trace_events,
    }


def build_minimal_workflow() -> Any:
    """Build the minimal Day 4 LangGraph workflow."""

    graph = StateGraph(LangGraphState)
    graph.add_node("mock_node", mock_node)
    graph.add_edge(START, "mock_node")
    graph.add_edge("mock_node", END)
    return graph.compile()
