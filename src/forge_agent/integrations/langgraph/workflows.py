"""LangGraph workflows for the runtime adapter."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from forge_agent.integrations.langgraph.state import LangGraphRoute, LangGraphState
from forge_agent.runtime.events import TraceEvent, TraceEventType
from forge_agent.security import PERMISSION_DENIED
from forge_agent.tools.base import ToolResult
from forge_agent.tools.registry import ToolRegistry

RAG_TRIGGER_PHRASES = (
    "knowledge base",
    "according to docs",
    "根据知识库",
)

SEARCH_KNOWLEDGE_BASE_TOOL = "search_knowledge_base"


def classify_task(state: LangGraphState) -> LangGraphState:
    """Classify whether the user task should use RAG."""

    user_input = state["user_input"]
    route: LangGraphRoute = "rag" if _needs_rag(user_input) else "direct"
    trace_events = _append_trace(
        state,
        event_type="workflow_node",
        node="classify_task",
        data={
            "route": route,
            "trigger_phrases": list(RAG_TRIGGER_PHRASES),
        },
    )
    trace_events.append(
        TraceEvent(
            event_type="workflow_route",
            step=len(trace_events),
            data={
                "runtime": "langgraph",
                "node": "classify_task",
                "route": route,
            },
        )
    )

    return {
        **state,
        "route": route,
        "steps": state.get("steps", 0) + 1,
        "trace_events": trace_events,
    }


def should_use_rag(state: LangGraphState) -> LangGraphRoute:
    """Route to RAG or direct execution based on classification result."""

    return state.get("route", "direct")


def make_retrieve_node(
    tool_registry: ToolRegistry | None,
) -> Any:
    """Create a retrieve node bound to the existing ToolRegistry."""

    def retrieve(state: LangGraphState) -> LangGraphState:
        """Retrieve context through the existing RAG tool when available."""

        query = _extract_rag_query(state["user_input"])
        trace_events = _append_trace(
            state,
            event_type="workflow_node",
            node="retrieve",
            data={
                "query": query,
                "tool_registry_available": tool_registry is not None,
            },
        )

        if tool_registry is None:
            retrieved_context = f"Mock knowledge base context for query: {query}"
            return {
                **state,
                "retrieved_context": retrieved_context,
                "steps": state.get("steps", 0) + 1,
                "trace_events": trace_events,
            }

        tool_call_id = "langgraph_search_knowledge_base"
        arguments = {
            "query": query,
            "top_k": 3,
        }

        trace_events.append(
            TraceEvent(
                event_type="tool_call",
                step=len(trace_events),
                data={
                    "runtime": "langgraph",
                    "node": "retrieve",
                    "tool_call_id": tool_call_id,
                    "tool_name": SEARCH_KNOWLEDGE_BASE_TOOL,
                    "arguments": arguments,
                },
            )
        )
        trace_events.append(
            TraceEvent(
                event_type="permission_check",
                step=len(trace_events),
                data={
                    "runtime": "langgraph",
                    "node": "retrieve",
                    "tool_call_id": tool_call_id,
                    "tool_name": SEARCH_KNOWLEDGE_BASE_TOOL,
                    "arguments": arguments,
                },
            )
        )

        tool_result = tool_registry.execute(
            name=SEARCH_KNOWLEDGE_BASE_TOOL,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )
        tool_results = [
            *state.get("tool_results", []),
            tool_result,
        ]

        trace_events.append(
            TraceEvent(
                event_type="tool_result",
                step=len(trace_events),
                data={
                    "runtime": "langgraph",
                    "node": "retrieve",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_result.tool_name,
                    "success": tool_result.success,
                    "error_code": tool_result.error_code,
                    "payload": tool_result.payload,
                    "safe_detail": tool_result.safe_detail,
                },
            )
        )

        if tool_result.error_code == PERMISSION_DENIED:
            trace_events.append(
                TraceEvent(
                    event_type="permission_denied",
                    step=len(trace_events),
                    data={
                        "runtime": "langgraph",
                        "node": "retrieve",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_result.tool_name,
                        "error_code": tool_result.error_code,
                        "reason": tool_result.payload.get("reason"),
                        "safe_detail": tool_result.safe_detail,
                    },
                )
            )

        retrieved_context = _context_from_tool_result(tool_result)

        return {
            **state,
            "retrieved_context": retrieved_context,
            "steps": state.get("steps", 0) + 1,
            "tool_results": tool_results,
            "trace_events": trace_events,
        }

    return retrieve


def answer_with_context(state: LangGraphState) -> LangGraphState:
    """Answer using retrieved context."""

    context = state.get("retrieved_context", "")
    trace_events = _append_trace(
        state,
        event_type="workflow_node",
        node="answer_with_context",
        data={
            "has_context": bool(context),
        },
    )
    trace_events.append(
        TraceEvent(
            event_type="final_answer",
            step=len(trace_events),
            data={
                "runtime": "langgraph",
                "route": "rag",
            },
        )
    )

    return {
        **state,
        "final_answer": f"Answered with context: {context}",
        "stopped_reason": "completed",
        "steps": state.get("steps", 0) + 1,
        "trace_events": trace_events,
    }


def direct_tool_agent(state: LangGraphState) -> LangGraphState:
    """Direct non-RAG path for the Day 4 routing workflow."""

    trace_events = _append_trace(
        state,
        event_type="workflow_node",
        node="direct_tool_agent",
        data={
            "route": "direct",
        },
    )
    trace_events.append(
        TraceEvent(
            event_type="final_answer",
            step=len(trace_events),
            data={
                "runtime": "langgraph",
                "route": "direct",
            },
        )
    )

    return {
        **state,
        "final_answer": f"Direct answer: {state['user_input']}",
        "stopped_reason": "completed",
        "steps": state.get("steps", 0) + 1,
        "trace_events": trace_events,
    }


def mock_node(state: LangGraphState) -> LangGraphState:
    """Return a deterministic response from a minimal LangGraph node."""

    trace_events = _append_trace(
        state,
        event_type="runtime_stop",
        node="mock_node",
        data={
            "reason": "completed",
        },
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


def build_rag_routing_workflow(
    tool_registry: ToolRegistry | None = None,
) -> Any:
    """Build the Day 4 task classification and RAG routing workflow."""

    graph = StateGraph(LangGraphState)
    graph.add_node("classify_task", classify_task)
    graph.add_node("retrieve", make_retrieve_node(tool_registry))
    graph.add_node("answer_with_context", answer_with_context)
    graph.add_node("direct_tool_agent", direct_tool_agent)

    graph.add_edge(START, "classify_task")
    graph.add_conditional_edges(
        "classify_task",
        should_use_rag,
        {
            "rag": "retrieve",
            "direct": "direct_tool_agent",
        },
    )
    graph.add_edge("retrieve", "answer_with_context")
    graph.add_edge("answer_with_context", END)
    graph.add_edge("direct_tool_agent", END)

    return graph.compile()


def _needs_rag(user_input: str) -> bool:
    normalized = user_input.casefold()
    return any(phrase.casefold() in normalized for phrase in RAG_TRIGGER_PHRASES)


def _extract_rag_query(user_input: str) -> str:
    query = user_input.strip()

    for prefix in (
        "根据知识库回答：",
        "根据知识库回答:",
        "根据知识库",
        "according to docs,",
        "according to docs:",
        "according to docs",
        "knowledge base:",
        "knowledge base",
    ):
        if query.casefold().startswith(prefix.casefold()):
            query = query[len(prefix) :].strip()
            break

    return query or user_input


def _context_from_tool_result(tool_result: ToolResult) -> str:
    if tool_result.success:
        context = tool_result.payload.get("context", "")
        return str(context)

    reason = tool_result.payload.get("reason") or tool_result.error_message
    return f"Tool {tool_result.tool_name} failed: {reason}"


def _append_trace(
    state: LangGraphState,
    *,
    event_type: TraceEventType,
    node: str,
    data: dict[str, Any],
) -> list[TraceEvent]:
    trace_events = list(state.get("trace_events", []))
    trace_events.append(
        TraceEvent(
            event_type=event_type,
            step=len(trace_events),
            data={
                "runtime": "langgraph",
                "node": node,
                **data,
            },
        )
    )
    return trace_events
