"""State objects used by the LangGraph runtime adapter."""

from __future__ import annotations

from typing import Literal, TypedDict

from forge_agent.runtime.events import TraceEvent

LangGraphRoute = Literal["rag", "direct"]


class LangGraphState(TypedDict, total=False):
    """Mutable state passed between LangGraph workflow nodes."""

    user_input: str
    route: LangGraphRoute
    retrieved_context: str
    final_answer: str | None
    stopped_reason: str
    steps: int
    trace_events: list[TraceEvent]
