"""State objects used by the LangGraph runtime adapter."""

from __future__ import annotations

from typing import TypedDict

from forge_agent.runtime.events import TraceEvent


class LangGraphState(TypedDict, total=False):
    """Mutable state passed between LangGraph workflow nodes."""

    user_input: str
    final_answer: str | None
    stopped_reason: str
    steps: int
    trace_events: list[TraceEvent]
