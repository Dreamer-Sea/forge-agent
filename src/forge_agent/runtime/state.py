from __future__ import annotations

from pydantic import BaseModel, Field

from forge_agent.providers.base import ModelMessage
from forge_agent.runtime.events import TraceEvent
from forge_agent.tools.base import ToolResult


class AgentState(BaseModel):
    """Mutable state for one agent run."""

    user_input: str
    step_index: int = 0
    messages: list[ModelMessage] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)