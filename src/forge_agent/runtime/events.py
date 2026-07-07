from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

TraceEventType = Literal[
    "model_call",
    "model_response",
    "tool_call",
    "permission_check",
    "tool_result",
    "permission_denied",
    "final_answer",
    "runtime_stop",
    "workflow_node",
    "workflow_route",
    "plan_created",
    "plan_step_started",
    "plan_step_completed",
    "plan_step_failed",
    "replan_triggered",
    "plan_completed",
    "reflection_started",
    "verification_result",
    "critique_generated",
    "revision_requested",
    "reflection_completed",
    "reflection_failed",
    "memory_recall_started",
    "memory_recall_result",
    "memory_write_attempted",
    "memory_write_skipped",
    "memory_write_completed",
]


class TraceEvent(BaseModel):
    """A minimal structured trace event recorded by the runtime."""

    event_type: TraceEventType
    step: int
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
