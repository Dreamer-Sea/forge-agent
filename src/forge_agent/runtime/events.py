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
]


class TraceEvent(BaseModel):
    """A minimal structured trace event recorded by the runtime."""

    event_type: TraceEventType
    step: int
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )