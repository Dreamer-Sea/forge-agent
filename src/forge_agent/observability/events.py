"""Trace event schema for agent observability."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

TraceEventType = Literal[
    "model_call",
    "tool_call",
    "tool_result",
    "permission_check",
    "permission_denied",
    "workflow_node",
    "final_answer",
]


def new_run_id() -> str:
    """Create a unique run identifier."""
    return str(uuid4())


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


class TraceEvent(BaseModel):
    """A single structured event emitted during an agent run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    event_type: TraceEventType
    timestamp: str = Field(default_factory=utc_now_iso)
    case_id: str | None = None
    step_index: int | None = Field(default=None, ge=0)
    name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
