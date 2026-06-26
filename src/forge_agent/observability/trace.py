"""In-memory trace recorder."""

from __future__ import annotations

from typing import Any

from forge_agent.observability.events import TraceEvent, TraceEventType, new_run_id


class TraceRecorder:
    """Collect structured trace events for one agent run."""

    def __init__(self, run_id: str | None = None, case_id: str | None = None) -> None:
        self.run_id = run_id or new_run_id()
        self.case_id = case_id
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Return recorded events as an immutable tuple."""
        return tuple(self._events)

    def record(
        self,
        event_type: TraceEventType,
        *,
        name: str | None = None,
        step_index: int | None = None,
        payload: dict[str, Any] | None = None,
        case_id: str | None = None,
    ) -> TraceEvent:
        """Record one trace event."""
        event = TraceEvent(
            run_id=self.run_id,
            case_id=case_id if case_id is not None else self.case_id,
            event_type=event_type,
            name=name,
            step_index=step_index,
            payload=payload or {},
        )
        self._events.append(event)
        return event

    def record_model_call(
        self,
        *,
        name: str,
        step_index: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Record a model call event."""
        return self.record(
            "model_call",
            name=name,
            step_index=step_index,
            payload=payload,
        )

    def record_tool_call(
        self,
        *,
        name: str,
        step_index: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Record a tool call event."""
        return self.record(
            "tool_call",
            name=name,
            step_index=step_index,
            payload=payload,
        )

    def record_tool_result(
        self,
        *,
        name: str,
        step_index: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Record a tool result event."""
        return self.record(
            "tool_result",
            name=name,
            step_index=step_index,
            payload=payload,
        )

    def record_final_answer(
        self,
        *,
        answer: str,
        stopped_reason: str,
        step_index: int | None = None,
    ) -> TraceEvent:
        """Record the final answer event."""
        return self.record(
            "final_answer",
            step_index=step_index,
            payload={
                "answer": answer,
                "stopped_reason": stopped_reason,
            },
        )
