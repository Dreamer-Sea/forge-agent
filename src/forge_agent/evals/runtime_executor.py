"""Runtime adapter for eval execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

from forge_agent.evals.dataset import EvalCase
from forge_agent.evals.runner import EvalRunOutput
from forge_agent.observability import TraceRecorder
from forge_agent.observability.events import TraceEventType


class ToolResultLike(Protocol):
    """Protocol for tool results emitted by runtime implementations."""

    tool_name: str
    success: bool


class RuntimeTraceEventLike(Protocol):
    """Protocol for runtime trace events consumed by eval."""

    event_type: str
    step: int
    data: dict[str, Any]


class VerificationResultLike(Protocol):
    """Protocol for verification results emitted by reflection attempts."""

    passed: bool
    reasons: list[str]
    unsupported_claims: list[str]
    missing_evidence: list[str]


class ReflectionAttemptLike(Protocol):
    """Protocol for reflection attempts emitted by reflection runtime."""

    verification_result: VerificationResultLike


class RuntimeResultLike(Protocol):
    """Protocol for runtime results consumed by eval."""

    final_answer: str | None
    stopped_reason: str
    tool_results: Sequence[ToolResultLike]
    trace_events: Sequence[RuntimeTraceEventLike]


class RuntimeLike(Protocol):
    """Protocol for runtime implementations used by eval."""

    def run(self, task: str) -> RuntimeResultLike:
        """Run one task and return a runtime result."""
        ...


class RuntimeEvalExecutor:
    """Adapt an existing agent runtime to EvalCaseExecutor."""

    def __init__(
        self,
        *,
        runtime: object,
        runtime_name: str,
    ) -> None:
        self.runtime = runtime
        self.runtime_name = runtime_name

    async def run_case(
        self,
        case: EvalCase,
        recorder: TraceRecorder,
    ) -> EvalRunOutput:
        """Run one eval case through the configured runtime."""
        runtime = cast(RuntimeLike, self.runtime)

        recorder.record_model_call(
            name=self.runtime_name,
            payload={"input": case.input},
        )

        result = runtime.run(case.input)

        for tool_result in result.tool_results:
            recorder.record_tool_call(
                name=tool_result.tool_name,
                payload={"tool_name": tool_result.tool_name},
            )
            recorder.record_tool_result(
                name=tool_result.tool_name,
                payload={
                    "tool_name": tool_result.tool_name,
                    "success": tool_result.success,
                },
            )

        self._record_runtime_reflection_trace(result, recorder)
        reflection_attempts = self._reflection_attempts(result)
        verification_result = self._last_verification_result(reflection_attempts)

        return EvalRunOutput(
            final_answer=result.final_answer or "",
            stopped_reason=result.stopped_reason,
            sources=[],
            reflection_attempts=len(reflection_attempts),
            verification_passed=(
                verification_result.passed if verification_result is not None else None
            ),
            verification_reasons=(
                verification_result.reasons if verification_result is not None else []
            ),
            unsupported_claims=(
                verification_result.unsupported_claims
                if verification_result is not None
                else []
            ),
            missing_evidence=(
                verification_result.missing_evidence
                if verification_result is not None
                else []
            ),
        )

    @staticmethod
    def _reflection_attempts(result: RuntimeResultLike) -> list[ReflectionAttemptLike]:
        attempts = getattr(result, "reflection_attempts", [])
        if not isinstance(attempts, list):
            return []
        return cast(list[ReflectionAttemptLike], attempts)

    @staticmethod
    def _last_verification_result(
        attempts: list[ReflectionAttemptLike],
    ) -> VerificationResultLike | None:
        if not attempts:
            return None
        return attempts[-1].verification_result

    @staticmethod
    def _record_runtime_reflection_trace(
        result: RuntimeResultLike,
        recorder: TraceRecorder,
    ) -> None:
        reflection_events = {
            "reflection_started",
            "verification_result",
            "critique_generated",
            "revision_requested",
            "reflection_completed",
            "reflection_failed",
        }

        for event in result.trace_events:
            if event.event_type not in reflection_events:
                continue

            recorder.record(
                cast(TraceEventType, event.event_type),
                step_index=event.step,
                payload=event.data,
            )
