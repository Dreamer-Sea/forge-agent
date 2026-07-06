"""Reflection runtime wrapper for verified agent outputs."""

from __future__ import annotations

from pydantic import Field

from forge_agent.reflection.models import (
    Critique,
    ReflectionAttempt,
    ReflectionConfig,
    VerificationResult,
)
from forge_agent.reflection.rule_verifier import RuleVerifier
from forge_agent.reflection.verifier import Verifier
from forge_agent.runtime.base import AgentRuntime, RunConfig, RunResult, StoppedReason
from forge_agent.runtime.events import TraceEvent, TraceEventType


class ReflectionRunResult(RunResult):
    """Run result enriched with runtime reflection attempts."""

    reflection_attempts: list[ReflectionAttempt] = Field(default_factory=list)


class ReflectionRuntime:
    """Runtime wrapper that verifies final answers before returning them."""

    def __init__(
        self,
        base_runtime: AgentRuntime,
        verifier: Verifier | None = None,
        reflection_config: ReflectionConfig | None = None,
    ) -> None:
        self._base_runtime = base_runtime
        self._verifier = verifier or RuleVerifier()
        self._reflection_config = reflection_config or ReflectionConfig()

    def run(
        self,
        user_input: str,
        config: RunConfig | None = None,
    ) -> ReflectionRunResult:
        """Run the base runtime and verify its output before returning."""
        result = self._base_runtime.run(user_input, config)
        trace_events = list(result.trace_events)
        attempts: list[ReflectionAttempt] = []
        revisions_used = 0

        self._append_trace(
            trace_events,
            event_type="reflection_started",
            step=result.steps,
            data={
                "max_attempts": self._reflection_config.max_attempts,
                "max_revisions": self._reflection_config.max_revisions,
                "enable_revision": self._reflection_config.enable_revision,
            },
        )

        for attempt_index in range(self._reflection_config.max_attempts):
            verification = self._verifier.verify(result, state=None)
            critique = self._critique_from_verification(verification)
            attempts.append(
                ReflectionAttempt(
                    attempt_index=attempt_index,
                    verification_result=verification,
                    critique=critique,
                )
            )

            self._append_verification_trace(
                trace_events=trace_events,
                step=result.steps,
                attempt_index=attempt_index,
                verification=verification,
            )

            if critique is not None:
                self._append_trace(
                    trace_events,
                    event_type="critique_generated",
                    step=result.steps,
                    data={
                        "attempt_index": attempt_index,
                        "summary": critique.summary,
                        "issues": critique.issues,
                        "suggested_fix": critique.suggested_fix,
                    },
                )

            if verification.passed:
                self._append_trace(
                    trace_events,
                    event_type="reflection_completed",
                    step=result.steps,
                    data={
                        "attempt_index": attempt_index,
                        "decision": verification.decision,
                    },
                )
                return self._to_reflection_result(
                    result=result,
                    trace_events=trace_events,
                    attempts=attempts,
                )

            if verification.decision == "abort":
                return self._fail_result(
                    result=result,
                    trace_events=trace_events,
                    attempts=attempts,
                    stopped_reason="verification_failed",
                    error_message=self._failure_message(verification),
                )

            if verification.decision == "revise":
                if not self._can_revise(attempt_index, revisions_used):
                    return self._fail_result(
                        result=result,
                        trace_events=trace_events,
                        attempts=attempts,
                        stopped_reason="reflection_limit_reached",
                        error_message=self._failure_message(verification),
                    )

                revisions_used += 1
                self._append_trace(
                    trace_events,
                    event_type="revision_requested",
                    step=result.steps,
                    data={
                        "attempt_index": attempt_index,
                        "revisions_used": revisions_used,
                        "reasons": verification.reasons,
                    },
                )
                result = self._revise_result(result, verification, critique)
                continue

            if verification.decision == "retry":
                if attempt_index + 1 >= self._reflection_config.max_attempts:
                    return self._fail_result(
                        result=result,
                        trace_events=trace_events,
                        attempts=attempts,
                        stopped_reason="reflection_limit_reached",
                        error_message=self._failure_message(verification),
                    )

                retry_result = self._base_runtime.run(user_input, config)
                trace_events.extend(retry_result.trace_events)
                result = retry_result
                continue

        return self._fail_result(
            result=result,
            trace_events=trace_events,
            attempts=attempts,
            stopped_reason="reflection_limit_reached",
            error_message="Reflection limit reached before verification passed.",
        )

    def _can_revise(self, attempt_index: int, revisions_used: int) -> bool:
        if not self._reflection_config.enable_revision:
            return False

        if revisions_used >= self._reflection_config.max_revisions:
            return False

        return attempt_index + 1 < self._reflection_config.max_attempts

    @staticmethod
    def _critique_from_verification(
        verification: VerificationResult,
    ) -> Critique | None:
        if verification.passed:
            return None

        issues = [
            *verification.reasons,
            *verification.missing_evidence,
            *verification.unsupported_claims,
        ]
        return Critique(
            summary="Verification failed.",
            issues=issues,
            suggested_fix="Revise the final answer so it satisfies verification rules.",
        )

    @staticmethod
    def _revise_result(
        result: RunResult,
        verification: VerificationResult,
        critique: Critique | None,
    ) -> RunResult:
        original_answer = (result.final_answer or "").strip()
        revised_parts = ["Verification requested a revision."]

        if original_answer:
            revised_parts.append(original_answer)

        if verification.reasons:
            revised_parts.append(
                "Verification issues: " + "; ".join(verification.reasons)
            )

        if critique is not None and critique.suggested_fix:
            revised_parts.append("Suggested fix: " + critique.suggested_fix)

        return result.model_copy(
            update={
                "final_answer": "\n".join(revised_parts),
                "stopped_reason": "completed",
            },
            deep=True,
        )

    @staticmethod
    def _append_verification_trace(
        *,
        trace_events: list[TraceEvent],
        step: int,
        attempt_index: int,
        verification: VerificationResult,
    ) -> None:
        ReflectionRuntime._append_trace(
            trace_events,
            event_type="verification_result",
            step=step,
            data={
                "attempt_index": attempt_index,
                "passed": verification.passed,
                "decision": verification.decision,
                "reasons": verification.reasons,
                "missing_evidence": verification.missing_evidence,
                "unsupported_claims": verification.unsupported_claims,
                "confidence": verification.confidence,
            },
        )

    @staticmethod
    def _append_trace(
        trace_events: list[TraceEvent],
        *,
        event_type: TraceEventType,
        step: int,
        data: dict[str, object],
    ) -> None:
        trace_events.append(
            TraceEvent(
                event_type=event_type,
                step=step,
                data=data,
            )
        )

    @staticmethod
    def _failure_message(verification: VerificationResult) -> str:
        if verification.reasons:
            return "; ".join(verification.reasons)

        return f"Verification failed with decision: {verification.decision}"

    @staticmethod
    def _to_reflection_result(
        *,
        result: RunResult,
        trace_events: list[TraceEvent],
        attempts: list[ReflectionAttempt],
    ) -> ReflectionRunResult:
        return ReflectionRunResult(
            final_answer=result.final_answer,
            stopped_reason=result.stopped_reason,
            steps=result.steps,
            tool_results=list(result.tool_results),
            tool_calls=list(result.tool_calls),
            trace_events=trace_events,
            error_message=result.error_message,
            reflection_attempts=attempts,
        )

    @staticmethod
    def _fail_result(
        *,
        result: RunResult,
        trace_events: list[TraceEvent],
        attempts: list[ReflectionAttempt],
        stopped_reason: StoppedReason,
        error_message: str,
    ) -> ReflectionRunResult:
        ReflectionRuntime._append_trace(
            trace_events,
            event_type="reflection_failed",
            step=result.steps,
            data={
                "reason": stopped_reason,
                "error_message": error_message,
            },
        )
        return ReflectionRunResult(
            final_answer=result.final_answer,
            stopped_reason=stopped_reason,
            steps=result.steps,
            tool_results=list(result.tool_results),
            tool_calls=list(result.tool_calls),
            trace_events=trace_events,
            error_message=error_message,
            reflection_attempts=attempts,
        )


__all__ = ["ReflectionRunResult", "ReflectionRuntime"]
