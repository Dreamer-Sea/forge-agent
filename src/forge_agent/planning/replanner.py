from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from forge_agent.planning.models import (
    Plan,
    PlanDecision,
    PlanStep,
    ReplanReason,
    StepStatus,
)
from forge_agent.tools.base import ToolResult

if TYPE_CHECKING:
    from forge_agent.runtime.state import AgentState


class ReplanOutcome(BaseModel):
    """Structured decision returned by ReplanPolicy."""

    decision: PlanDecision
    reason: ReplanReason | None = None
    message: str
    replacement_steps: list[PlanStep] = Field(default_factory=list)


class ReplanPolicy:
    """Deterministic policy for deciding whether a failed step should replan."""

    def __init__(self, max_replans: int = 2) -> None:
        if max_replans < 0:
            raise ValueError("max_replans must be non-negative")

        self.max_replans = max_replans

    def evaluate_after_step(
        self,
        state: AgentState,
        plan: Plan,
        step: PlanStep,
        tool_result: ToolResult | None = None,
    ) -> ReplanOutcome:
        """Evaluate a step observation and decide how planning should proceed."""

        if state.replan_count >= self.max_replans:
            return ReplanOutcome(
                decision=PlanDecision.STOP,
                reason=ReplanReason.RETRY_LIMIT_REACHED,
                message="Replan limit reached; stop planning to avoid an infinite loop.",
            )

        if tool_result is not None and not tool_result.success:
            reason = self._reason_for_failed_tool(tool_result)
            failure_reason = self._format_tool_failure(tool_result)
            self._mark_step_failed_if_needed(plan, step, failure_reason)

            return ReplanOutcome(
                decision=PlanDecision.REPLAN,
                reason=reason,
                message=self._message_for_reason(reason),
                replacement_steps=self._replacement_steps_for(reason, step),
            )

        if step.evidence_required and self._has_empty_evidence(tool_result):
            self._mark_step_failed_if_needed(
                plan,
                step,
                "evidence required but no retrieval result was found",
            )

            return ReplanOutcome(
                decision=PlanDecision.REPLAN,
                reason=ReplanReason.EMPTY_RETRIEVAL,
                message="Required evidence was missing; replan with a fallback step.",
                replacement_steps=self._replacement_steps_for(
                    ReplanReason.EMPTY_RETRIEVAL,
                    step,
                ),
            )

        return ReplanOutcome(
            decision=PlanDecision.CONTINUE,
            message="Step observation is acceptable; continue with the current plan.",
        )

    def _mark_step_failed_if_needed(
        self,
        plan: Plan,
        step: PlanStep,
        reason: str,
    ) -> None:
        if step.status != StepStatus.FAILED:
            plan.mark_step_failed(step.id, reason)

    def _reason_for_failed_tool(self, tool_result: ToolResult) -> ReplanReason:
        if self._is_permission_denied(tool_result):
            return ReplanReason.PERMISSION_DENIED

        return ReplanReason.TOOL_FAILED

    def _is_permission_denied(self, tool_result: ToolResult) -> bool:
        values: list[str] = [
            tool_result.error_code or "",
            tool_result.error_message or "",
        ]
        values.extend(str(value) for value in tool_result.safe_detail.values())

        text = " ".join(values).lower()

        return (
            "permission_denied" in text
            or ("permission" in text and "denied" in text)
            or "not allowed" in text
        )

    def _format_tool_failure(self, tool_result: ToolResult) -> str:
        if tool_result.error_message:
            return tool_result.error_message

        if tool_result.error_code:
            return tool_result.error_code

        return f"tool {tool_result.tool_name} failed"

    def _has_empty_evidence(self, tool_result: ToolResult | None) -> bool:
        if tool_result is None:
            return True

        if not tool_result.success:
            return False

        payload = tool_result.payload

        if not payload:
            return True

        evidence_keys = ("results", "sources", "documents", "chunks")
        for key in evidence_keys:
            value = payload.get(key)
            if isinstance(value, list) and len(value) == 0:
                return True

        context = payload.get("context")
        if isinstance(context, str) and not context.strip():
            return True

        return False

    def _message_for_reason(self, reason: ReplanReason) -> str:
        messages: dict[ReplanReason, str] = {
            ReplanReason.TOOL_FAILED: "Tool execution failed; replan with a recovery step.",
            ReplanReason.PERMISSION_DENIED: (
                "Permission was denied; do not retry the denied action."
            ),
            ReplanReason.EMPTY_RETRIEVAL: (
                "Retrieval returned no evidence; replan with a fallback step."
            ),
            ReplanReason.EVIDENCE_MISSING: (
                "Required evidence is missing; replan with a safer answer step."
            ),
            ReplanReason.STEP_NOT_EXECUTABLE: (
                "Current step is not executable; replan the task."
            ),
            ReplanReason.RETRY_LIMIT_REACHED: (
                "Replan limit reached; stop planning."
            ),
        }
        return messages[reason]

    def _replacement_steps_for(
        self,
        reason: ReplanReason,
        step: PlanStep,
    ) -> list[PlanStep]:
        if reason == ReplanReason.PERMISSION_DENIED:
            return [
                PlanStep(
                    id="explain_permission_denial",
                    description=(
                        "Explain why the requested operation was denied and "
                        "provide a safe alternative."
                    ),
                    expected_outcome=(
                        "The user receives a safe explanation without retrying "
                        "the denied action."
                    ),
                )
            ]

        if reason in {
            ReplanReason.EMPTY_RETRIEVAL,
            ReplanReason.EVIDENCE_MISSING,
        }:
            return [
                PlanStep(
                    id="answer_without_evidence",
                    description=(
                        "Explain that no supporting evidence was found and "
                        "provide a bounded answer."
                    ),
                    expected_outcome=(
                        "The user receives an answer that clearly states the "
                        "evidence limitation."
                    ),
                )
            ]

        return [
            PlanStep(
                id=f"recover_{step.id}",
                description=f"Recover from failed step: {step.description}",
                expected_outcome="A safe recovery path is attempted.",
            )
        ]


def payload_has_evidence(payload: dict[str, Any]) -> bool:
    """Return whether a tool payload contains some evidence-like data."""

    if not payload:
        return False

    for key in ("results", "sources", "documents", "chunks"):
        value = payload.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True

    context = payload.get("context")
    return isinstance(context, str) and bool(context.strip())
