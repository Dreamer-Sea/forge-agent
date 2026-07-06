from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class StepStatus(StrEnum):
    """Lifecycle status for one plan step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(StrEnum):
    """Lifecycle status for a whole plan."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReplanReason(StrEnum):
    """Reasons that can trigger replanning."""

    TOOL_FAILED = "tool_failed"
    EMPTY_RETRIEVAL = "empty_retrieval"
    EVIDENCE_MISSING = "evidence_missing"
    STEP_NOT_EXECUTABLE = "step_not_executable"
    PERMISSION_DENIED = "permission_denied"
    RETRY_LIMIT_REACHED = "retry_limit_reached"


class PlanDecision(StrEnum):
    """Decision made after observing a step result."""

    CONTINUE = "continue"
    REPLAN = "replan"
    STOP = "stop"


class PlanStep(BaseModel):
    """One executable unit inside a plan."""

    id: str = Field(default_factory=lambda: _new_id("step"))
    description: str
    expected_outcome: str
    status: StepStatus = StepStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)
    tool_hint: str | None = None
    evidence_required: bool = False
    failure_reason: str | None = None

    def mark_running(self) -> Self:
        self.status = StepStatus.RUNNING
        self.failure_reason = None
        return self

    def mark_succeeded(self) -> Self:
        self.status = StepStatus.SUCCEEDED
        self.failure_reason = None
        return self

    def mark_failed(self, reason: str) -> Self:
        self.status = StepStatus.FAILED
        self.failure_reason = reason
        return self


class Plan(BaseModel):
    """A traceable plan created for one user task."""

    id: str = Field(default_factory=lambda: _new_id("plan"))
    user_input: str
    steps: list[PlanStep]
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    def mark_running(self) -> Self:
        self.status = PlanStatus.RUNNING
        self._touch()
        return self

    def mark_succeeded(self) -> Self:
        self.status = PlanStatus.SUCCEEDED
        self._touch()
        return self

    def mark_failed(self) -> Self:
        self.status = PlanStatus.FAILED
        self._touch()
        return self

    def is_completed(self) -> bool:
        return bool(self.steps) and all(
            step.status == StepStatus.SUCCEEDED for step in self.steps
        )

    def next_executable_step(self) -> PlanStep | None:
        succeeded_step_ids = {
            step.id for step in self.steps if step.status == StepStatus.SUCCEEDED
        }

        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue

            if all(dependency in succeeded_step_ids for dependency in step.depends_on):
                return step

        return None

    def mark_step_running(self, step_id: str) -> PlanStep:
        step = self._get_step(step_id)
        step.mark_running()
        self.mark_running()
        self._touch()
        return step

    def mark_step_succeeded(self, step_id: str) -> PlanStep:
        step = self._get_step(step_id)
        step.mark_succeeded()

        if self.is_completed():
            self.mark_succeeded()
        else:
            self.mark_running()

        self._touch()
        return step

    def mark_step_failed(self, step_id: str, reason: str) -> PlanStep:
        step = self._get_step(step_id)
        step.mark_failed(reason)
        self.mark_failed()
        self._touch()
        return step

    def _get_step(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step

        raise ValueError(f"Unknown plan step: {step_id}")

    def _touch(self) -> None:
        self.updated_at = _utc_now()
