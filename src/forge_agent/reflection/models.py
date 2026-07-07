"""Structured models for runtime reflection and verification."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

type ReflectionDecision = Literal["accept", "revise", "retry", "abort"]


class VerificationResult(BaseModel):
    """Deterministic verification result for an agent answer."""

    passed: bool
    decision: ReflectionDecision
    reasons: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> Self:
        """Keep pass/fail state consistent with the next reflection decision."""
        if self.passed and self.decision != "accept":
            msg = "passed verification results must use the accept decision"
            raise ValueError(msg)

        if not self.passed and self.decision == "accept":
            msg = "failed verification results cannot use the accept decision"
            raise ValueError(msg)

        return self


class Critique(BaseModel):
    """Human-readable critique generated from a failed verification."""

    summary: str
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None


class ReflectionAttempt(BaseModel):
    """One verification or reflection attempt within a runtime run."""

    attempt_index: int = Field(ge=0)
    verification_result: VerificationResult
    critique: Critique | None = None


class ReflectionConfig(BaseModel):
    """Configuration for bounded runtime reflection."""

    max_attempts: int = Field(default=1, ge=1)
    max_revisions: int = Field(default=1, ge=0)
    enable_revision: bool = True
    fail_fast: bool = True
