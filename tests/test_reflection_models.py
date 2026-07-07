import pytest
from pydantic import ValidationError

from forge_agent.reflection import (
    Critique,
    ReflectionAttempt,
    ReflectionConfig,
    VerificationResult,
)


def test_verification_result_can_accept_valid_answer() -> None:
    result = VerificationResult(
        passed=True,
        decision="accept",
        reasons=["answer is complete"],
        confidence=0.95,
    )

    assert result.passed is True
    assert result.decision == "accept"
    assert result.reasons == ["answer is complete"]
    assert result.confidence == 0.95


@pytest.mark.parametrize("decision", ["revise", "retry", "abort"])
def test_verification_result_can_reject_with_non_accept_decisions(
    decision: str,
) -> None:
    result = VerificationResult(
        passed=False,
        decision=decision,  # type: ignore[arg-type]
        reasons=["answer failed verification"],
        missing_evidence=["citation"],
        unsupported_claims=["claim without source"],
        confidence=0.4,
    )

    assert result.passed is False
    assert result.decision == decision
    assert result.missing_evidence == ["citation"]
    assert result.unsupported_claims == ["claim without source"]


def test_passed_result_must_use_accept_decision() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            passed=True,
            decision="revise",
            reasons=["inconsistent decision"],
        )


def test_failed_result_cannot_use_accept_decision() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            passed=False,
            decision="accept",
            reasons=["inconsistent decision"],
        )


def test_critique_records_issues_and_suggested_fix() -> None:
    critique = Critique(
        summary="The answer lacks evidence.",
        issues=["missing citation", "unsupported claim"],
        suggested_fix="Add citations from retrieved context.",
    )

    assert critique.summary == "The answer lacks evidence."
    assert critique.issues == ["missing citation", "unsupported claim"]
    assert critique.suggested_fix == "Add citations from retrieved context."


def test_reflection_attempt_records_attempt_index_result_and_critique() -> None:
    verification = VerificationResult(
        passed=False,
        decision="revise",
        reasons=["missing evidence"],
        missing_evidence=["retrieved context citation"],
    )
    critique = Critique(
        summary="The answer should cite retrieved context.",
        issues=["missing citation"],
        suggested_fix="Revise the final answer with citations.",
    )

    attempt = ReflectionAttempt(
        attempt_index=0,
        verification_result=verification,
        critique=critique,
    )

    assert attempt.attempt_index == 0
    assert attempt.verification_result.decision == "revise"
    assert attempt.critique is not None
    assert attempt.critique.issues == ["missing citation"]


def test_reflection_config_limits_attempts_and_revisions() -> None:
    config = ReflectionConfig(
        max_attempts=2,
        max_revisions=1,
        enable_revision=True,
        fail_fast=False,
    )

    assert config.max_attempts == 2
    assert config.max_revisions == 1
    assert config.enable_revision is True
    assert config.fail_fast is False


def test_reflection_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValidationError):
        ReflectionConfig(max_attempts=0)

    with pytest.raises(ValidationError):
        ReflectionConfig(max_revisions=-1)
