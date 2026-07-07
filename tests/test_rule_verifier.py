from forge_agent.reflection import RuleVerifier
from forge_agent.runtime.base import RunResult
from forge_agent.tools.base import ToolResult


def test_rule_verifier_accepts_completed_answer() -> None:
    result = RunResult(
        final_answer="The task is complete.",
        stopped_reason="completed",
        steps=1,
    )

    verification = RuleVerifier().verify(result)

    assert verification.passed is True
    assert verification.decision == "accept"


def test_rule_verifier_rejects_empty_answer() -> None:
    result = RunResult(
        final_answer="",
        stopped_reason="completed",
        steps=1,
    )

    verification = RuleVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "revise"
    assert "final_answer is empty" in verification.reasons


def test_rule_verifier_retries_when_max_steps_reached() -> None:
    result = RunResult(
        final_answer="Partial answer.",
        stopped_reason="max_steps",
        steps=5,
    )

    verification = RuleVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "retry"
    assert "runtime reached max_steps before completion" in verification.reasons


def test_rule_verifier_aborts_on_runtime_error() -> None:
    result = RunResult(
        final_answer=None,
        stopped_reason="error",
        steps=1,
        error_message="provider failed",
    )

    verification = RuleVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "abort"
    assert "runtime stopped with reason: error" in verification.reasons


def test_rule_verifier_rejects_failed_tool_execution() -> None:
    result = RunResult(
        final_answer="The task completed successfully.",
        stopped_reason="completed",
        steps=1,
        tool_results=[
            ToolResult(
                tool_name="read_file",
                success=False,
                error_code="file_not_found",
                error_message="File not found.",
            )
        ],
    )

    verification = RuleVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "revise"
    assert "tool execution failed: read_file" in verification.reasons
    assert (
        "final_answer claims success despite failed tool execution"
        in verification.unsupported_claims
    )


def test_rule_verifier_aborts_on_permission_denied_tool() -> None:
    result = RunResult(
        final_answer="The write operation completed successfully.",
        stopped_reason="completed",
        steps=1,
        tool_results=[
            ToolResult(
                tool_name="write_file",
                success=False,
                error_code="PERMISSION_DENIED",
                error_message="Permission denied.",
            )
        ],
    )

    verification = RuleVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "abort"
    assert "permission denied by tool: write_file" in verification.reasons


def test_rule_verifier_rejects_abnormal_output() -> None:
    result = RunResult(
        final_answer="Traceback (most recent call last): ValueError",
        stopped_reason="completed",
        steps=1,
    )

    verification = RuleVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "revise"
    assert "final_answer contains abnormal runtime output" in verification.reasons
