"""Rule-based verifier for generic agent run results."""

from __future__ import annotations

from forge_agent.reflection.models import ReflectionDecision, VerificationResult
from forge_agent.runtime.base import RunResult
from forge_agent.runtime.state import AgentState
from forge_agent.tools.base import ToolResult


class RuleVerifier:
    """Verify generic runtime correctness with deterministic rules."""

    _ABNORMAL_OUTPUT_MARKERS = (
        "traceback (most recent call last)",
        "unhandled exception",
        "internal server error",
    )

    def verify(
        self,
        result: RunResult,
        state: AgentState | None = None,
    ) -> VerificationResult:
        """Verify a runtime result with deterministic, testable rules."""
        del state

        reasons: list[str] = []
        unsupported_claims: list[str] = []

        final_answer = (result.final_answer or "").strip()
        if not final_answer:
            reasons.append("final_answer is empty")

        if result.stopped_reason == "max_steps":
            reasons.append("runtime reached max_steps before completion")
        elif result.stopped_reason != "completed":
            reasons.append(f"runtime stopped with reason: {result.stopped_reason}")

        failed_tools = self._failed_tools(result)
        if failed_tools:
            failed_tool_names = ", ".join(tool.tool_name for tool in failed_tools)
            reasons.append(f"tool execution failed: {failed_tool_names}")

        permission_denied_tools = [
            tool for tool in failed_tools if self._is_permission_denied(tool)
        ]
        if permission_denied_tools:
            tool_names = ", ".join(tool.tool_name for tool in permission_denied_tools)
            reasons.append(f"permission denied by tool: {tool_names}")

        if self._contains_abnormal_output(final_answer):
            unsupported_claims.append("final_answer contains abnormal runtime output")
            reasons.append("final_answer contains abnormal runtime output")

        if failed_tools and self._claims_success(final_answer):
            unsupported_claims.append(
                "final_answer claims success despite failed tool execution"
            )
            reasons.append("final_answer claims success despite failed tool execution")

        if not reasons:
            return VerificationResult(
                passed=True,
                decision="accept",
                reasons=["rule verification passed"],
                confidence=1.0,
            )

        return VerificationResult(
            passed=False,
            decision=self._decision_for_failure(result, permission_denied_tools),
            reasons=reasons,
            unsupported_claims=unsupported_claims,
            confidence=0.2,
        )

    @staticmethod
    def _failed_tools(result: RunResult) -> list[ToolResult]:
        return [tool for tool in result.tool_results if not tool.success]

    @staticmethod
    def _is_permission_denied(tool: ToolResult) -> bool:
        error_code = (tool.error_code or "").upper()
        error_message = (tool.error_message or "").lower()
        return error_code == "PERMISSION_DENIED" or "permission denied" in error_message

    @classmethod
    def _contains_abnormal_output(cls, final_answer: str) -> bool:
        normalized = final_answer.lower()
        return any(marker in normalized for marker in cls._ABNORMAL_OUTPUT_MARKERS)

    @staticmethod
    def _claims_success(final_answer: str) -> bool:
        normalized = final_answer.lower()
        success_markers = (
            "success",
            "succeeded",
            "completed successfully",
            "执行成功",
            "已成功",
            "成功完成",
        )
        return any(marker in normalized for marker in success_markers)

    @staticmethod
    def _decision_for_failure(
        result: RunResult,
        permission_denied_tools: list[ToolResult],
    ) -> ReflectionDecision:
        if permission_denied_tools:
            return "abort"

        if result.stopped_reason == "max_steps":
            return "retry"

        if result.stopped_reason != "completed":
            return "abort"

        return "revise"
