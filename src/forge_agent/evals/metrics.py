"""Metrics for deterministic agent evals."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge_agent.evals.runner import EvalSuiteResult


class EvalMetrics(BaseModel):
    """Aggregated metrics for an eval suite."""

    model_config = ConfigDict(extra="forbid")

    case_count: int = 0
    passed_case_count: int = 0
    failed_case_count: int = 0
    success_rate: float = 1.0
    tool_call_success_rate: float = 1.0
    expected_contains_pass_rate: float = 1.0
    failed_cases: list[str] = Field(default_factory=list)

    @classmethod
    def from_suite(cls, suite: EvalSuiteResult) -> EvalMetrics:
        """Build metrics from a suite result."""
        case_count = suite.case_count
        passed_case_count = sum(1 for result in suite.results if result.passed)
        failed_case_count = case_count - passed_case_count

        expected_tool_count = sum(
            len(result.expected_tools) for result in suite.results
        )
        missing_tool_count = sum(
            len(result.missing_expected_tools) for result in suite.results
        )

        expected_text_count = sum(
            len(result.expected_contains) for result in suite.results
        )
        missing_text_count = sum(
            len(result.missing_expected_texts) for result in suite.results
        )

        return cls(
            case_count=case_count,
            passed_case_count=passed_case_count,
            failed_case_count=failed_case_count,
            success_rate=_safe_rate(passed_case_count, case_count),
            tool_call_success_rate=_safe_rate(
                expected_tool_count - missing_tool_count,
                expected_tool_count,
            ),
            expected_contains_pass_rate=_safe_rate(
                expected_text_count - missing_text_count,
                expected_text_count,
            ),
            failed_cases=[result.case_id for result in suite.failed_cases],
        )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0

    return numerator / denominator