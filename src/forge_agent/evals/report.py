"""Eval report generation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from forge_agent.evals.metrics import EvalMetrics
from forge_agent.evals.runner import EvalResult, EvalSuiteResult


class EvalReport(BaseModel):
    """Serializable eval report."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    metrics: EvalMetrics
    results: list[EvalResult]
    trace_file: str

    @classmethod
    def from_suite(
        cls,
        suite: EvalSuiteResult,
        *,
        trace_file: str | Path,
    ) -> EvalReport:
        """Build a report from suite results."""
        return cls(
            metrics=EvalMetrics.from_suite(suite),
            results=suite.results,
            trace_file=str(trace_file),
        )

    def to_markdown(self) -> str:
        """Render the report as markdown."""
        lines = [
            "# Forge Agent Eval Report",
            "",
            "## Summary",
            "",
            f"- case_count: {self.metrics.case_count}",
            f"- success_rate: {_format_rate(self.metrics.success_rate)}",
            (
                f"- tool_call_success_rate: "
                f"{_format_rate(self.metrics.tool_call_success_rate)}"
            ),
            (
                "- expected_contains_pass_rate: "
                f"{_format_rate(self.metrics.expected_contains_pass_rate)}"
            ),
            f"- failed_cases: {len(self.metrics.failed_cases)}",
            f"- trace_file: {self.trace_file}",
            "",
            "## Reflection Verification",
            "",
        ]

        if not self.results:
            lines.append("No eval results.")
            lines.append("")
        else:
            lines.extend(
                [
                    (
                        "| case_id | reflection_attempts | verification_passed | "
                        "verification_reasons | unsupported_claims | missing_evidence |"
                    ),
                    "|---|---:|---|---|---|---|",
                ]
            )
            for result in self.results:
                lines.append(
                    "| "
                    f"{result.case_id} | "
                    f"{result.reflection_attempts} | "
                    f"{_format_optional_bool(result.verification_passed)} | "
                    f"{_format_list(result.verification_reasons)} | "
                    f"{_format_list(result.unsupported_claims)} | "
                    f"{_format_list(result.missing_evidence)} |"
                )
            lines.append("")

        lines.extend(
            [
                "## Failed Cases",
                "",
            ]
        )

        failed_results = [result for result in self.results if not result.passed]
        if not failed_results:
            lines.append("No failed cases.")
            lines.append("")
            return "\n".join(lines)

        lines.extend(
            [
                "| case_id | status | reason | missing |",
                "|---|---|---|---|",
            ]
        )

        for result in failed_results:
            lines.append(
                "| "
                f"{result.case_id} | "
                f"{result.status} | "
                f"{', '.join(result.failure_reasons)} | "
                f"{_format_missing(result)} |"
            )

        lines.append("")
        return "\n".join(lines)

    def to_json_text(self) -> str:
        """Render the report as formatted JSON."""
        return self.model_dump_json(indent=2)

    def write_markdown(self, path: str | Path) -> Path:
        """Write markdown report to disk."""
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(), encoding="utf-8")
        return report_path

    def write_json(self, path: str | Path) -> Path:
        """Write JSON report to disk."""
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_json_text(), encoding="utf-8")
        return report_path


def _format_rate(value: float) -> str:
    return f"{value:.2%}"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(value).lower()


def _format_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(values)


def _format_missing(result: EvalResult) -> str:
    missing_items = [
        *result.missing_expected_tools,
        *result.missing_expected_texts,
        *result.missing_expected_sources,
    ]

    if result.error_message is not None:
        missing_items.append(result.error_message)

    if not missing_items:
        return "-"

    return ", ".join(missing_items)
