from __future__ import annotations

from pathlib import Path

from forge_agent.evals import EvalMetrics, EvalReport, EvalResult, EvalSuiteResult


def test_eval_metrics_calculates_success_rate() -> None:
    suite = EvalSuiteResult(
        results=[
            EvalResult(
                case_id="case_001",
                status="passed",
                input="ok",
                run_id="run-001",
            ),
            EvalResult(
                case_id="case_002",
                status="failed",
                input="bad",
                failure_reasons=["missing_expected_text"],
                run_id="run-002",
            ),
        ]
    )

    metrics = EvalMetrics.from_suite(suite)

    assert metrics.case_count == 2
    assert metrics.passed_case_count == 1
    assert metrics.failed_case_count == 1
    assert metrics.success_rate == 0.5
    assert metrics.failed_cases == ["case_002"]


def test_eval_metrics_calculates_tool_call_success_rate() -> None:
    suite = EvalSuiteResult(
        results=[
            EvalResult(
                case_id="case_001",
                status="passed",
                input="ok",
                expected_tools=["read_file"],
                observed_tools=["read_file"],
                run_id="run-001",
            ),
            EvalResult(
                case_id="case_002",
                status="failed",
                input="bad",
                expected_tools=["search_knowledge_base"],
                missing_expected_tools=["search_knowledge_base"],
                failure_reasons=["missing_expected_tool"],
                run_id="run-002",
            ),
        ]
    )

    metrics = EvalMetrics.from_suite(suite)

    assert metrics.tool_call_success_rate == 0.5


def test_eval_metrics_calculates_expected_contains_pass_rate() -> None:
    suite = EvalSuiteResult(
        results=[
            EvalResult(
                case_id="case_001",
                status="passed",
                input="ok",
                expected_contains=["Runtime", "Tool"],
                run_id="run-001",
            ),
            EvalResult(
                case_id="case_002",
                status="failed",
                input="bad",
                expected_contains=["Agent", "Missing"],
                missing_expected_texts=["Missing"],
                failure_reasons=["missing_expected_text"],
                run_id="run-002",
            ),
        ]
    )

    metrics = EvalMetrics.from_suite(suite)

    assert metrics.expected_contains_pass_rate == 0.75


def test_eval_report_includes_failed_cases() -> None:
    suite = EvalSuiteResult(
        results=[
            EvalResult(
                case_id="bad_001",
                status="failed",
                input="Agent Runtime 有哪些核心模块？",
                expected_contains=["不存在的模块X"],
                missing_expected_texts=["不存在的模块X"],
                failure_reasons=["missing_expected_text"],
                run_id="run-001",
            )
        ]
    )

    report = EvalReport.from_suite(suite, trace_file="reports/traces.jsonl")
    markdown = report.to_markdown()

    assert "bad_001" in markdown
    assert "failed" in markdown
    assert "missing_expected_text" in markdown
    assert "不存在的模块X" in markdown
    assert "reports/traces.jsonl" in markdown


def test_eval_report_writes_markdown(tmp_path: Path) -> None:
    suite = EvalSuiteResult(
        results=[
            EvalResult(
                case_id="case_001",
                status="passed",
                input="ok",
                run_id="run-001",
            )
        ]
    )
    report = EvalReport.from_suite(suite, trace_file="reports/traces.jsonl")
    report_path = tmp_path / "eval-report.md"

    written_path = report.write_markdown(report_path)

    assert written_path == report_path
    text = report_path.read_text(encoding="utf-8")
    assert "# Forge Agent Eval Report" in text
    assert "case_count: 1" in text
    assert "success_rate: 100.00%" in text


def test_eval_report_writes_json(tmp_path: Path) -> None:
    suite = EvalSuiteResult(
        results=[
            EvalResult(
                case_id="case_001",
                status="passed",
                input="ok",
                run_id="run-001",
            )
        ]
    )
    report = EvalReport.from_suite(suite, trace_file="reports/traces.jsonl")
    report_path = tmp_path / "eval-report.json"

    written_path = report.write_json(report_path)

    assert written_path == report_path
    text = report_path.read_text(encoding="utf-8")
    assert '"case_count": 1' in text
    assert '"trace_file": "reports/traces.jsonl"' in text
