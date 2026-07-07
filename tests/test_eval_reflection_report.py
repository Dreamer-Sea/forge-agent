from pathlib import Path

from typer.testing import CliRunner

from forge_agent.cli.app import app
from forge_agent.evals.report import EvalReport
from forge_agent.evals.runner import EvalResult, EvalSuiteResult


def test_eval_report_renders_reflection_fields() -> None:
    suite = EvalSuiteResult(
        results=[
            EvalResult(
                case_id="reflection_001",
                status="passed",
                input="echo hello",
                final_answer="I inspected the workspace using these tools: echo_text.",
                stopped_reason="completed",
                run_id="run-1",
                reflection_attempts=1,
                verification_passed=True,
                verification_reasons=["rule verification passed"],
                unsupported_claims=[],
                missing_evidence=[],
            )
        ]
    )

    report = EvalReport.from_suite(suite, trace_file="traces.jsonl")
    markdown = report.to_markdown()
    json_text = report.to_json_text()

    assert "## Reflection Verification" in markdown
    assert "reflection_attempts" in markdown
    assert "verification_passed" in markdown
    assert "verification_reasons" in markdown
    assert "unsupported_claims" in markdown
    assert "missing_evidence" in markdown
    assert "reflection_001" in markdown
    assert "rule verification passed" in markdown

    assert '"reflection_attempts": 1' in json_text
    assert '"verification_passed": true' in json_text
    assert '"verification_reasons": [' in json_text


def test_cli_eval_with_reflection_runtime_writes_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    report_path = tmp_path / "reflection-report.md"
    trace_path = tmp_path / "reflection-traces.jsonl"

    dataset_path.write_text(
        (
            '{"id":"reflection_eval_001","input":"echo hello",'
            '"expected_tools":["echo_text"],'
            '"expected_contains":["echo_text"]}'
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(dataset_path),
            "--runtime",
            "reflection",
            "--output",
            str(report_path),
            "--trace-out",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    assert "case_count: 1" in result.output
    assert report_path.exists()
    assert trace_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    trace_text = trace_path.read_text(encoding="utf-8")

    assert "## Reflection Verification" in report_text
    assert "reflection_eval_001" in report_text
    assert "verification_passed" in report_text
    assert "reflection_started" in trace_text
    assert "verification_result" in trace_text
