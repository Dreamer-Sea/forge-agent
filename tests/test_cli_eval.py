from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from forge_agent.cli.app import app


def test_cli_eval_runs_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    report_path = tmp_path / "eval-report.md"
    trace_path = tmp_path / "traces.jsonl"

    dataset_path.write_text(
        (
            '{"id":"echo_001","input":"echo hello",'
            '"expected_tools":["echo_text"],'
            '"expected_contains":["hello"]}'
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(dataset_path),
            "--output",
            str(report_path),
            "--trace-out",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    assert "case_count: 1" in result.output
    assert "success_rate:" in result.output
    assert "tool_call_success_rate:" in result.output
    assert "expected_contains_pass_rate:" in result.output
    assert "failed_cases:" in result.output
    assert "trace_file:" in result.output
    assert report_path.exists()
    assert trace_path.exists()


def test_cli_eval_writes_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    report_path = tmp_path / "eval-report.md"
    trace_path = tmp_path / "traces.jsonl"

    dataset_path.write_text(
        (
            '{"id":"bad_001","input":"echo hello",'
            '"expected_tools":["echo_text"],'
            '"expected_contains":["不存在的模块X"]}'
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(dataset_path),
            "--output",
            str(report_path),
            "--trace-out",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    report_text = report_path.read_text(encoding="utf-8")

    assert "# Forge Agent Eval Report" in report_text
    assert "bad_001" in report_text
    assert "missing_expected_text" in report_text
    assert "不存在的模块X" in report_text


def test_cli_eval_writes_trace_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    report_path = tmp_path / "eval-report.md"
    trace_path = tmp_path / "traces.jsonl"

    dataset_path.write_text(
        (
            '{"id":"trace_001","input":"echo hello",'
            '"expected_tools":["echo_text"],'
            '"expected_contains":["hello"]}'
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(dataset_path),
            "--output",
            str(report_path),
            "--trace-out",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0

    trace_text = trace_path.read_text(encoding="utf-8")

    assert '"case_id":"trace_001"' in trace_text
    assert '"event_type":"model_call"' in trace_text
    assert '"event_type":"tool_call"' in trace_text
    assert '"event_type":"tool_result"' in trace_text
    assert '"event_type":"final_answer"' in trace_text


def test_cli_eval_writes_json_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    report_path = tmp_path / "eval-report.md"
    trace_path = tmp_path / "traces.jsonl"
    json_report_path = tmp_path / "eval-report.json"

    dataset_path.write_text(
        (
            '{"id":"echo_001","input":"echo hello",'
            '"expected_tools":["echo_text"],'
            '"expected_contains":["hello"]}'
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(dataset_path),
            "--output",
            str(report_path),
            "--trace-out",
            str(trace_path),
            "--json-output",
            str(json_report_path),
        ],
    )

    assert result.exit_code == 0
    assert json_report_path.exists()
    assert '"case_count": 1' in json_report_path.read_text(encoding="utf-8")


def test_cli_eval_rejects_invalid_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad.jsonl"

    dataset_path.write_text(
        '{"input":"missing id"}',
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(dataset_path),
        ],
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "invalid eval case" in result.output
