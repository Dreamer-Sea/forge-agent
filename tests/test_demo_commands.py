from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from forge_agent.cli.app import app

runner = CliRunner()


def test_demo_tool_calling_command_succeeds() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "Read the project README and summarize the architecture.",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "runtime:" in result.output
    assert "tools_used:" in result.output
    assert "stopped_reason:" in result.output
    assert "final_answer:" in result.output


def test_demo_tool_calling_uses_read_file_tool() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "Read the project README and summarize the architecture.",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "read_file" in result.output
    assert "completed" in result.output


def test_demo_rag_command_succeeds() -> None:
    index_result = runner.invoke(
        app,
        [
            "rag",
            "index",
            "examples/knowledge_base",
        ],
    )

    assert index_result.exit_code == 0, index_result.output
    assert "Documents:" in index_result.output
    assert "Chunks:" in index_result.output

    run_result = runner.invoke(
        app,
        [
            "run",
            "According to the knowledge base, how does the permission system work?",
        ],
    )

    assert run_result.exit_code == 0, run_result.output
    assert "runtime:" in run_result.output
    assert "tools_used:" in run_result.output
    assert "search_knowledge_base" in run_result.output
    assert "final_answer:" in run_result.output


def test_demo_rag_answer_includes_citation() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "According to the knowledge base, how does the permission system work?",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Citations:" in result.output
    assert "security.md" in result.output
    assert "Permission System" in result.output


def test_demo_rag_trace_includes_search_tool() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "According to the knowledge base, how does the permission system work?",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Tool calls:" in result.output
    assert "search_knowledge_base: SUCCESS" in result.output
    assert "Trace events:" in result.output
    assert "tool_call" in result.output
    assert "tool_result" in result.output

def test_demo_eval_command_succeeds() -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "examples/evals/agent_platform.jsonl",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "case_count: 3" in result.output
    assert "success_rate: 100.00%" in result.output
    assert "failed_cases: 0" in result.output


def test_demo_eval_report_contains_success_rate() -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "examples/evals/agent_platform.jsonl",
        ],
    )

    assert result.exit_code == 0, result.output

    report = Path("reports/eval-report.md")
    assert report.exists()
    assert "success_rate" in report.read_text(encoding="utf-8")


def test_demo_eval_generates_trace_file() -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "examples/evals/agent_platform.jsonl",
        ],
    )

    assert result.exit_code == 0, result.output

    trace = Path("reports/traces.jsonl")
    assert trace.exists()
    assert trace.read_text(encoding="utf-8").strip()
