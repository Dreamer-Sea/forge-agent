from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge_agent.cli.app import app

runner = CliRunner()


def test_cli_run_with_planning_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# demo\n\nPlanning runtime demo.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "echo hello",
            "--runtime",
            "planning",
        ],
    )

    assert result.exit_code == 0
    assert "runtime: planning" in result.output
    assert "stopped_reason: completed" in result.output
    assert "final_answer:" in result.output
    assert "- echo_text: SUCCESS" in result.output
    assert "- plan_created" in result.output
    assert "- plan_step_started" in result.output
    assert "- plan_step_completed" in result.output
    assert "- plan_completed" in result.output


def test_cli_planning_runtime_returns_structured_stop_reason() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "echo hello",
            "--runtime",
            "planning",
            "--max-steps",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert "runtime: planning" in result.output
    assert "stopped_reason: max_steps" in result.output
    assert "- plan_created" in result.output
    assert "- runtime_stop" in result.output


def test_cli_help_mentions_planning_runtime() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "Runtime backend to use: native" in result.output
    assert "langgraph, or planning" in result.output


def test_cli_rejects_unknown_runtime_with_planning_in_supported_list() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "hello",
            "--runtime",
            "unknown",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown runtime: unknown" in result.output
    assert "Supported runtimes: native, langgraph, planning" in result.output
