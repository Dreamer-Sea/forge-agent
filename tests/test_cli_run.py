from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge_agent.cli.app import app

runner = CliRunner()


def test_cli_run_executes_tool_calling_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# forge-agent", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "list files and read README"])

    assert result.exit_code == 0
    assert "Final answer:" in result.stdout
    assert "Stopped reason: completed" in result.stdout
    assert "Steps: 2" in result.stdout
    assert "- list_files: SUCCESS" in result.stdout
    assert "- read_file: SUCCESS" in result.stdout
    assert "- model_call" in result.stdout
    assert "- tool_result" in result.stdout
    assert "- final_answer" in result.stdout

def test_cli_run_with_native_runtime() -> None:
    from typer.testing import CliRunner

    from forge_agent.cli.app import app

    result = CliRunner().invoke(
        app,
        [
            "run",
            "echo hello",
            "--runtime",
            "native",
        ],
    )

    assert result.exit_code == 0
    assert "Runtime: native" in result.output
    assert "Stopped reason:" in result.output


def test_cli_run_with_langgraph_runtime() -> None:
    from typer.testing import CliRunner

    from forge_agent.cli.app import app

    result = CliRunner().invoke(
        app,
        [
            "run",
            "根据知识库回答 Permission system 如何工作",
            "--runtime",
            "langgraph",
        ],
    )

    assert result.exit_code == 0
    assert "Runtime: langgraph" in result.output
    assert "Stopped reason: completed" in result.output
    assert "Trace events:" in result.output


def test_cli_rejects_unknown_runtime() -> None:
    from typer.testing import CliRunner

    from forge_agent.cli.app import app

    result = CliRunner().invoke(
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
