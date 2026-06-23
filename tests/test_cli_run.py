from __future__ import annotations

from typer.testing import CliRunner

from forge_agent.cli.app import app

runner = CliRunner()


def test_cli_run_skeleton() -> None:
    result = runner.invoke(app, ["run", "list files and read README"])

    assert result.exit_code == 0
    assert "Task: list files and read README" in result.stdout
    assert "Status: Day 1 skeleton ready" in result.stdout