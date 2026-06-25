from __future__ import annotations

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
