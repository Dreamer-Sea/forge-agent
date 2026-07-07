from typer.testing import CliRunner

from forge_agent.cli.app import app


def test_cli_run_supports_reflection_runtime() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "echo hello",
            "--runtime",
            "reflection",
        ],
    )

    assert result.exit_code == 0
    assert "runtime: reflection" in result.output
    assert "stopped_reason:" in result.output
    assert "final_answer:" in result.output
    assert "reflection_started" in result.output
    assert "verification_result" in result.output


def test_cli_rejects_unknown_runtime_with_reflection_hint() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "echo hello",
            "--runtime",
            "unknown-runtime",
        ],
    )

    assert result.exit_code != 0
    assert "Supported runtimes: native, langgraph, planning, reflection." in result.output
