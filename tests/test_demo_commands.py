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
