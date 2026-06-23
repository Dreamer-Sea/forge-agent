from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from forge_agent.cli.app import app

runner = CliRunner()


def test_cli_rag_index_indexes_knowledge_base(tmp_path: Path) -> None:
    (tmp_path / "agent-runtime.md").write_text(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "Agent Runtime includes Agent Loop and Tool Registry.\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["rag", "index", str(tmp_path)])

    assert result.exit_code == 0
    assert "Documents: 1" in result.output
    assert "Chunks:" in result.output
    assert "- agent-runtime.md: Agent Runtime" in result.output


def test_cli_run_can_use_knowledge_base(tmp_path: Path) -> None:
    (tmp_path / "agent-runtime.md").write_text(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "Agent Runtime includes Agent Loop, Model Provider, Tool Registry, "
        "Tool Executor, Trace Recorder, and Settings.\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "run",
            "根据知识库回答：Agent Runtime 有哪些核心模块？",
            "--knowledge-base",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Final answer:" in result.output
    assert "Agent Runtime includes Agent Loop" in result.output
    assert "- search_knowledge_base: SUCCESS" in result.output
