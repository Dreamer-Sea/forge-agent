from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge_agent.cli.app import app

runner = CliRunner()


def write_knowledge_base(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "agent-runtime.md").write_text(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "Agent Runtime includes Agent Loop, Model Provider, Tool Registry, "
        "Tool Executor, Trace Recorder, and Settings.\n",
        encoding="utf-8",
    )


def test_cli_rag_index_indexes_knowledge_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    write_knowledge_base(knowledge_base)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["rag", "index", "knowledge_base"])

    assert result.exit_code == 0, result.output
    assert "Knowledge base: knowledge_base" in result.output
    assert "Documents: 1" in result.output
    assert "Chunks:" in result.output
    assert "agent-runtime.md" in result.output


def test_cli_run_can_use_knowledge_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    write_knowledge_base(knowledge_base)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "根据知识库回答：Agent Runtime 有哪些核心模块？",
            "--knowledge-base",
            "knowledge_base",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "final_answer:" in result.output
    assert "Agent Runtime includes Agent Loop" in result.output
    assert "- search_knowledge_base: SUCCESS" in result.output


def test_cli_rag_index_outside_workspace_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "project"
    outside_kb = tmp_path / "outside_kb"
    workspace_root.mkdir()
    outside_kb.mkdir()
    (outside_kb / "secret.md").write_text("# Secret\n", encoding="utf-8")
    monkeypatch.chdir(workspace_root)

    result = runner.invoke(app, ["rag", "index", str(outside_kb)])

    assert result.exit_code == 1
    assert "PATH_OUTSIDE_WORKSPACE" in result.output
    assert str(outside_kb) not in result.output


def test_cli_rag_index_inside_workspace_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    write_knowledge_base(knowledge_base)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["rag", "index", "knowledge_base"])

    assert result.exit_code == 0, result.output
    assert "Knowledge base: knowledge_base" in result.output
    assert "Documents: 1" in result.output
    assert "agent-runtime.md" in result.output
