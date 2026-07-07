from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge_agent.cli.app import app

runner = CliRunner()


def test_cli_run_can_persist_and_recall_session_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    write_result = runner.invoke(
        app,
        [
            "run",
            "记住：我的项目默认使用 Python 3.13 和 uv",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo",
            "--max-steps",
            "1",
        ],
    )

    assert write_result.exit_code == 0, write_result.output
    assert "memory: enabled" in write_result.output
    assert "memory_session_id: demo" in write_result.output
    assert "memory_write_completed" in write_result.output

    recall_result = runner.invoke(
        app,
        [
            "run",
            "我的项目默认使用什么 Python 版本？",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo",
            "--max-steps",
            "1",
        ],
    )

    assert recall_result.exit_code == 0, recall_result.output
    assert "memory_recall_started" in recall_result.output
    assert "memory_recall_result" in recall_result.output
    assert "final_answer: 根据长期记忆，项目默认使用 Python 3.13 和 uv。" in (
        recall_result.output
    )


def test_cli_memory_list_outputs_persisted_memories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app,
        [
            "run",
            "记住：我的项目默认使用 Python 3.13 和 uv",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo",
            "--max-steps",
            "1",
        ],
    )

    result = runner.invoke(
        app,
        [
            "memory",
            "list",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo",
            "--type",
            "semantic",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Records: 1" in result.output
    assert "type=semantic" in result.output
    assert "scope=session:demo" in result.output
    assert "我的项目默认使用 Python 3.13 和 uv" in result.output


def test_cli_memory_search_outputs_ranked_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app,
        [
            "run",
            "记住：我的项目默认使用 Python 3.13 和 uv",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo",
            "--max-steps",
            "1",
        ],
    )

    result = runner.invoke(
        app,
        [
            "memory",
            "search",
            "Python 3.13",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo",
            "--top-k",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Results: 1" in result.output
    assert "#1 score=" in result.output
    assert "我的项目默认使用 Python 3.13 和 uv" in result.output


def test_cli_memory_search_respects_session_id_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app,
        [
            "run",
            "记住：我的项目默认使用 Python 3.13 和 uv",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo-a",
            "--max-steps",
            "1",
        ],
    )
    runner.invoke(
        app,
        [
            "run",
            "记住：我的项目默认使用 Python 3.12",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo-b",
            "--max-steps",
            "1",
        ],
    )

    result = runner.invoke(
        app,
        [
            "memory",
            "search",
            "Python",
            "--memory-path",
            ".memory",
            "--session-id",
            "demo-a",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Python 3.13" in result.output
    assert "Python 3.12" not in result.output


def test_cli_memory_path_outside_workspace_is_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    result = runner.invoke(
        app,
        [
            "memory",
            "list",
            "--memory-path",
            "../outside-memory",
        ],
    )

    assert result.exit_code == 1
    assert "PATH_TRAVERSAL_BLOCKED" in result.output
