from __future__ import annotations

import pytest
from typer.testing import CliRunner

from forge_agent.cli import langchain as langchain_cli
from forge_agent.cli.app import app
from forge_agent.integrations.langchain.errors import LangChainIntegrationError

pytest.importorskip("langchain_core")

runner = CliRunner()


def test_cli_langchain_tools_lists_compatible_tools() -> None:
    result = runner.invoke(app, ["langchain", "tools"])

    assert result.exit_code == 0
    assert "LangChain-compatible tools:" in result.output
    assert "- calculator:" in result.output
    assert "- echo_text:" in result.output


def test_cli_langchain_tools_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_dependency(_registry: object) -> object:
        raise LangChainIntegrationError(
            "LangChain integration requires optional dependency. "
            "Install it with: uv sync --extra langchain"
        )

    monkeypatch.setattr(
        langchain_cli,
        "forge_registry_to_langchain_tools",
        raise_missing_dependency,
    )

    result = runner.invoke(app, ["langchain", "tools"])

    assert result.exit_code == 1
    assert "LangChain integration requires optional dependency" in result.output
    assert "uv sync --extra langchain" in result.output


def test_cli_langchain_rag_returns_documents() -> None:
    result = runner.invoke(
        app,
        [
            "langchain",
            "rag",
            "workspace guard permission",
            "--knowledge-base",
            "examples/knowledge_base",
            "--retriever",
            "keyword",
            "--top-k",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "Knowledge base: examples/knowledge_base" in result.output
    assert "Retriever: keyword" in result.output
    assert "Query: workspace guard permission" in result.output
    assert "Documents: 3" in result.output
    assert "source=security.md" in result.output
    assert "chunk_id=" in result.output
    assert "retriever=keyword" in result.output


def test_cli_langchain_rag_rejects_unknown_retriever() -> None:
    result = runner.invoke(
        app,
        [
            "langchain",
            "rag",
            "workspace guard permission",
            "--knowledge-base",
            "examples/knowledge_base",
            "--retriever",
            "unknown",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown retriever: unknown" in result.output
    assert "Supported" in result.output
    assert "retrievers: keyword, vector." in result.output
