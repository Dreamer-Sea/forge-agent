from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge_agent.cli.app import app

runner = CliRunner()


def test_cli_rag_eval_runs_keyword_retriever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    dataset = tmp_path / "rag_eval.jsonl"
    _write_security_knowledge_base(knowledge_base)
    _write_eval_dataset(dataset)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "rag",
            "eval",
            "rag_eval.jsonl",
            "--knowledge-base",
            "knowledge_base",
            "--retriever",
            "keyword",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RAG retrieval eval" in result.output
    assert "Retriever: keyword" in result.output
    assert "Reranker: none" in result.output
    assert "Cases: 2" in result.output
    assert "Answerable cases: 1" in result.output
    assert "No-answer cases: 1" in result.output
    assert "source_hit_rate:" in result.output
    assert "no_answer_accuracy:" in result.output
    assert "security_case" in result.output
    assert "no_answer_case" in result.output


def test_cli_rag_eval_runs_hybrid_with_keyword_overlap_reranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    dataset = tmp_path / "rag_eval.jsonl"
    _write_security_knowledge_base(knowledge_base)
    _write_eval_dataset(dataset)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "rag",
            "eval",
            "rag_eval.jsonl",
            "--knowledge-base",
            "knowledge_base",
            "--retriever",
            "hybrid",
            "--reranker",
            "keyword-overlap",
            "--top-k",
            "3",
            "--top-n",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Retriever: hybrid" in result.output
    assert "Reranker: keyword-overlap" in result.output
    assert "Cases: 2" in result.output
    assert "citation_presence_rate:" in result.output


def test_cli_rag_eval_rejects_unknown_reranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    dataset = tmp_path / "rag_eval.jsonl"
    _write_security_knowledge_base(knowledge_base)
    _write_eval_dataset(dataset)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "rag",
            "eval",
            "rag_eval.jsonl",
            "--knowledge-base",
            "knowledge_base",
            "--reranker",
            "cross-encoder",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown reranker" in result.output
    assert "none, identity, keyword-overlap" in result.output


def test_cli_rag_eval_rejects_invalid_top_n(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    dataset = tmp_path / "rag_eval.jsonl"
    _write_security_knowledge_base(knowledge_base)
    _write_eval_dataset(dataset)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "rag",
            "eval",
            "rag_eval.jsonl",
            "--knowledge-base",
            "knowledge_base",
            "--top-n",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "top-n must be greater than 0" in result.output


def _write_security_knowledge_base(path: Path) -> None:
    path.mkdir()
    (path / "security.md").write_text(
        "# Security\n\n"
        "## Workspace Guard\n\n"
        "Workspace guard checks file tool permissions and blocks path escape.\n",
        encoding="utf-8",
    )


def _write_eval_dataset(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                (
                    '{"case_id":"security_case",'
                    '"query":"workspace guard permission",'
                    '"expected_sources":["security.md"],'
                    '"expected_terms":["workspace guard","permission"],'
                    '"top_k":3,'
                    '"expect_no_answer":false}'
                ),
                (
                    '{"case_id":"no_answer_case",'
                    '"query":"distributed transaction saga retries",'
                    '"expected_sources":[],'
                    '"expected_terms":[],'
                    '"top_k":3,'
                    '"expect_no_answer":true}'
                ),
            ]
        ),
        encoding="utf-8",
    )


def test_cli_rag_eval_writes_markdown_and_json_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    dataset = tmp_path / "rag_eval.jsonl"
    report = tmp_path / "reports" / "rag-eval-report.md"
    json_report = tmp_path / "reports" / "rag-eval-results.json"

    _write_security_knowledge_base(knowledge_base)
    _write_eval_dataset(dataset)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "rag",
            "eval",
            "rag_eval.jsonl",
            "--knowledge-base",
            "knowledge_base",
            "--retriever",
            "keyword",
            "--output",
            "reports/rag-eval-report.md",
            "--json-output",
            "reports/rag-eval-results.json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Markdown report written: reports/rag-eval-report.md" in result.output
    assert "JSON report written: reports/rag-eval-results.json" in result.output

    assert report.exists()
    assert json_report.exists()
    assert "# RAG Retrieval Evaluation Report" in report.read_text(encoding="utf-8")

    data = json.loads(json_report.read_text(encoding="utf-8"))
    assert data["retriever"] == "keyword"
    assert data["summary"]["total_cases"] == 2
