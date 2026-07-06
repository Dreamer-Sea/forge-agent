from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rag_quality_document_covers_day2_concepts() -> None:
    text = (ROOT / "docs" / "rag-quality.md").read_text(encoding="utf-8")

    required_phrases = [
        "RAG Quality Evaluation",
        "Hybrid retrieval",
        "Reciprocal Rank Fusion",
        "keyword-overlap",
        "source_hit_rate",
        "recall_at_k",
        "MRR",
        "term_hit_rate",
        "no_answer_accuracy",
        "citation_presence_rate",
        "uv run forge rag eval",
        "no-answer threshold",
    ]

    for phrase in required_phrases:
        assert phrase in text


def test_readme_links_to_rag_quality_document_and_reports() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## RAG Quality Evaluation" in text
    assert "docs/rag-quality.md" in text
    assert "examples/reports/rag-eval-report.md" in text
    assert "examples/reports/rag-eval-results.json" in text
    assert "uv run forge rag eval" in text


def test_committed_rag_eval_report_matches_expected_summary() -> None:
    markdown = (ROOT / "examples" / "reports" / "rag-eval-report.md").read_text(encoding="utf-8")
    data = json.loads(
        (ROOT / "examples" / "reports" / "rag-eval-results.json").read_text(encoding="utf-8")
    )

    assert "# RAG Retrieval Evaluation Report" in markdown
    assert "## Failure Analysis" in markdown
    assert "no_answer_accuracy | 0.00%" in markdown

    assert data["retriever"] == "hybrid"
    assert data["reranker"] == "keyword-overlap"
    assert data["summary"]["total_cases"] == 6
    assert data["summary"]["answerable_cases"] == 5
    assert data["summary"]["no_answer_cases"] == 1
