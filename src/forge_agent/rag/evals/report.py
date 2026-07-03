"""Report writers for RAG retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge_agent.rag.evals.runner import RagRetrievalEvalSuite


def rag_retrieval_eval_suite_to_dict(
    suite: RagRetrievalEvalSuite,
) -> dict[str, Any]:
    """Convert a RAG retrieval eval suite to a JSON-serializable dict."""
    summary = suite.summary

    return {
        "dataset_path": _display_path(suite.dataset_path),
        "knowledge_base_path": _display_path(suite.knowledge_base_path),
        "retriever": suite.retriever_type,
        "reranker": suite.reranker_name,
        "summary": {
            "total_cases": summary.total_cases,
            "answerable_cases": summary.answerable_cases,
            "no_answer_cases": summary.no_answer_cases,
            "source_hit_rate": summary.source_hit_rate,
            "recall_at_k": summary.recall_at_k,
            "mrr": summary.mrr,
            "term_hit_rate": summary.term_hit_rate,
            "no_answer_accuracy": summary.no_answer_accuracy,
            "citation_presence_rate": summary.citation_presence_rate,
        },
        "cases": [
            {
                "case_id": evaluation.case_id,
                "query": evaluation.query,
                "expected_sources": list(evaluation.expected_sources),
                "retrieved_sources": list(evaluation.retrieved_sources),
                "expected_terms": list(evaluation.expected_terms),
                "matched_terms": list(evaluation.matched_terms),
                "top_k": evaluation.top_k,
                "expect_no_answer": evaluation.expect_no_answer,
                "source_hit": evaluation.source_hit,
                "recall_at_k": evaluation.recall_at_k,
                "mrr": evaluation.reciprocal_rank,
                "term_hit_rate": evaluation.term_hit_rate,
                "no_answer_correct": evaluation.no_answer_correct,
                "citation_present": evaluation.citation_present,
            }
            for evaluation in suite.evaluations
        ],
    }


def render_rag_retrieval_markdown_report(
    suite: RagRetrievalEvalSuite,
) -> str:
    """Render a Markdown report for a RAG retrieval eval suite."""
    summary = suite.summary
    lines: list[str] = [
        "# RAG Retrieval Evaluation Report",
        "",
        "## Configuration",
        "",
        f"- Dataset: `{_display_path(suite.dataset_path)}`",
        f"- Knowledge base: `{_display_path(suite.knowledge_base_path)}`",
        f"- Retriever: `{suite.retriever_type}`",
        f"- Reranker: `{suite.reranker_name}`",
        f"- Total cases: `{summary.total_cases}`",
        f"- Answerable cases: `{summary.answerable_cases}`",
        f"- No-answer cases: `{summary.no_answer_cases}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| source_hit_rate | {_format_rate(summary.source_hit_rate)} |",
        f"| recall_at_k | {_format_rate(summary.recall_at_k)} |",
        f"| MRR | {_format_rate(summary.mrr)} |",
        f"| term_hit_rate | {_format_rate(summary.term_hit_rate)} |",
        f"| no_answer_accuracy | {_format_rate(summary.no_answer_accuracy)} |",
        (f"| citation_presence_rate | {_format_rate(summary.citation_presence_rate)} |"),
        "",
        "## Case Results",
        "",
        (
            "| Case | Source Hit | Recall@K | MRR | Term Hit Rate | "
            "No-answer Correct | Retrieved Sources |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for evaluation in suite.evaluations:
        retrieved_sources = ", ".join(evaluation.retrieved_sources) or "none"
        lines.append(
            f"| `{evaluation.case_id}` "
            f"| {str(evaluation.source_hit)} "
            f"| {evaluation.recall_at_k:.2f} "
            f"| {evaluation.reciprocal_rank:.2f} "
            f"| {_format_rate(evaluation.term_hit_rate)} "
            f"| {str(evaluation.no_answer_correct)} "
            f"| {retrieved_sources} |"
        )

    lines.extend(
        [
            "",
            "## Failure Analysis",
            "",
        ]
    )

    failures = _failure_notes(suite)
    if failures:
        lines.extend(failures)
    else:
        lines.append("- No failed retrieval cases under the current metrics.")

    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            (
                "- Add a no-answer threshold or abstention policy so retrieval can "
                "return no result when all candidates are weak."
            ),
            (
                "- Replace the deterministic keyword-overlap reranker with a "
                "production reranker such as a cross-encoder, BGE reranker, "
                "Cohere rerank, Jina reranker, or LLM reranker."
            ),
            (
                "- Extend retrieval eval into answer eval with groundedness, "
                "faithfulness, citation correctness, and answer correctness."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_rag_retrieval_markdown_report(
    suite: RagRetrievalEvalSuite,
    path: str | Path,
) -> None:
    """Write a Markdown report for a RAG retrieval eval suite."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_rag_retrieval_markdown_report(suite),
        encoding="utf-8",
    )


def write_rag_retrieval_json_report(
    suite: RagRetrievalEvalSuite,
    path: str | Path,
) -> None:
    """Write a JSON report for a RAG retrieval eval suite."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            rag_retrieval_eval_suite_to_dict(suite),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _failure_notes(suite: RagRetrievalEvalSuite) -> list[str]:
    notes: list[str] = []

    for evaluation in suite.evaluations:
        if evaluation.expect_no_answer and not evaluation.no_answer_correct:
            retrieved_sources = ", ".join(evaluation.retrieved_sources) or "none"
            notes.append(
                f"- `{evaluation.case_id}` expected no answer, but retrieved: "
                f"{retrieved_sources}. This indicates the current retrieval "
                "pipeline does not have an abstention threshold."
            )
            continue

        if not evaluation.expect_no_answer and not evaluation.source_hit:
            expected_sources = ", ".join(evaluation.expected_sources) or "none"
            retrieved_sources = ", ".join(evaluation.retrieved_sources) or "none"
            notes.append(
                f"- `{evaluation.case_id}` missed expected source "
                f"`{expected_sources}`. Retrieved: {retrieved_sources}."
            )
            continue

        if not evaluation.expect_no_answer and evaluation.term_hit_rate < 1.0:
            missing_terms = sorted(set(evaluation.expected_terms) - set(evaluation.matched_terms))
            notes.append(
                f"- `{evaluation.case_id}` hit the expected source but missed "
                f"terms: {', '.join(missing_terms)}."
            )

    return notes


def _display_path(path: Path) -> str:
    resolved_path = path.resolve()
    resolved_cwd = Path.cwd().resolve()
    try:
        return str(resolved_path.relative_to(resolved_cwd))
    except ValueError:
        return str(path)


def _format_rate(value: float) -> str:
    return f"{value * 100:.2f}%"
