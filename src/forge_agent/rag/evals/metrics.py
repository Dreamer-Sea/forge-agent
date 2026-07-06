"""Deterministic metrics for RAG retrieval evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean

from forge_agent.rag.evals.dataset import RagRetrievalEvalCase
from forge_agent.rag.retrievers import SearchResult


@dataclass(frozen=True, slots=True)
class RagRetrievalCaseEvaluation:
    """Per-case RAG retrieval metric result."""

    case_id: str
    query: str
    expected_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    expected_terms: tuple[str, ...]
    matched_terms: tuple[str, ...]
    top_k: int
    expect_no_answer: bool
    source_hit: bool
    recall_at_k: float
    reciprocal_rank: float
    term_hit_rate: float
    no_answer_correct: bool
    citation_present: bool


@dataclass(frozen=True, slots=True)
class RagRetrievalMetricsSummary:
    """Aggregate RAG retrieval metrics."""

    total_cases: int
    answerable_cases: int
    no_answer_cases: int
    source_hit_rate: float
    recall_at_k: float
    mrr: float
    term_hit_rate: float
    no_answer_accuracy: float
    citation_presence_rate: float


def evaluate_rag_retrieval_case(
    case: RagRetrievalEvalCase,
    results: list[SearchResult],
    *,
    citation_present: bool | None = None,
) -> RagRetrievalCaseEvaluation:
    """Evaluate one RAG retrieval case against ranked search results."""
    limited_results = results[: case.top_k]
    retrieved_sources = tuple(result.chunk.metadata.relative_path for result in limited_results)
    inferred_citation_present = bool(limited_results)
    actual_citation_present = (
        inferred_citation_present if citation_present is None else citation_present
    )

    no_answer_correct = case.expect_no_answer and not limited_results

    source_hit = _source_hit(case.expected_sources, retrieved_sources)
    recall_at_k = _recall_at_k(case.expected_sources, retrieved_sources)
    reciprocal_rank = _reciprocal_rank(case.expected_sources, retrieved_sources)

    matched_terms = _matched_terms(case.expected_terms, limited_results)
    term_hit_rate = len(matched_terms) / len(case.expected_terms) if case.expected_terms else 0.0

    return RagRetrievalCaseEvaluation(
        case_id=case.case_id,
        query=case.query,
        expected_sources=case.expected_sources,
        retrieved_sources=retrieved_sources,
        expected_terms=case.expected_terms,
        matched_terms=matched_terms,
        top_k=case.top_k,
        expect_no_answer=case.expect_no_answer,
        source_hit=source_hit,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        term_hit_rate=term_hit_rate,
        no_answer_correct=no_answer_correct,
        citation_present=actual_citation_present,
    )


def summarize_rag_retrieval_evaluations(
    evaluations: list[RagRetrievalCaseEvaluation],
) -> RagRetrievalMetricsSummary:
    """Aggregate per-case RAG retrieval evaluations."""
    if not evaluations:
        raise ValueError("evaluations must not be empty")

    answerable = [evaluation for evaluation in evaluations if not evaluation.expect_no_answer]
    no_answer = [evaluation for evaluation in evaluations if evaluation.expect_no_answer]

    return RagRetrievalMetricsSummary(
        total_cases=len(evaluations),
        answerable_cases=len(answerable),
        no_answer_cases=len(no_answer),
        source_hit_rate=_average(
            1.0 if evaluation.source_hit else 0.0 for evaluation in answerable
        ),
        recall_at_k=_average(evaluation.recall_at_k for evaluation in answerable),
        mrr=_average(evaluation.reciprocal_rank for evaluation in answerable),
        term_hit_rate=_average(evaluation.term_hit_rate for evaluation in answerable),
        no_answer_accuracy=_average(
            1.0 if evaluation.no_answer_correct else 0.0 for evaluation in no_answer
        ),
        citation_presence_rate=_average(
            1.0 if evaluation.citation_present else 0.0 for evaluation in answerable
        ),
    )


def _source_hit(
    expected_sources: tuple[str, ...],
    retrieved_sources: tuple[str, ...],
) -> bool:
    if not expected_sources:
        return False
    return bool(set(expected_sources) & set(retrieved_sources))


def _recall_at_k(
    expected_sources: tuple[str, ...],
    retrieved_sources: tuple[str, ...],
) -> float:
    if not expected_sources:
        return 0.0

    expected = set(expected_sources)
    retrieved = set(retrieved_sources)

    return len(expected & retrieved) / len(expected)


def _reciprocal_rank(
    expected_sources: tuple[str, ...],
    retrieved_sources: tuple[str, ...],
) -> float:
    if not expected_sources:
        return 0.0

    expected = set(expected_sources)

    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1.0 / rank

    return 0.0


def _matched_terms(
    expected_terms: tuple[str, ...],
    results: list[SearchResult],
) -> tuple[str, ...]:
    if not expected_terms:
        return ()

    haystack_parts: list[str] = []
    for result in results:
        haystack_parts.extend(
            [
                result.chunk.metadata.relative_path,
                result.chunk.metadata.title,
                " ".join(result.chunk.metadata.heading_path),
                result.chunk.content,
            ]
        )

    haystack = "\n".join(haystack_parts).lower()

    return tuple(term for term in expected_terms if term.lower() in haystack)


def _average(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return float(mean(numbers))
