"""Runner for deterministic RAG retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from forge_agent.rag.evals.dataset import load_rag_retrieval_eval_cases
from forge_agent.rag.evals.metrics import (
    RagRetrievalCaseEvaluation,
    RagRetrievalMetricsSummary,
    evaluate_rag_retrieval_case,
    summarize_rag_retrieval_evaluations,
)
from forge_agent.rag.knowledge_base import KnowledgeBase, RetrieverType
from forge_agent.rag.rerankers import (
    IdentityReranker,
    KeywordOverlapReranker,
    Reranker,
)

RerankerName = Literal["none", "identity", "keyword-overlap"]


@dataclass(frozen=True, slots=True)
class RagRetrievalEvalSuite:
    """Complete RAG retrieval eval output."""

    dataset_path: Path
    knowledge_base_path: Path
    retriever_type: RetrieverType
    reranker_name: RerankerName
    evaluations: tuple[RagRetrievalCaseEvaluation, ...]
    summary: RagRetrievalMetricsSummary


def run_rag_retrieval_eval(
    *,
    dataset_path: str | Path,
    knowledge_base_path: str | Path,
    retriever_type: RetrieverType,
    reranker_name: RerankerName = "none",
    top_k: int | None = None,
    top_n: int | None = None,
) -> RagRetrievalEvalSuite:
    """Run deterministic retrieval eval cases against one knowledge base."""
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be greater than 0")

    resolved_dataset_path = Path(dataset_path)
    resolved_knowledge_base_path = Path(knowledge_base_path)

    cases = load_rag_retrieval_eval_cases(resolved_dataset_path)
    max_top_k = max(case.top_k for case in cases)
    search_top_k = top_k if top_k is not None else max_top_k

    knowledge_base = KnowledgeBase.from_directory(
        resolved_knowledge_base_path,
        retriever_type=retriever_type,
        default_top_k=search_top_k,
        context_max_chunks=top_n if top_n is not None else search_top_k,
    )
    reranker = _create_reranker(reranker_name)

    evaluations: list[RagRetrievalCaseEvaluation] = []
    for case in cases:
        effective_case = replace(case, top_k=top_k if top_k is not None else case.top_k)
        search = knowledge_base.search(
            effective_case.query,
            top_k=effective_case.top_k,
        )

        results = list(search.results)
        if reranker is not None:
            results = reranker.rerank(
                effective_case.query,
                results,
                top_k=top_n,
            )
        elif top_n is not None:
            results = results[:top_n]

        evaluations.append(
            evaluate_rag_retrieval_case(
                effective_case,
                results,
                citation_present=bool(results),
            )
        )

    return RagRetrievalEvalSuite(
        dataset_path=resolved_dataset_path,
        knowledge_base_path=resolved_knowledge_base_path,
        retriever_type=retriever_type,
        reranker_name=reranker_name,
        evaluations=tuple(evaluations),
        summary=summarize_rag_retrieval_evaluations(evaluations),
    )


def _create_reranker(reranker_name: RerankerName) -> Reranker | None:
    if reranker_name == "none":
        return None
    if reranker_name == "identity":
        return IdentityReranker()
    if reranker_name == "keyword-overlap":
        return KeywordOverlapReranker()
    raise ValueError(f"Unsupported reranker: {reranker_name}")
