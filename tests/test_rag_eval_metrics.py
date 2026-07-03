from forge_agent.rag.chunker import Chunk, ChunkMetadata
from forge_agent.rag.evals import (
    RagRetrievalEvalCase,
    evaluate_rag_retrieval_case,
    summarize_rag_retrieval_evaluations,
)
from forge_agent.rag.retrievers import SearchResult


def test_evaluate_rag_retrieval_case_calculates_source_metrics() -> None:
    case = RagRetrievalEvalCase(
        case_id="security_case",
        query="workspace guard permission",
        expected_sources=("security.md",),
        expected_terms=("workspace guard", "permission"),
        top_k=3,
        expect_no_answer=False,
    )
    results = [
        SearchResult(
            chunk=_chunk(
                "runtime",
                "runtime.md",
                "Agent runtime executes tool calls.",
                1,
            ),
            score=0.9,
            rank=1,
        ),
        SearchResult(
            chunk=_chunk(
                "security",
                "security.md",
                "Workspace guard checks permission policy.",
                2,
            ),
            score=0.8,
            rank=2,
        ),
    ]

    evaluation = evaluate_rag_retrieval_case(case, results)

    assert evaluation.source_hit is True
    assert evaluation.recall_at_k == 1.0
    assert evaluation.reciprocal_rank == 0.5
    assert evaluation.term_hit_rate == 1.0
    assert evaluation.matched_terms == ("workspace guard", "permission")
    assert evaluation.citation_present is True


def test_evaluate_rag_retrieval_case_respects_top_k() -> None:
    case = RagRetrievalEvalCase(
        case_id="security_case",
        query="workspace guard permission",
        expected_sources=("security.md",),
        expected_terms=("permission",),
        top_k=1,
        expect_no_answer=False,
    )
    results = [
        SearchResult(
            chunk=_chunk("runtime", "runtime.md", "Agent runtime.", 1),
            score=0.9,
            rank=1,
        ),
        SearchResult(
            chunk=_chunk("security", "security.md", "Permission checks.", 2),
            score=0.8,
            rank=2,
        ),
    ]

    evaluation = evaluate_rag_retrieval_case(case, results)

    assert evaluation.source_hit is False
    assert evaluation.recall_at_k == 0.0
    assert evaluation.reciprocal_rank == 0.0
    assert evaluation.term_hit_rate == 0.0


def test_evaluate_rag_retrieval_case_detects_no_answer_success() -> None:
    case = RagRetrievalEvalCase(
        case_id="no_answer_case",
        query="billing saga retries",
        expected_sources=(),
        expected_terms=(),
        top_k=3,
        expect_no_answer=True,
    )

    evaluation = evaluate_rag_retrieval_case(case, [])

    assert evaluation.no_answer_correct is True
    assert evaluation.retrieved_sources == ()
    assert evaluation.citation_present is False


def test_evaluate_rag_retrieval_case_detects_no_answer_failure() -> None:
    case = RagRetrievalEvalCase(
        case_id="no_answer_case",
        query="billing saga retries",
        expected_sources=(),
        expected_terms=(),
        top_k=3,
        expect_no_answer=True,
    )

    evaluation = evaluate_rag_retrieval_case(
        case,
        [
            SearchResult(
                chunk=_chunk("runtime", "runtime.md", "Agent runtime.", 1),
                score=0.1,
                rank=1,
            )
        ],
    )

    assert evaluation.no_answer_correct is False
    assert evaluation.retrieved_sources == ("runtime.md",)


def test_summarize_rag_retrieval_evaluations_calculates_aggregates() -> None:
    answerable_hit = evaluate_rag_retrieval_case(
        RagRetrievalEvalCase(
            case_id="hit",
            query="permission",
            expected_sources=("security.md",),
            expected_terms=("permission",),
            top_k=3,
            expect_no_answer=False,
        ),
        [
            SearchResult(
                chunk=_chunk("security", "security.md", "Permission policy.", 1),
                score=0.9,
                rank=1,
            )
        ],
    )
    answerable_miss = evaluate_rag_retrieval_case(
        RagRetrievalEvalCase(
            case_id="miss",
            query="runtime",
            expected_sources=("agent-runtime.md",),
            expected_terms=("runtime",),
            top_k=3,
            expect_no_answer=False,
        ),
        [
            SearchResult(
                chunk=_chunk("security", "security.md", "Permission policy.", 1),
                score=0.9,
                rank=1,
            )
        ],
        citation_present=False,
    )
    no_answer_hit = evaluate_rag_retrieval_case(
        RagRetrievalEvalCase(
            case_id="no_answer",
            query="billing saga retries",
            expected_sources=(),
            expected_terms=(),
            top_k=3,
            expect_no_answer=True,
        ),
        [],
    )

    summary = summarize_rag_retrieval_evaluations([answerable_hit, answerable_miss, no_answer_hit])

    assert summary.total_cases == 3
    assert summary.answerable_cases == 2
    assert summary.no_answer_cases == 1
    assert summary.source_hit_rate == 0.5
    assert summary.recall_at_k == 0.5
    assert summary.mrr == 0.5
    assert summary.term_hit_rate == 0.5
    assert summary.no_answer_accuracy == 1.0
    assert summary.citation_presence_rate == 0.5


def test_summarize_rag_retrieval_evaluations_rejects_empty_input() -> None:
    try:
        summarize_rag_retrieval_evaluations([])
    except ValueError as error:
        assert "evaluations" in str(error)
    else:
        raise AssertionError("expected ValueError")


def _chunk(
    chunk_id: str,
    relative_path: str,
    content: str,
    ordinal: int,
) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            source_path=f"/workspace/{relative_path}",
            relative_path=relative_path,
            source_id=relative_path,
            title="Test",
            heading_path=("Test",),
            chunk_id=chunk_id,
            ordinal=ordinal,
        ),
    )
