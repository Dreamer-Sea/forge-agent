from forge_agent.rag.chunker import Chunk, ChunkMetadata
from forge_agent.rag.rerankers import KeywordOverlapReranker
from forge_agent.rag.retrievers import SearchResult


def test_keyword_overlap_reranker_promotes_more_relevant_result() -> None:
    runtime = SearchResult(
        chunk=_chunk(
            "runtime",
            "runtime.md",
            "Agent runtime executes model calls and tool calls.",
            1,
        ),
        score=0.99,
        rank=1,
    )
    security = SearchResult(
        chunk=_chunk(
            "security",
            "security.md",
            "Workspace guard checks file tool permissions and blocks path escape.",
            2,
        ),
        score=0.10,
        rank=2,
    )

    reranker = KeywordOverlapReranker()
    reranked = reranker.rerank(
        "workspace guard permission",
        [runtime, security],
    )

    assert [result.chunk.metadata.chunk_id for result in reranked] == [
        "security",
        "runtime",
    ]
    assert [result.rank for result in reranked] == [1, 2]
    assert reranked[0].score == 0.10


def test_keyword_overlap_reranker_respects_top_k() -> None:
    results = [
        SearchResult(
            chunk=_chunk("runtime", "runtime.md", "Agent runtime.", 1),
            score=0.9,
            rank=1,
        ),
        SearchResult(
            chunk=_chunk(
                "security",
                "security.md",
                "Workspace guard permission checks.",
                2,
            ),
            score=0.8,
            rank=2,
        ),
    ]

    reranker = KeywordOverlapReranker()
    reranked = reranker.rerank(
        "workspace guard permission",
        results,
        top_k=1,
    )

    assert len(reranked) == 1
    assert reranked[0].chunk.metadata.chunk_id == "security"
    assert reranked[0].rank == 1


def test_keyword_overlap_reranker_preserves_citation_metadata() -> None:
    result = SearchResult(
        chunk=_chunk(
            "security",
            "security.md",
            "Workspace guard permission checks.",
            3,
        ),
        score=0.8,
        rank=7,
    )

    reranker = KeywordOverlapReranker()
    reranked = reranker.rerank("workspace guard", [result])

    assert reranked[0].chunk.metadata.source_path == "/workspace/security.md"
    assert reranked[0].chunk.metadata.relative_path == "security.md"
    assert reranked[0].chunk.metadata.source_id == "security.md"
    assert reranked[0].chunk.metadata.heading_path == ("Security", "Permission")
    assert reranked[0].chunk.metadata.chunk_id == "security"
    assert reranked[0].chunk.metadata.ordinal == 3


def test_keyword_overlap_reranker_uses_stable_tie_breaking() -> None:
    b = SearchResult(
        chunk=_chunk("b", "b.md", "Permission checks.", 2),
        score=0.8,
        rank=2,
    )
    a = SearchResult(
        chunk=_chunk("a", "a.md", "Permission checks.", 1),
        score=0.9,
        rank=1,
    )

    reranker = KeywordOverlapReranker()
    reranked = reranker.rerank("permission", [b, a])

    assert [result.chunk.metadata.chunk_id for result in reranked] == ["a", "b"]


def test_keyword_overlap_reranker_keeps_order_for_empty_query() -> None:
    results = [
        SearchResult(chunk=_chunk("a", "a.md", "Alpha.", 1), score=0.9, rank=1),
        SearchResult(chunk=_chunk("b", "b.md", "Beta.", 2), score=0.8, rank=2),
    ]

    reranker = KeywordOverlapReranker()
    reranked = reranker.rerank("", results)

    assert [result.chunk.metadata.chunk_id for result in reranked] == ["a", "b"]
    assert [result.rank for result in reranked] == [1, 2]


def test_keyword_overlap_reranker_rejects_invalid_top_k() -> None:
    reranker = KeywordOverlapReranker()

    try:
        reranker.rerank("query", [], top_k=0)
    except ValueError as error:
        assert "top_k" in str(error)
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
            heading_path=("Security", "Permission"),
            chunk_id=chunk_id,
            ordinal=ordinal,
        ),
    )
