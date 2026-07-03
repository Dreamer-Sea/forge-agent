from forge_agent.rag.chunker import Chunk, ChunkMetadata
from forge_agent.rag.rerankers import IdentityReranker
from forge_agent.rag.retrievers import SearchResult


def test_identity_reranker_preserves_order() -> None:
    results = [
        SearchResult(chunk=_chunk("a", 1), score=0.9, rank=1),
        SearchResult(chunk=_chunk("b", 2), score=0.8, rank=2),
    ]
    reranker = IdentityReranker()

    reranked = reranker.rerank("query", results)

    assert reranked == results


def test_identity_reranker_respects_top_k() -> None:
    results = [
        SearchResult(chunk=_chunk("a", 1), score=0.9, rank=1),
        SearchResult(chunk=_chunk("b", 2), score=0.8, rank=2),
    ]
    reranker = IdentityReranker()

    reranked = reranker.rerank("query", results, top_k=1)

    assert [result.chunk.metadata.chunk_id for result in reranked] == ["a"]


def test_identity_reranker_rejects_invalid_top_k() -> None:
    reranker = IdentityReranker()

    try:
        reranker.rerank("query", [], top_k=0)
    except ValueError as error:
        assert "top_k" in str(error)
    else:
        raise AssertionError("expected ValueError")


def _chunk(chunk_id: str, ordinal: int) -> Chunk:
    return Chunk(
        content=f"content {chunk_id}",
        metadata=ChunkMetadata(
            source_path=f"/workspace/{chunk_id}.md",
            relative_path=f"{chunk_id}.md",
            source_id=f"{chunk_id}.md",
            title="Test",
            heading_path=("Test",),
            chunk_id=chunk_id,
            ordinal=ordinal,
        ),
    )
