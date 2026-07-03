from forge_agent.rag.chunker import Chunk, ChunkMetadata
from forge_agent.rag.retrievers import VectorRetriever


def test_vector_retriever_returns_ranked_chunks() -> None:
    chunks = [
        _chunk(
            "security",
            "security.md",
            "Workspace guard checks file tool permissions.",
            1,
        ),
        _chunk(
            "runtime",
            "runtime.md",
            "Agent runtime executes model and tool steps.",
            2,
        ),
    ]
    retriever = VectorRetriever(chunks)

    results = retriever.search("workspace guard permission", top_k=2)

    assert results
    assert results[0].rank == 1
    assert results[0].score > 0


def test_vector_retriever_is_deterministic_for_same_query() -> None:
    chunks = [
        _chunk(
            "security",
            "security.md",
            "Workspace guard checks file tool permissions.",
            1,
        ),
        _chunk(
            "runtime",
            "runtime.md",
            "Agent runtime executes model and tool steps.",
            2,
        ),
    ]
    retriever = VectorRetriever(chunks)

    first = retriever.search("workspace guard permission", top_k=2)
    second = retriever.search("workspace guard permission", top_k=2)

    assert [(result.chunk.metadata.chunk_id, result.score) for result in first] == [
        (result.chunk.metadata.chunk_id, result.score) for result in second
    ]


def test_vector_retriever_respects_top_k() -> None:
    chunks = [
        _chunk("a", "a.md", "workspace guard permission", 1),
        _chunk("b", "b.md", "workspace permission policy", 2),
        _chunk("c", "c.md", "agent runtime tool call", 3),
    ]
    retriever = VectorRetriever(chunks)

    results = retriever.search("workspace permission", top_k=1)

    assert len(results) == 1


def test_vector_retriever_rejects_invalid_top_k() -> None:
    retriever = VectorRetriever([_chunk("a", "a.md", "workspace", 1)])

    try:
        retriever.search("workspace", top_k=0)
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
            heading_path=("Test",),
            chunk_id=chunk_id,
            ordinal=ordinal,
        ),
    )
