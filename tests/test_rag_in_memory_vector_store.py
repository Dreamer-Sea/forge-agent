from forge_agent.rag.chunker import Chunk, ChunkMetadata
from forge_agent.rag.stores import InMemoryVectorStore, VectorStoreItem


def test_in_memory_vector_store_returns_most_similar_chunk() -> None:
    first = _chunk("a", "security.md", 1)
    second = _chunk("b", "runtime.md", 2)
    store = InMemoryVectorStore()
    store.add(
        [
            VectorStoreItem(chunk=first, vector=[1.0, 0.0]),
            VectorStoreItem(chunk=second, vector=[0.0, 1.0]),
        ]
    )

    results = store.similarity_search([1.0, 0.0], top_k=1)

    assert [result.chunk.metadata.chunk_id for result in results] == ["a"]


def test_in_memory_vector_store_is_deterministic_for_ties() -> None:
    first = _chunk("b", "b.md", 2)
    second = _chunk("a", "a.md", 1)
    store = InMemoryVectorStore()
    store.add(
        [
            VectorStoreItem(chunk=first, vector=[1.0, 0.0]),
            VectorStoreItem(chunk=second, vector=[1.0, 0.0]),
        ]
    )

    results = store.similarity_search([1.0, 0.0], top_k=2)

    assert [result.chunk.metadata.relative_path for result in results] == [
        "a.md",
        "b.md",
    ]


def test_in_memory_vector_store_rejects_invalid_top_k() -> None:
    store = InMemoryVectorStore()

    try:
        store.similarity_search([1.0], top_k=0)
    except ValueError as error:
        assert "top_k" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_in_memory_vector_store_rejects_dimension_mismatch() -> None:
    chunk = _chunk("a", "security.md", 1)
    store = InMemoryVectorStore()
    store.add([VectorStoreItem(chunk=chunk, vector=[1.0, 0.0])])

    try:
        store.similarity_search([1.0], top_k=1)
    except ValueError as error:
        assert "dimensions" in str(error)
    else:
        raise AssertionError("expected ValueError")


def _chunk(chunk_id: str, relative_path: str, ordinal: int) -> Chunk:
    return Chunk(
        content=f"content for {chunk_id}",
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
