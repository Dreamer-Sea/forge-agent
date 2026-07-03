from __future__ import annotations

from collections.abc import Sequence

from forge_agent.rag.chunker import Chunk, ChunkMetadata
from forge_agent.rag.retrievers import HybridRetriever, SearchResult


class _FakeRetriever:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.calls.append((query, top_k))
        if top_k is None:
            return list(self._results)
        return list(self._results[:top_k])


def test_hybrid_retriever_fuses_keyword_and_vector_results() -> None:
    runtime = _chunk("runtime", "runtime.md", "Agent runtime tool loop.", 1)
    security = _chunk("security", "security.md", "Workspace guard permission.", 2)
    rag = _chunk("rag", "rag.md", "Retriever context citation.", 3)

    keyword = _FakeRetriever(
        [
            SearchResult(chunk=runtime, score=9.0, rank=1),
            SearchResult(chunk=security, score=5.0, rank=2),
        ]
    )
    vector = _FakeRetriever(
        [
            SearchResult(chunk=security, score=0.9, rank=1),
            SearchResult(chunk=rag, score=0.8, rank=2),
        ]
    )

    retriever = HybridRetriever([keyword, vector])
    results = retriever.search("workspace guard", top_k=3)

    assert [result.chunk.metadata.chunk_id for result in results] == [
        "security",
        "runtime",
        "rag",
    ]
    assert results[0].rank == 1
    assert results[0].score > results[1].score
    assert keyword.calls == [("workspace guard", 3)]
    assert vector.calls == [("workspace guard", 3)]


def test_hybrid_retriever_deduplicates_same_chunk() -> None:
    security = _chunk("security", "security.md", "Workspace guard permission.", 1)

    keyword = _FakeRetriever([SearchResult(chunk=security, score=10.0, rank=1)])
    vector = _FakeRetriever([SearchResult(chunk=security, score=0.9, rank=1)])

    retriever = HybridRetriever([keyword, vector])
    results = retriever.search("workspace guard", top_k=5)

    assert len(results) == 1
    assert results[0].chunk.metadata.chunk_id == "security"
    assert results[0].rank == 1


def test_hybrid_retriever_uses_stable_tie_breaking() -> None:
    a = _chunk("a", "a.md", "Alpha content.", 1)
    b = _chunk("b", "b.md", "Beta content.", 2)

    first = _FakeRetriever(
        [
            SearchResult(chunk=a, score=1.0, rank=1),
            SearchResult(chunk=b, score=1.0, rank=2),
        ]
    )
    second = _FakeRetriever(
        [
            SearchResult(chunk=b, score=1.0, rank=1),
            SearchResult(chunk=a, score=1.0, rank=2),
        ]
    )

    retriever = HybridRetriever([first, second])
    results = retriever.search("content", top_k=2)

    assert [result.chunk.metadata.relative_path for result in results] == [
        "a.md",
        "b.md",
    ]


def test_hybrid_retriever_respects_top_k() -> None:
    chunks = [
        _chunk("a", "a.md", "Alpha.", 1),
        _chunk("b", "b.md", "Beta.", 2),
        _chunk("c", "c.md", "Gamma.", 3),
    ]
    keyword = _FakeRetriever(
        [
            SearchResult(chunk=chunk, score=1.0, rank=rank)
            for rank, chunk in enumerate(chunks, start=1)
        ]
    )
    vector = _FakeRetriever([])

    retriever = HybridRetriever([keyword, vector])
    results = retriever.search("content", top_k=2)

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_hybrid_retriever_rejects_invalid_inputs() -> None:
    chunk = _chunk("a", "a.md", "Alpha.", 1)
    child = _FakeRetriever([SearchResult(chunk=chunk, score=1.0, rank=1)])

    try:
        HybridRetriever([])
    except ValueError as error:
        assert "retrievers" in str(error)
    else:
        raise AssertionError("expected ValueError")

    try:
        HybridRetriever([child], default_top_k=0)
    except ValueError as error:
        assert "default_top_k" in str(error)
    else:
        raise AssertionError("expected ValueError")

    try:
        HybridRetriever([child], rrf_k=0)
    except ValueError as error:
        assert "rrf_k" in str(error)
    else:
        raise AssertionError("expected ValueError")

    retriever = HybridRetriever([child])
    try:
        retriever.search("alpha", top_k=0)
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
