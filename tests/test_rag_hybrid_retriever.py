from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.rag.chunker import Chunk, MarkdownChunker
from forge_agent.rag.loader import MarkdownLoader
from forge_agent.rag.retrievers import HybridRetriever


def test_hybrid_retriever_returns_ranked_chunks(tmp_path: Path) -> None:
    chunks = _load_chunks(tmp_path)
    retriever = HybridRetriever(chunks, default_top_k=3)

    results = retriever.search("workspace guard permission", top_k=3)

    assert results
    assert results[0].rank == 1
    assert results[0].score > 0
    assert results[0].chunk.metadata.relative_path == "security.md"
    assert "Workspace Guard" in results[0].chunk.content


def test_hybrid_retriever_respects_top_k(tmp_path: Path) -> None:
    chunks = _load_chunks(tmp_path)
    retriever = HybridRetriever(chunks, default_top_k=3)

    results = retriever.search("workspace guard permission", top_k=2)

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_hybrid_retriever_rejects_invalid_default_top_k(
    tmp_path: Path,
) -> None:
    chunks = _load_chunks(tmp_path)

    with pytest.raises(ValueError, match="default_top_k must be greater than 0"):
        HybridRetriever(chunks, default_top_k=0)


def test_hybrid_retriever_rejects_invalid_top_k(tmp_path: Path) -> None:
    chunks = _load_chunks(tmp_path)
    retriever = HybridRetriever(chunks, default_top_k=3)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retriever.search("workspace guard permission", top_k=0)


def _load_chunks(tmp_path: Path) -> list[Chunk]:
    (tmp_path / "security.md").write_text(
        "# Security\n\n"
        "## Workspace Guard\n\n"
        "Workspace Guard checks file tool permissions and blocks path escape.\n\n"
        "## Permission System\n\n"
        "Permission System decides whether a tool call is allowed before execution.\n",
        encoding="utf-8",
    )
    (tmp_path / "runtime.md").write_text(
        "# Runtime\n\n## Agent Loop\n\nAgent runtime executes model calls and tool calls.\n",
        encoding="utf-8",
    )

    documents = MarkdownLoader().load_dir(tmp_path)
    chunker = MarkdownChunker(max_chars=500)
    return [chunk for document in documents for chunk in chunker.chunk(document)]
