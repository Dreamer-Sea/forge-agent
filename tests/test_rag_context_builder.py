from __future__ import annotations

from forge_agent.rag.chunker import Chunk, ChunkMetadata
from forge_agent.rag.citation import Citation
from forge_agent.rag.context_builder import ContextBuilder
from forge_agent.rag.retriever import SearchResult


def make_chunk(
    *,
    content: str,
    relative_path: str = "agent-runtime.md",
    heading_path: tuple[str, ...] = ("Agent Runtime", "Components"),
    chunk_id: str = "markdown:agent-runtime.md:chunk:0",
    ordinal: int = 0,
) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            source_path=f"/workspace/examples/knowledge_base/{relative_path}",
            relative_path=relative_path,
            source_id=f"markdown:{relative_path}",
            title=heading_path[0],
            heading_path=heading_path,
            chunk_id=chunk_id,
            ordinal=ordinal,
        ),
    )


def make_result(
    *,
    content: str,
    relative_path: str = "agent-runtime.md",
    heading_path: tuple[str, ...] = ("Agent Runtime", "Components"),
    chunk_id: str = "markdown:agent-runtime.md:chunk:0",
    ordinal: int = 0,
    rank: int = 1,
    score: float = 1.0,
) -> SearchResult:
    return SearchResult(
        chunk=make_chunk(
            content=content,
            relative_path=relative_path,
            heading_path=heading_path,
            chunk_id=chunk_id,
            ordinal=ordinal,
        ),
        rank=rank,
        score=score,
    )


def test_context_builder_includes_citations() -> None:
    result = make_result(
        content="Agent Runtime includes Agent Loop and Tool Registry.",
    )

    built = ContextBuilder(max_chars=1_000, max_chunks=3).build([result])

    assert "[source: agent-runtime.md#Agent Runtime > Components" in built.context
    assert "markdown:agent-runtime.md:chunk:0" in built.context
    assert "Agent Runtime includes Agent Loop" in built.context
    assert len(built.citations) == 1


def test_context_builder_respects_max_chars() -> None:
    result = make_result(
        content=(
            "Agent Runtime includes Agent Loop, Model Provider, Tool Registry, "
            "Tool Executor, Trace Recorder, Settings, and runtime limits."
        )
    )

    built = ContextBuilder(max_chars=120, max_chunks=3).build([result])

    assert len(built.context) <= 120
    assert "[source:" in built.context


def test_context_builder_deduplicates_chunks() -> None:
    first = make_result(
        content="Tool Registry stores available tools.",
        chunk_id="markdown:agent-runtime.md:chunk:1",
        rank=1,
    )
    duplicate = make_result(
        content="Tool Registry stores available tools.",
        chunk_id="markdown:agent-runtime.md:chunk:1",
        rank=2,
    )

    built = ContextBuilder(max_chars=1_000, max_chunks=3).build(
        [first, duplicate]
    )

    assert len(built.used_results) == 1
    assert built.context.count("Tool Registry stores available tools.") == 1


def test_citation_formats_source_and_heading() -> None:
    chunk = make_chunk(
        content="Citation keeps answers traceable.",
        relative_path="rag.md",
        heading_path=("Retrieval Augmented Generation", "Components", "Citation"),
        chunk_id="markdown:rag.md:chunk:2",
        ordinal=2,
    )

    citation = Citation.from_chunk(chunk)

    assert citation.format() == (
        "[source: rag.md#Retrieval Augmented Generation > Components > "
        "Citation markdown:rag.md:chunk:2]"
    )


def test_context_builder_respects_max_chunks() -> None:
    first = make_result(
        content="Agent Runtime includes Agent Loop.",
        chunk_id="markdown:agent-runtime.md:chunk:0",
        rank=1,
    )
    second = make_result(
        content="RAG includes Loader, Chunker, and Retriever.",
        relative_path="rag.md",
        heading_path=("Retrieval Augmented Generation", "Components"),
        chunk_id="markdown:rag.md:chunk:0",
        rank=2,
    )

    built = ContextBuilder(max_chars=1_000, max_chunks=1).build([first, second])

    assert len(built.used_results) == 1
    assert "Agent Runtime includes Agent Loop." in built.context
    assert "RAG includes Loader" not in built.context
