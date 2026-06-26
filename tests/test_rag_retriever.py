from __future__ import annotations

from forge_agent.rag.chunker import Chunk, ChunkMetadata
from forge_agent.rag.retriever import KeywordRetriever, expand_query_tokens, tokenize


def make_chunk(
    *,
    content: str,
    relative_path: str,
    heading_path: tuple[str, ...],
    ordinal: int,
) -> Chunk:
    source_id = f"markdown:{relative_path}"

    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            source_path=f"/workspace/examples/knowledge_base/{relative_path}",
            relative_path=relative_path,
            source_id=source_id,
            title=heading_path[0],
            heading_path=heading_path,
            chunk_id=f"{source_id}:chunk:{ordinal}",
            ordinal=ordinal,
        ),
    )


def make_chunks() -> list[Chunk]:
    return [
        make_chunk(
            content=(
                "Agent Runtime includes Agent Loop, Model Provider, "
                "Tool Registry, Tool Executor, Trace Recorder, and Settings."
            ),
            relative_path="agent-runtime.md",
            heading_path=("Agent Runtime", "Components"),
            ordinal=0,
        ),
        make_chunk(
            content=("RAG includes Loader, Chunker, Retriever, Context Builder, and Citation."),
            relative_path="rag.md",
            heading_path=("Retrieval Augmented Generation", "Components"),
            ordinal=0,
        ),
        make_chunk(
            content=(
                "Permission System checks tool name, input arguments, "
                "workspace path, user approval, and runtime policy."
            ),
            relative_path="security.md",
            heading_path=("Security", "Permission System"),
            ordinal=0,
        ),
    ]


def test_retriever_returns_relevant_chunks() -> None:
    retriever = KeywordRetriever(make_chunks())

    results = retriever.search("Agent Runtime core modules", top_k=3)

    assert results
    assert results[0].chunk.metadata.relative_path == "agent-runtime.md"


def test_retriever_orders_by_score() -> None:
    chunks = [
        make_chunk(
            content="Tool Registry registers tools.",
            relative_path="agent-runtime.md",
            heading_path=("Agent Runtime", "ToolRegistry"),
            ordinal=0,
        ),
        make_chunk(
            content="Tool Registry registers tools and exposes tool schemas.",
            relative_path="agent-runtime.md",
            heading_path=("Agent Runtime", "Components", "ToolRegistry"),
            ordinal=1,
        ),
    ]
    retriever = KeywordRetriever(chunks)

    results = retriever.search("tool registry schemas", top_k=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].chunk.metadata.ordinal == 1


def test_retriever_returns_empty_for_no_match() -> None:
    retriever = KeywordRetriever(make_chunks())

    results = retriever.search("distributed transaction saga", top_k=3)

    assert results == []


def test_retriever_includes_rank_and_score() -> None:
    retriever = KeywordRetriever(make_chunks())

    results = retriever.search("citation context builder", top_k=2)

    assert results
    assert results[0].rank == 1
    assert results[0].score > 0
    assert results[0].chunk.metadata.relative_path == "rag.md"


def test_retriever_respects_top_k() -> None:
    retriever = KeywordRetriever(make_chunks())

    results = retriever.search("tool runtime agent rag security", top_k=2)

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_tokenize_supports_english_numbers_and_cjk() -> None:
    tokens = tokenize("Agent Runtime 2026 核心模块")

    assert "agent" in tokens
    assert "runtime" in tokens
    assert "2026" in tokens
    assert "核" in tokens
    assert "心" in tokens


def test_retriever_expands_chinese_core_module_query() -> None:
    retriever = KeywordRetriever(make_chunks())

    results = retriever.search("Agent Runtime 有哪些核心模块？", top_k=3)

    assert results
    assert results[0].chunk.metadata.relative_path == "agent-runtime.md"
    assert results[0].chunk.metadata.heading_path == (
        "Agent Runtime",
        "Components",
    )


def test_expand_query_tokens_adds_demo_aliases() -> None:
    tokens = expand_query_tokens("Agent Runtime 有哪些核心模块？")

    assert "agent" in tokens
    assert "runtime" in tokens
    assert "core" in tokens
    assert "modules" in tokens
    assert "components" in tokens
