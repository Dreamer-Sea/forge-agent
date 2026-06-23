from __future__ import annotations

from pathlib import Path

from forge_agent.rag.knowledge_base import KnowledgeBase


def test_knowledge_base_indexes_markdown_directory(tmp_path: Path) -> None:
    (tmp_path / "agent-runtime.md").write_text(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "Agent Runtime includes Agent Loop, Model Provider, Tool Registry, "
        "Tool Executor, Trace Recorder, and Settings.\n",
        encoding="utf-8",
    )
    (tmp_path / "rag.md").write_text(
        "# Retrieval Augmented Generation\n\n"
        "## Components\n\n"
        "RAG includes Loader, Chunker, Retriever, Context Builder, and Citation.\n",
        encoding="utf-8",
    )

    knowledge_base = KnowledgeBase.from_directory(tmp_path)

    assert len(knowledge_base.index.documents) == 2
    assert len(knowledge_base.index.chunks) >= 2


def test_knowledge_base_search_returns_grounded_context(tmp_path: Path) -> None:
    (tmp_path / "agent-runtime.md").write_text(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "Agent Runtime includes Agent Loop, Model Provider, Tool Registry, "
        "Tool Executor, Trace Recorder, and Settings.\n",
        encoding="utf-8",
    )

    knowledge_base = KnowledgeBase.from_directory(tmp_path)
    search = knowledge_base.search("Agent Runtime core modules", top_k=3)

    assert search.results
    assert search.results[0].chunk.metadata.relative_path == "agent-runtime.md"
    assert "[source: agent-runtime.md#Agent Runtime > Components" in (
        search.built_context.context
    )
    assert "Tool Registry" in search.built_context.context


def test_knowledge_base_search_returns_empty_context_for_no_match(
    tmp_path: Path,
) -> None:
    (tmp_path / "security.md").write_text(
        "# Security\n\n"
        "## Permission System\n\n"
        "Permission System checks tool name and workspace path.\n",
        encoding="utf-8",
    )

    knowledge_base = KnowledgeBase.from_directory(tmp_path)
    search = knowledge_base.search("distributed transaction saga", top_k=3)

    assert search.results == ()
    assert search.built_context.context == ""
    assert search.built_context.citations == ()


def test_day2_evaluation_document_is_retrievable() -> None:
    knowledge_base = KnowledgeBase.from_directory(Path("examples/knowledge_base"))

    search = knowledge_base.search("How does eval work?", top_k=3)

    actual_top_3_sources = [
        result.chunk.metadata.relative_path for result in search.results
    ]

    assert "evaluation.md" in actual_top_3_sources
    assert search.built_context.citations
    assert "evaluation.md" in search.built_context.context
    assert "Eval work follows a repeatable loop" in search.built_context.context
