from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.integrations.langchain.retrievers import (
    forge_knowledge_base_to_langchain_retriever,
    search_result_to_langchain_document,
)
from forge_agent.rag.knowledge_base import KnowledgeBase

pytest.importorskip("langchain_core")


def test_search_result_to_langchain_document_preserves_metadata(
    tmp_path: Path,
) -> None:
    knowledge_base = _build_knowledge_base(tmp_path)

    search = knowledge_base.search("workspace guard permission", top_k=1)
    document = search_result_to_langchain_document(
        search.results[0],
        retriever_name="keyword",
    )

    result = search.results[0]
    chunk_metadata = result.chunk.metadata

    assert document.page_content == result.chunk.content
    assert document.metadata["source"] == chunk_metadata.relative_path
    assert document.metadata["source_path"] == chunk_metadata.source_path
    assert document.metadata["source_id"] == chunk_metadata.source_id
    assert document.metadata["title"] == chunk_metadata.title
    assert document.metadata["heading_path"] == list(chunk_metadata.heading_path)
    assert document.metadata["chunk_id"] == chunk_metadata.chunk_id
    assert document.metadata["ordinal"] == chunk_metadata.ordinal
    assert document.metadata["rank"] == result.rank
    assert document.metadata["score"] == result.score
    assert document.metadata["retriever"] == "keyword"


def test_langchain_retriever_returns_documents(tmp_path: Path) -> None:
    knowledge_base = _build_knowledge_base(tmp_path)
    retriever = forge_knowledge_base_to_langchain_retriever(
        knowledge_base,
        top_k=1,
        retriever_name="keyword",
    )

    documents = retriever.invoke("workspace guard permission")

    assert len(documents) == 1
    assert "Permission System" in documents[0].page_content
    assert documents[0].metadata["source"] == "security.md"
    assert documents[0].metadata["chunk_id"]
    assert documents[0].metadata["rank"] == 1
    assert documents[0].metadata["score"] > 0
    assert documents[0].metadata["retriever"] == "keyword"


def test_langchain_retriever_respects_top_k(tmp_path: Path) -> None:
    knowledge_base = _build_knowledge_base(tmp_path)
    retriever = forge_knowledge_base_to_langchain_retriever(
        knowledge_base,
        top_k=2,
        retriever_name="keyword",
    )

    documents = retriever.invoke("workspace guard permission")

    assert len(documents) == 2
    assert [document.metadata["rank"] for document in documents] == [1, 2]


def _build_knowledge_base(tmp_path: Path) -> KnowledgeBase:
    knowledge_base_dir = tmp_path / "knowledge_base"
    knowledge_base_dir.mkdir()

    (knowledge_base_dir / "security.md").write_text(
        """# Security

## Permission System

The permission system decides whether a tool call is allowed before the tool is executed.
It can check tool name, input arguments, workspace path, user approval, and runtime policy.

## Workspace Boundary

The workspace boundary prevents tools from reading or writing files outside the allowed workspace.
It protects local files from unsafe agent tool calls.
""",
        encoding="utf-8",
    )

    return KnowledgeBase.from_directory(
        knowledge_base_dir,
        retriever_type="keyword",
        default_top_k=5,
    )
