from __future__ import annotations

from forge_agent.rag.chunker import MarkdownChunker
from forge_agent.rag.document import Document, DocumentMetadata


def make_document(content: str) -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(
            source_path="/workspace/examples/knowledge_base/agent-runtime.md",
            relative_path="agent-runtime.md",
            source_id="markdown:agent-runtime.md",
            title="Agent Runtime",
            mtime=1_700_000_000.0,
        ),
    )


def test_chunker_preserves_source_metadata() -> None:
    document = make_document(
        "# Agent Runtime\n\n"
        "## Concept\n\n"
        "Agent Runtime coordinates model calls and tool execution.\n"
    )

    chunks = MarkdownChunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].metadata.source_path == document.metadata.source_path
    assert chunks[0].metadata.relative_path == document.metadata.relative_path
    assert chunks[0].metadata.source_id == document.metadata.source_id
    assert chunks[0].metadata.title == document.metadata.title


def test_chunker_preserves_heading_path() -> None:
    document = make_document(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "The runtime has several components.\n\n"
        "### ToolRegistry\n\n"
        "ToolRegistry stores tools and exposes their schemas.\n"
    )

    chunks = MarkdownChunker().chunk(document)

    tool_registry_chunk = next(
        chunk for chunk in chunks if "ToolRegistry stores tools" in chunk.content
    )

    assert tool_registry_chunk.metadata.heading_path == (
        "Agent Runtime",
        "Components",
        "ToolRegistry",
    )


def test_chunker_keeps_code_block_together() -> None:
    document = make_document(
        "# Agent Runtime\n\n"
        "## Design Notes\n\n"
        "A tool call should be represented as structured data.\n\n"
        "```python\n"
        "def call_tool(name: str, arguments: dict[str, object]) -> object:\n"
        '    return {"tool": name, "arguments": arguments}\n'
        "```\n\n"
        "The runtime records the tool result after execution.\n"
    )

    chunks = MarkdownChunker(max_chars=80).chunk(document)

    code_chunks = [chunk for chunk in chunks if "def call_tool" in chunk.content]

    assert len(code_chunks) == 1
    assert "```python" in code_chunks[0].content
    assert "return" in code_chunks[0].content
    assert "```" in code_chunks[0].content


def test_chunker_generates_stable_chunk_ids() -> None:
    document = make_document(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "Agent Loop coordinates model calls and tool execution.\n\n"
        "Tool Registry stores available tools.\n"
    )

    first_run = MarkdownChunker(max_chars=80).chunk(document)
    second_run = MarkdownChunker(max_chars=80).chunk(document)

    first_ids = [chunk.metadata.chunk_id for chunk in first_run]
    second_ids = [chunk.metadata.chunk_id for chunk in second_run]

    assert first_ids == second_ids


def test_chunker_splits_large_sections_by_blocks() -> None:
    document = make_document(
        "# Agent Runtime\n\n"
        "## Components\n\n"
        "Agent Loop coordinates model calls and tool execution.\n\n"
        "Model Provider abstracts model-specific APIs.\n\n"
        "Tool Registry stores available tools and schemas.\n"
    )

    chunks = MarkdownChunker(max_chars=90).chunk(document)

    assert len(chunks) >= 2
    assert all(chunk.content for chunk in chunks)
    assert [chunk.metadata.ordinal for chunk in chunks] == list(range(len(chunks)))
