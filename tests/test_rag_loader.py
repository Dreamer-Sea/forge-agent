from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.rag.loader import MarkdownLoader


def test_markdown_loader_loads_documents(tmp_path: Path) -> None:
    markdown_file = tmp_path / "agent-runtime.md"
    markdown_file.write_text(
        "# Agent Runtime\n\n"
        "## Concept\n\n"
        "Agent Runtime coordinates model calls and tool execution.\n",
        encoding="utf-8",
    )

    documents = MarkdownLoader().load_dir(tmp_path)

    assert len(documents) == 1

    document = documents[0]
    assert document.content.startswith("# Agent Runtime")
    assert document.metadata.title == "Agent Runtime"
    assert document.metadata.relative_path == "agent-runtime.md"
    assert document.metadata.source_id == "markdown:agent-runtime.md"
    assert document.metadata.mtime > 0


def test_markdown_loader_ignores_non_markdown_files(tmp_path: Path) -> None:
    markdown_file = tmp_path / "rag.md"
    text_file = tmp_path / "notes.txt"

    markdown_file.write_text("# RAG\n\n## Concept\n\nRetrieval.\n", encoding="utf-8")
    text_file.write_text("This file should be ignored.\n", encoding="utf-8")

    documents = MarkdownLoader().load_dir(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata.relative_path == "rag.md"


def test_markdown_loader_preserves_source_path(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()

    markdown_file = nested_dir / "security.md"
    markdown_file.write_text(
        "# Security\n\n## Concept\n\nSecurity protects the agent runtime.\n",
        encoding="utf-8",
    )

    documents = MarkdownLoader().load_dir(tmp_path)

    assert len(documents) == 1

    document = documents[0]
    assert document.metadata.source_path == markdown_file.as_posix()
    assert document.metadata.relative_path == "nested/security.md"
    assert document.metadata.title == "Security"


def test_markdown_loader_uses_file_stem_when_title_is_missing(
    tmp_path: Path,
) -> None:
    markdown_file = tmp_path / "untitled-doc.md"
    markdown_file.write_text("## Concept\n\nNo H1 title here.\n", encoding="utf-8")

    documents = MarkdownLoader().load_dir(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata.title == "untitled-doc"


def test_markdown_loader_rejects_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        MarkdownLoader().load_dir(missing_dir)


def test_markdown_loader_rejects_non_directory(tmp_path: Path) -> None:
    markdown_file = tmp_path / "rag.md"
    markdown_file.write_text("# RAG\n", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        MarkdownLoader().load_dir(markdown_file)
