from __future__ import annotations

from pathlib import Path


def test_memory_system_document_covers_day3_core_concepts() -> None:
    content = Path("docs/memory-system.md").read_text(encoding="utf-8")

    required_terms = [
        "memory recall -> context compose -> agent run -> memory write",
        "RAG Knowledge Base",
        "Long-term Memory",
        "semantic",
        "episodic",
        "summary",
        "global",
        "user",
        "session",
        "project",
        "MemoryStore",
        "InMemoryStore",
        "JsonlMemoryStore",
        "MemoryRetriever",
        "MemoryWriter",
        "MemoryWritePolicy",
        "ContextComposer",
        "memory_recall_started",
        "memory_recall_result",
        "memory_write_attempted",
        "memory_write_skipped",
        "memory_write_completed",
        "forge memory list",
        "forge memory search",
    ]

    for term in required_terms:
        assert term in content


def test_readme_links_to_memory_system_document() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "## Long-term Memory" in content
    assert "docs/memory-system.md" in content
    assert "--memory-path .memory" in content
    assert "--session-id demo" in content
    assert "forge memory list" in content
    assert "forge memory search" in content
