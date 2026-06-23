"""Document data structures for the local RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata that makes a loaded document traceable."""

    source_path: str
    relative_path: str
    source_id: str
    title: str
    mtime: float


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded source document."""

    content: str
    metadata: DocumentMetadata
