"""Embedding provider abstractions for RAG pipelines."""

from __future__ import annotations

from typing import Protocol

EmbeddingVector = list[float]


class EmbeddingProvider(Protocol):
    """Convert text into deterministic or model-backed vectors."""

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single text."""

    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed multiple texts."""
