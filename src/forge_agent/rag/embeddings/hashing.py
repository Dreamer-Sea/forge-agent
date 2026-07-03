"""Deterministic hashing embeddings for testable vector retrieval."""

from __future__ import annotations

import hashlib
import math
import re

from forge_agent.rag.embeddings.base import EmbeddingVector

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class HashingEmbeddingProvider:
    """A deterministic embedding provider for CI-stable RAG tests.

    This is not a semantic production embedding model. It uses feature hashing
    to produce stable vectors so the vector retrieval pipeline can be tested
    without network calls, API keys, or local model dependencies.
    """

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Return the vector dimensionality."""
        return self._dimensions

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single text into a normalized hashing vector."""
        vector = [0.0] * self._dimensions
        tokens = _tokenize(text)

        if not tokens:
            return vector

        for token in tokens:
            index = _stable_hash(token) % self._dimensions
            sign = 1.0 if _stable_hash(f"{token}:sign") % 2 == 0 else -1.0
            vector[index] += sign

        return _normalize(vector)

    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed multiple texts into normalized hashing vectors."""
        return [self.embed_text(text) for text in texts]


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def _stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _normalize(vector: EmbeddingVector) -> EmbeddingVector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
