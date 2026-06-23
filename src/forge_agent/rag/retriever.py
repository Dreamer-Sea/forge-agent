"""Local keyword retriever for RAG chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from forge_agent.rag.chunker import Chunk

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A ranked retrieval result."""

    chunk: Chunk
    score: float
    rank: int


class Retriever(Protocol):
    """Search interface for local or remote retrieval backends."""

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return ranked chunks for a query."""


class KeywordRetriever:
    """A small BM25-like retriever for demo-level local RAG."""

    def __init__(self, chunks: list[Chunk], default_top_k: int = 5) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than 0")

        self._chunks = chunks
        self._default_top_k = default_top_k
        self._tokenized_chunks = [self._tokenize_chunk(chunk) for chunk in chunks]
        self._term_frequencies = [
            Counter(tokens) for tokens in self._tokenized_chunks
        ]
        self._doc_frequencies = self._build_doc_frequencies()
        self._avg_doc_len = self._calculate_avg_doc_len()

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search chunks and return ranked results."""
        limit = top_k or self._default_top_k

        if limit <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = expand_query_tokens(query)

        if not query_tokens:
            return []

        scored: list[tuple[Chunk, float]] = []

        for chunk, term_frequency, doc_len in zip(
            self._chunks,
            self._term_frequencies,
            self._document_lengths(),
            strict=True,
        ):
            score = self._score_chunk(
                query_tokens=query_tokens,
                term_frequency=term_frequency,
                doc_len=doc_len,
            )

            if score > 0:
                scored.append((chunk, score))

        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].metadata.relative_path,
                item[0].metadata.ordinal,
            )
        )

        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(scored[:limit], start=1)
        ]

    def _score_chunk(
        self,
        *,
        query_tokens: list[str],
        term_frequency: Counter[str],
        doc_len: int,
    ) -> float:
        score = 0.0
        k1 = 1.5
        b = 0.75
        total_docs = len(self._chunks)

        for token in query_tokens:
            frequency = term_frequency[token]

            if frequency == 0:
                continue

            doc_frequency = self._doc_frequencies[token]
            idf = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * doc_len / self._avg_doc_len
            )
            score += idf * (frequency * (k1 + 1)) / denominator

        return score

    def _tokenize_chunk(self, chunk: Chunk) -> list[str]:
        metadata_text = " ".join(
            [
                chunk.metadata.title,
                chunk.metadata.relative_path,
                " ".join(chunk.metadata.heading_path),
            ]
        )
        return tokenize(f"{metadata_text}\n{chunk.content}")

    def _build_doc_frequencies(self) -> Counter[str]:
        doc_frequencies: Counter[str] = Counter()

        for tokens in self._tokenized_chunks:
            doc_frequencies.update(set(tokens))

        return doc_frequencies

    def _calculate_avg_doc_len(self) -> float:
        if not self._tokenized_chunks:
            return 1.0

        total_len = sum(len(tokens) for tokens in self._tokenized_chunks)

        return max(total_len / len(self._tokenized_chunks), 1.0)

    def _document_lengths(self) -> list[int]:
        return [max(len(tokens), 1) for tokens in self._tokenized_chunks]


_QUERY_ALIASES = {
    "核心模块": ["core", "modules", "components"],
    "模块": ["modules", "components"],
    "作用": ["role", "responsible", "purpose"],
    "检索": ["retrieval", "retriever", "search"],
    "引用": ["citation", "citations", "source"],
    "溯源": ["citation", "citations", "source", "traceable"],
    "权限": ["permission", "policy", "allowed"],
    "如何工作": ["checks", "controls", "works"],
    "记录哪些内容": ["record", "records", "events"],
    "事件": ["event", "events"],
}


def expand_query_tokens(query: str) -> list[str]:
    """Tokenize a query and add small demo-level multilingual aliases.

    This is intentionally lightweight. Day 2 uses keyword/BM25 retrieval, so
    common Chinese queries need a few English aliases to match the English
    sample knowledge base.
    """
    tokens = tokenize(query)
    expanded = list(tokens)

    lowered = query.lower()

    for phrase, aliases in _QUERY_ALIASES.items():
        if phrase in lowered:
            expanded.extend(aliases)

    return expanded


def tokenize(text: str) -> list[str]:
    """Tokenize English words, numbers, underscores, and CJK characters."""
    return [
        match.group(0).lower()
        for match in _TOKEN_PATTERN.finditer(text)
    ]
