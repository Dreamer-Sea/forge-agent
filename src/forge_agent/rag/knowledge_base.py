"""Composable local knowledge base for RAG search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forge_agent.rag.chunker import Chunk, MarkdownChunker
from forge_agent.rag.context_builder import BuiltContext, ContextBuilder
from forge_agent.rag.document import Document
from forge_agent.rag.loader import MarkdownLoader
from forge_agent.rag.retriever import KeywordRetriever, SearchResult


@dataclass(frozen=True, slots=True)
class KnowledgeBaseIndex:
    """In-memory local knowledge base index."""

    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSearch:
    """Knowledge base search output."""

    results: tuple[SearchResult, ...]
    built_context: BuiltContext


class KnowledgeBase:
    """Local Markdown knowledge base backed by keyword retrieval."""

    def __init__(
        self,
        *,
        index: KnowledgeBaseIndex,
        retriever: KeywordRetriever,
        context_builder: ContextBuilder,
    ) -> None:
        self._index = index
        self._retriever = retriever
        self._context_builder = context_builder

    @property
    def index(self) -> KnowledgeBaseIndex:
        """Return index metadata."""
        return self._index

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        chunk_max_chars: int = 1_200,
        context_max_chars: int = 4_000,
        context_max_chunks: int = 5,
        default_top_k: int = 5,
    ) -> KnowledgeBase:
        """Build an in-memory knowledge base from local Markdown files."""
        loader = MarkdownLoader()
        chunker = MarkdownChunker(max_chars=chunk_max_chars)

        documents = loader.load_dir(directory)
        chunks = [chunk for document in documents for chunk in chunker.chunk(document)]

        index = KnowledgeBaseIndex(
            documents=tuple(documents),
            chunks=tuple(chunks),
        )

        return cls(
            index=index,
            retriever=KeywordRetriever(list(chunks), default_top_k=default_top_k),
            context_builder=ContextBuilder(
                max_chars=context_max_chars,
                max_chunks=context_max_chunks,
            ),
        )

    def search(self, query: str, *, top_k: int | None = None) -> KnowledgeBaseSearch:
        """Search the knowledge base and build grounded context."""
        results = self._retriever.search(query, top_k=top_k)
        built_context = self._context_builder.build(results)

        return KnowledgeBaseSearch(
            results=tuple(results),
            built_context=built_context,
        )
