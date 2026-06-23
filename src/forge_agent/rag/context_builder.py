"""Build prompt-ready RAG context from search results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from forge_agent.rag.citation import Citation
from forge_agent.rag.retriever import SearchResult


@dataclass(frozen=True, slots=True)
class BuiltContext:
    """Prompt-ready RAG context and its citations."""

    context: str
    citations: tuple[Citation, ...]
    used_results: tuple[SearchResult, ...]


class ContextBuilder:
    """Convert ranked search results into grounded prompt context."""

    def __init__(self, max_chars: int = 4_000, max_chunks: int = 5) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")

        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than 0")

        self.max_chars = max_chars
        self.max_chunks = max_chunks

    def build(self, results: Iterable[SearchResult]) -> BuiltContext:
        """Build context with citations, deduplication, and size limits."""
        deduplicated_results = self._deduplicate_results(results)

        context_blocks: list[str] = []
        citations: list[Citation] = []
        used_results: list[SearchResult] = []
        current_chars = 0

        for result in deduplicated_results:
            if len(used_results) >= self.max_chunks:
                break

            citation = Citation.from_chunk(result.chunk)
            block = self._format_block(citation, result)

            remaining_chars = self.max_chars - current_chars
            separator_len = 2 if context_blocks else 0

            if remaining_chars <= separator_len:
                break

            available_for_block = remaining_chars - separator_len

            if len(block) > available_for_block:
                if not context_blocks:
                    block = self._truncate_block(citation, result, available_for_block)
                else:
                    break

            if not block:
                break

            context_blocks.append(block)
            citations.append(citation)
            used_results.append(result)
            current_chars += separator_len + len(block)

        return BuiltContext(
            context="\n\n".join(context_blocks),
            citations=tuple(citations),
            used_results=tuple(used_results),
        )

    def _deduplicate_results(
        self,
        results: Iterable[SearchResult],
    ) -> list[SearchResult]:
        seen_chunk_ids: set[str] = set()
        deduplicated: list[SearchResult] = []

        for result in results:
            chunk_id = result.chunk.metadata.chunk_id

            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)
            deduplicated.append(result)

        return deduplicated

    @staticmethod
    def _format_block(citation: Citation, result: SearchResult) -> str:
        content = result.chunk.content.strip()

        return f"{citation.format()}\n{content}"

    @staticmethod
    def _truncate_block(
        citation: Citation,
        result: SearchResult,
        max_chars: int,
    ) -> str:
        citation_text = citation.format()
        prefix = f"{citation_text}\n"

        if max_chars <= len(prefix):
            return citation_text[:max_chars]

        max_content_chars = max_chars - len(prefix)
        content = result.chunk.content.strip()

        if len(content) <= max_content_chars:
            return f"{prefix}{content}"

        if max_content_chars <= 1:
            return prefix.rstrip()

        return f"{prefix}{content[: max_content_chars - 1].rstrip()}…"
