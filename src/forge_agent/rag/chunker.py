"""Heading-aware Markdown chunker for local RAG."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from forge_agent.rag.document import Document


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Metadata that makes a chunk traceable to its source document."""

    source_path: str
    relative_path: str
    source_id: str
    title: str
    heading_path: tuple[str, ...]
    chunk_id: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class Chunk:
    """A searchable document chunk."""

    content: str
    metadata: ChunkMetadata


@dataclass(frozen=True, slots=True)
class _Section:
    heading_path: tuple[str, ...]
    lines: tuple[str, ...]


class MarkdownChunker:
    """Split Markdown documents while preserving heading structure."""

    def __init__(self, max_chars: int = 1_200) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")

        self.max_chars = max_chars

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into stable, traceable chunks."""
        sections = self._split_sections(document)
        chunks: list[Chunk] = []

        for section in sections:
            section_text = "\n".join(section.lines).strip()

            if not section_text:
                continue

            for chunk_text in self._split_large_section(section_text):
                ordinal = len(chunks)
                chunk_id = self._build_chunk_id(
                    source_id=document.metadata.source_id,
                    heading_path=section.heading_path,
                    content=chunk_text,
                    ordinal=ordinal,
                )

                metadata = ChunkMetadata(
                    source_path=document.metadata.source_path,
                    relative_path=document.metadata.relative_path,
                    source_id=document.metadata.source_id,
                    title=document.metadata.title,
                    heading_path=section.heading_path,
                    chunk_id=chunk_id,
                    ordinal=ordinal,
                )

                chunks.append(Chunk(content=chunk_text, metadata=metadata))

        return chunks

    def _split_sections(self, document: Document) -> list[_Section]:
        lines = document.content.splitlines()
        sections: list[_Section] = []
        current_lines: list[str] = []
        heading_stack: list[tuple[int, str]] = []
        current_heading_path: tuple[str, ...] = (document.metadata.title,)

        in_code_block = False
        fence_marker: str | None = None

        def flush_current_section() -> None:
            if self._has_meaningful_body(current_lines):
                sections.append(
                    _Section(
                        heading_path=current_heading_path,
                        lines=tuple(current_lines),
                    )
                )

        for line in lines:
            stripped = line.lstrip()

            if self._is_fence(stripped):
                if not in_code_block:
                    in_code_block = True
                    fence_marker = stripped[:3]
                elif fence_marker is not None and stripped.startswith(fence_marker):
                    in_code_block = False
                    fence_marker = None

                current_lines.append(line)
                continue

            if not in_code_block:
                heading = self._parse_heading(line)

                if heading is not None:
                    flush_current_section()

                    level, title = heading
                    heading_stack = [
                        item for item in heading_stack if item[0] < level
                    ]
                    heading_stack.append((level, title))
                    current_heading_path = tuple(
                        heading_title for _, heading_title in heading_stack
                    )
                    current_lines = [line]
                    continue

            current_lines.append(line)

        flush_current_section()

        return sections

    def _split_large_section(self, text: str) -> list[str]:
        blocks = self._split_blocks(text)
        chunks: list[str] = []
        current_blocks: list[str] = []
        current_len = 0

        for block in blocks:
            separator_len = 2 if current_blocks else 0
            projected_len = current_len + separator_len + len(block)

            if current_blocks and projected_len > self.max_chars:
                chunks.append("\n\n".join(current_blocks).strip())
                current_blocks = [block]
                current_len = len(block)
            else:
                current_blocks.append(block)
                current_len = projected_len

        if current_blocks:
            chunks.append("\n\n".join(current_blocks).strip())

        return chunks

    def _split_blocks(self, text: str) -> list[str]:
        blocks: list[str] = []
        current_lines: list[str] = []

        in_code_block = False
        fence_marker: str | None = None

        def flush_current_block() -> None:
            if current_lines:
                block = "\n".join(current_lines).strip()

                if block:
                    blocks.append(block)

                current_lines.clear()

        for line in text.splitlines():
            stripped = line.lstrip()

            if self._is_fence(stripped):
                if not in_code_block:
                    in_code_block = True
                    fence_marker = stripped[:3]
                elif fence_marker is not None and stripped.startswith(fence_marker):
                    in_code_block = False
                    fence_marker = None

                current_lines.append(line)
                continue

            if not in_code_block and not line.strip():
                flush_current_block()
                continue

            current_lines.append(line)

        flush_current_block()

        return blocks

    @classmethod
    def _has_meaningful_body(cls, lines: list[str]) -> bool:
        return any(
            line.strip() and cls._parse_heading(line) is None
            for line in lines
        )

    @staticmethod
    def _parse_heading(line: str) -> tuple[int, str] | None:
        stripped = line.strip()

        if not stripped.startswith("#"):
            return None

        level = 0

        for char in stripped:
            if char == "#":
                level += 1
                continue

            break

        if level == 0 or level > 6:
            return None

        if len(stripped) <= level or stripped[level] != " ":
            return None

        title = stripped[level + 1 :].strip()

        if not title:
            return None

        return level, title

    @staticmethod
    def _is_fence(stripped_line: str) -> bool:
        return stripped_line.startswith("```") or stripped_line.startswith("~~~")

    @staticmethod
    def _build_chunk_id(
        *,
        source_id: str,
        heading_path: tuple[str, ...],
        content: str,
        ordinal: int,
    ) -> str:
        raw = f"{source_id}:{ordinal}:{'/'.join(heading_path)}:{content}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

        return f"{source_id}:chunk:{ordinal}:{digest}"
