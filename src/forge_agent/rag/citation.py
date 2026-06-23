"""Citation primitives for grounded RAG context."""

from __future__ import annotations

from dataclasses import dataclass

from forge_agent.rag.chunker import Chunk


@dataclass(frozen=True, slots=True)
class Citation:
    """A traceable reference to a retrieved chunk."""

    relative_path: str
    heading_path: tuple[str, ...]
    chunk_id: str

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Citation:
        """Build a citation from a chunk."""
        return cls(
            relative_path=chunk.metadata.relative_path,
            heading_path=chunk.metadata.heading_path,
            chunk_id=chunk.metadata.chunk_id,
        )

    def format(self) -> str:
        """Format the citation for prompt context."""
        heading = " > ".join(self.heading_path)

        return f"[source: {self.relative_path}#{heading} {self.chunk_id}]"
