"""Vector retriever for RAG chunks."""

from __future__ import annotations

from forge_agent.rag.chunker import Chunk
from forge_agent.rag.embeddings.base import EmbeddingProvider
from forge_agent.rag.embeddings.hashing import HashingEmbeddingProvider
from forge_agent.rag.retrievers.base import SearchResult
from forge_agent.rag.stores.base import VectorStore, VectorStoreItem
from forge_agent.rag.stores.memory import InMemoryVectorStore


class VectorRetriever:
    """Retrieve chunks by vector similarity."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than 0")

        self._default_top_k = default_top_k
        self._embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self._vector_store = vector_store or InMemoryVectorStore()

        vectors = self._embedding_provider.embed_texts(
            [_chunk_to_embedding_text(chunk) for chunk in chunks]
        )
        self._vector_store.add(
            [
                VectorStoreItem(chunk=chunk, vector=vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
        )

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search chunks by query vector similarity."""
        limit = self._default_top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than 0")

        query_vector = self._embedding_provider.embed_text(query)
        vector_results = self._vector_store.similarity_search(
            query_vector,
            top_k=limit,
        )

        return [
            SearchResult(
                chunk=result.chunk,
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(vector_results, start=1)
        ]


def _chunk_to_embedding_text(chunk: Chunk) -> str:
    metadata_text = " ".join(
        [
            chunk.metadata.title,
            chunk.metadata.relative_path,
            " ".join(chunk.metadata.heading_path),
        ]
    )
    return f"{metadata_text}\n{chunk.content}"
