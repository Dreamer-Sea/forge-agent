from __future__ import annotations

from typing import Any

from pydantic import ConfigDict

from forge_agent.integrations.langchain.errors import (
    missing_langchain_dependency_error,
)
from forge_agent.rag.knowledge_base import KnowledgeBase
from forge_agent.rag.retrievers import SearchResult


def forge_knowledge_base_to_langchain_retriever(
    knowledge_base: KnowledgeBase,
    *,
    top_k: int | None = None,
    retriever_name: str = "forge-agent",
) -> Any:
    """Convert a forge-agent knowledge base into a LangChain BaseRetriever.

    The adapter keeps forge-agent as the retrieval authority. Chunk content,
    score, rank, source, chunk id, and heading metadata are preserved so
    downstream LangChain chains can still support citation, debug, and eval.
    """
    try:
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except ImportError as error:
        raise missing_langchain_dependency_error() from error

    class _ForgeKnowledgeBaseRetriever(BaseRetriever):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        knowledge_base: KnowledgeBase
        top_k: int | None = None
        retriever_name: str = "forge-agent"

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> list[Document]:
            search = self.knowledge_base.search(query, top_k=self.top_k)
            return [
                search_result_to_langchain_document(
                    result,
                    retriever_name=self.retriever_name,
                )
                for result in search.results
            ]

    return _ForgeKnowledgeBaseRetriever(
        knowledge_base=knowledge_base,
        top_k=top_k,
        retriever_name=retriever_name,
    )


def search_result_to_langchain_document(
    result: SearchResult,
    *,
    retriever_name: str = "forge-agent",
) -> Any:
    """Convert a forge-agent SearchResult into a LangChain Document."""
    try:
        from langchain_core.documents import Document
    except ImportError as error:
        raise missing_langchain_dependency_error() from error

    chunk = result.chunk
    metadata = chunk.metadata

    return Document(
        page_content=chunk.content,
        metadata={
            "source": metadata.relative_path,
            "source_path": metadata.source_path,
            "source_id": metadata.source_id,
            "title": metadata.title,
            "heading_path": list(metadata.heading_path),
            "chunk_id": metadata.chunk_id,
            "ordinal": metadata.ordinal,
            "rank": result.rank,
            "score": result.score,
            "retriever": retriever_name,
        },
    )
