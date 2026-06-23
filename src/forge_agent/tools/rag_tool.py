from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from forge_agent.rag.knowledge_base import KnowledgeBase
from forge_agent.tools.base import ToolResult, ToolSchema


class SearchKnowledgeBaseArguments(BaseModel):
    """Arguments for searching the local knowledge base."""

    query: str = Field(
        min_length=1,
        description="Search query used to retrieve relevant knowledge chunks.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of chunks to retrieve.",
    )


class SearchKnowledgeBaseTool:
    """Search the local Markdown knowledge base and return grounded context."""

    name = "search_knowledge_base"
    description = (
        "Search the local Markdown knowledge base and return grounded context "
        "with citations."
    )

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._knowledge_base = knowledge_base

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=SearchKnowledgeBaseArguments.model_json_schema(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        try:
            parsed = SearchKnowledgeBaseArguments.model_validate(arguments)
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )

        try:
            search = self._knowledge_base.search(
                parsed.query,
                top_k=parsed.top_k,
            )
        except Exception as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="knowledge_base_search_failed",
                error_message=str(error),
            )

        citations = [
            {
                "source": citation.relative_path,
                "heading_path": list(citation.heading_path),
                "chunk_id": citation.chunk_id,
                "citation": citation.format(),
            }
            for citation in search.built_context.citations
        ]

        chunks = [
            {
                "rank": result.rank,
                "score": result.score,
                "source": result.chunk.metadata.relative_path,
                "heading_path": list(result.chunk.metadata.heading_path),
                "chunk_id": result.chunk.metadata.chunk_id,
                "content": result.chunk.content,
            }
            for result in search.results
        ]

        return ToolResult(
            tool_name=self.name,
            tool_call_id=tool_call_id,
            success=True,
            payload={
                "query": parsed.query,
                "top_k": parsed.top_k,
                "context": search.built_context.context,
                "citations": citations,
                "chunks": chunks,
            },
        )
