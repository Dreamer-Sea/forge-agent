from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from forge_agent.rag.knowledge_base import KnowledgeBase
from forge_agent.security import (
    PermissionAction,
    PermissionDeniedError,
    PermissionPolicy,
    ToolError,
)
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

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        *,
        permission_policy: PermissionPolicy | None = None,
        safe_index_path: str | None = None,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._permission_policy = permission_policy or PermissionPolicy()
        self._safe_index_path = safe_index_path or "<authorized-knowledge-base>"

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

        decision = self._permission_policy.check(PermissionAction.SEARCH_KB)
        if not decision.allowed:
            permission_error = PermissionDeniedError(
                tool_name=self.name,
                reason=decision.reason,
                safe_detail={
                    "action": PermissionAction.SEARCH_KB.value,
                    "index_path": self._safe_index_path,
                },
            )
            return self._error_to_result(
                permission_error,
                tool_call_id=tool_call_id,
            )

        try:
            search = self._knowledge_base.search(
                parsed.query,
                top_k=parsed.top_k,
            )
        except Exception as search_error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="knowledge_base_search_failed",
                error_message=str(search_error),
                safe_detail={"index_path": self._safe_index_path},
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
                "index_path": self._safe_index_path,
                "context": search.built_context.context,
                "citations": citations,
                "chunks": chunks,
            },
        )

    def _error_to_result(
        self,
        error: ToolError,
        *,
        tool_call_id: str | None,
    ) -> ToolResult:
        return ToolResult(
            tool_name=error.tool_name,
            tool_call_id=tool_call_id,
            success=False,
            error_code=error.error_code,
            error_message=error.message,
            payload={"reason": error.reason},
            safe_detail=error.safe_detail,
        )