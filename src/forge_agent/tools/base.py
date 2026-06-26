from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """A normalized schema exposed to model providers."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """A structured result returned by a tool execution."""

    tool_name: str
    success: bool
    tool_call_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    safe_detail: dict[str, Any] = Field(default_factory=dict)


class Tool(Protocol):
    """Tool interface used by the agent runtime."""

    name: str
    description: str

    def schema(self) -> ToolSchema:
        """Return the tool schema."""
        ...

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Execute the tool with validated arguments."""
        ...
