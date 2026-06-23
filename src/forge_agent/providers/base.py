from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ModelMessage(BaseModel):
    """A normalized message used by the agent runtime."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ToolCall(BaseModel):
    """A normalized tool call requested by a model provider."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """A normalized model response returned by a provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class ModelProvider(Protocol):
    """Model provider interface used by the agent runtime."""

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        """Return the next model response."""
        ...