"""Runtime abstractions for pluggable agent backends."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from forge_agent.runtime.events import TraceEvent
from forge_agent.tools.base import ToolResult

RuntimeName = Literal["native", "langgraph"]
StoppedReason = Literal["completed", "max_steps", "error"]

DEFAULT_RUNTIME_NAME: RuntimeName = "native"


class RunConfig(BaseModel):
    """Configuration shared by all runtime implementations."""

    runtime_name: RuntimeName = DEFAULT_RUNTIME_NAME
    max_steps: int = 5
    workspace: Path | None = None
    trace_enabled: bool = True


class RunResult(BaseModel):
    """Stable result shape returned by all runtime implementations."""

    final_answer: str | None
    stopped_reason: StoppedReason
    steps: int
    tool_results: list[ToolResult] = Field(default_factory=list)
    tool_calls: list[ToolResult] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    error_message: str | None = None

    @model_validator(mode="after")
    def sync_tool_result_fields(self) -> Self:
        """Keep old tool_results and new tool_calls fields compatible."""

        if self.tool_results and not self.tool_calls:
            self.tool_calls = list(self.tool_results)

        if self.tool_calls and not self.tool_results:
            self.tool_results = list(self.tool_calls)

        return self


@runtime_checkable
class AgentRuntime(Protocol):
    """Protocol implemented by all runtime backends."""

    def run(
        self,
        user_input: str,
        config: RunConfig | None = None,
    ) -> RunResult:
        """Run one user task and return a stable result shape."""
        ...


def select_runtime_name(runtime_name: RuntimeName | None = None) -> RuntimeName:
    """Resolve a requested runtime name to the default runtime when omitted."""

    return runtime_name or DEFAULT_RUNTIME_NAME
