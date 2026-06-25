"""Runtime adapter for eval execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from forge_agent.evals.dataset import EvalCase
from forge_agent.evals.runner import EvalRunOutput
from forge_agent.observability import TraceRecorder


class ToolResultLike(Protocol):
    """Protocol for tool results emitted by runtime implementations."""

    tool_name: str
    success: bool


class RuntimeResultLike(Protocol):
    """Protocol for runtime results consumed by eval."""

    final_answer: str | None
    stopped_reason: str
    tool_results: Sequence[ToolResultLike]


class RuntimeLike(Protocol):
    """Protocol for runtime implementations used by eval."""

    def run(self, task: str) -> RuntimeResultLike:
        """Run one task and return a runtime result."""
        ...


class RuntimeEvalExecutor:
    """Adapt an existing agent runtime to EvalCaseExecutor."""

    def __init__(
        self,
        *,
        runtime: object,
        runtime_name: str,
    ) -> None:
        self.runtime = runtime
        self.runtime_name = runtime_name

    async def run_case(
        self,
        case: EvalCase,
        recorder: TraceRecorder,
    ) -> EvalRunOutput:
        """Run one eval case through the configured runtime."""
        runtime = cast(RuntimeLike, self.runtime)

        recorder.record_model_call(
            name=self.runtime_name,
            payload={"input": case.input},
        )

        result = runtime.run(case.input)

        for tool_result in result.tool_results:
            recorder.record_tool_call(
                name=tool_result.tool_name,
                payload={"tool_name": tool_result.tool_name},
            )
            recorder.record_tool_result(
                name=tool_result.tool_name,
                payload={
                    "tool_name": tool_result.tool_name,
                    "success": tool_result.success,
                },
            )

        return EvalRunOutput(
            final_answer=result.final_answer or "",
            stopped_reason=result.stopped_reason,
            sources=[],
        )
