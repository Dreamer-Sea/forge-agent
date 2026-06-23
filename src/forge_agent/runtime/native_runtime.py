from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from forge_agent.providers.base import ModelMessage, ModelProvider
from forge_agent.runtime.events import TraceEvent
from forge_agent.runtime.state import AgentState
from forge_agent.tools.base import ToolResult
from forge_agent.tools.registry import ToolRegistry

StoppedReason = Literal["completed", "max_steps", "error"]


class AgentRunResult(BaseModel):
    """Structured result returned by an agent runtime run."""

    final_answer: str | None
    stopped_reason: StoppedReason
    steps: int
    tool_results: list[ToolResult] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    error_message: str | None = None


class NativeAgentRuntime:
    """Minimal native agent runtime with model/tool loop."""

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        max_steps: int = 5,
    ) -> None:
        self._provider = provider
        self._tool_registry = tool_registry
        self._max_steps = max_steps

    def run(self, user_input: str) -> AgentRunResult:
        state = AgentState(user_input=user_input)
        state.messages.append(ModelMessage(role="user", content=user_input))

        try:
            for step in range(self._max_steps):
                state.step_index = step

                state.trace_events.append(
                    TraceEvent(
                        event_type="model_call",
                        step=step,
                        data={
                            "message_count": len(state.messages),
                            "tool_count": len(self._tool_registry.schemas()),
                        },
                    )
                )

                response = self._provider.complete(
                    messages=state.messages,
                    tools=self._tool_registry.provider_schemas(),
                )

                state.trace_events.append(
                    TraceEvent(
                        event_type="model_response",
                        step=step,
                        data={
                            "has_tool_calls": response.has_tool_calls,
                            "tool_call_count": len(response.tool_calls),
                            "has_final_answer": response.final_answer is not None,
                        },
                    )
                )

                if response.final_answer is not None:
                    state.trace_events.append(
                        TraceEvent(
                            event_type="final_answer",
                            step=step,
                            data={"final_answer": response.final_answer},
                        )
                    )

                    return AgentRunResult(
                        final_answer=response.final_answer,
                        stopped_reason="completed",
                        steps=step + 1,
                        tool_results=list(state.tool_results),
                        trace_events=list(state.trace_events),
                    )

                if not response.has_tool_calls:
                    if response.content is None:
                        error_message = (
                            "Provider returned empty response without tool calls or final answer."
                        )

                        state.trace_events.append(
                            TraceEvent(
                                event_type="runtime_stop",
                                step=step,
                                data={
                                    "reason": "error",
                                    "error_message": error_message,
                                },
                            )
                        )

                        return AgentRunResult(
                            final_answer=None,
                            stopped_reason="error",
                            steps=step + 1,
                            tool_results=list(state.tool_results),
                            trace_events=list(state.trace_events),
                            error_message=error_message,
                        )

                    final_answer = response.content

                    state.trace_events.append(
                        TraceEvent(
                            event_type="final_answer",
                            step=step,
                            data={"final_answer": final_answer},
                        )
                    )

                    return AgentRunResult(
                        final_answer=final_answer,
                        stopped_reason="completed",
                        steps=step + 1,
                        tool_results=list(state.tool_results),
                        trace_events=list(state.trace_events),
                    )

                if response.content is not None:
                    state.messages.append(
                        ModelMessage(role="assistant", content=response.content)
                    )

                for tool_call in response.tool_calls:
                    state.trace_events.append(
                        TraceEvent(
                            event_type="tool_call",
                            step=step,
                            data={
                                "tool_call_id": tool_call.id,
                                "tool_name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        )
                    )

                    tool_result = self._tool_registry.execute(
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        tool_call_id=tool_call.id,
                    )

                    state.tool_results.append(tool_result)
                    state.messages.append(
                        ModelMessage(
                            role="tool",
                            name=tool_result.tool_name,
                            content=tool_result.model_dump_json(),
                        )
                    )

                    state.trace_events.append(
                        TraceEvent(
                            event_type="tool_result",
                            step=step,
                            data={
                                "tool_call_id": tool_call.id,
                                "tool_name": tool_result.tool_name,
                                "success": tool_result.success,
                                "error_code": tool_result.error_code,
                            },
                        )
                    )

            state.trace_events.append(
                TraceEvent(
                    event_type="runtime_stop",
                    step=self._max_steps,
                    data={"reason": "max_steps"},
                )
            )

            return AgentRunResult(
                final_answer=None,
                stopped_reason="max_steps",
                steps=self._max_steps,
                tool_results=list(state.tool_results),
                trace_events=list(state.trace_events),
            )

        except Exception as error:
            state.trace_events.append(
                TraceEvent(
                    event_type="runtime_stop",
                    step=state.step_index,
                    data={"reason": "error", "error_message": str(error)},
                )
            )

            return AgentRunResult(
                final_answer=None,
                stopped_reason="error",
                steps=state.step_index + 1,
                tool_results=list(state.tool_results),
                trace_events=list(state.trace_events),
                error_message=str(error),
            )