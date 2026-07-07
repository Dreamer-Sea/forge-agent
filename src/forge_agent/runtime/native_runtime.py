from __future__ import annotations

from typing import Any

from forge_agent.context import ContextComposer
from forge_agent.memory import (
    MemoryRetriever,
    MemoryScope,
    MemorySearchResult,
    MemoryStore,
    MemoryWriter,
)
from forge_agent.providers.base import ModelMessage, ModelProvider
from forge_agent.runtime.base import RunConfig, RunResult
from forge_agent.runtime.events import TraceEvent, TraceEventType
from forge_agent.runtime.state import AgentState
from forge_agent.security import (
    PATH_OUTSIDE_WORKSPACE,
    PATH_TRAVERSAL_BLOCKED,
    PERMISSION_DENIED,
)
from forge_agent.tools.base import ToolResult
from forge_agent.tools.registry import ToolRegistry

SECURITY_DENIAL_ERROR_CODES = {
    PERMISSION_DENIED,
    PATH_OUTSIDE_WORKSPACE,
    PATH_TRAVERSAL_BLOCKED,
}

PATH_ARGUMENT_KEYS = {
    "path",
    "file_path",
    "directory",
    "workspace",
    "knowledge_base",
    "knowledge_base_path",
}


class AgentRunResult(RunResult):
    """Backward-compatible result returned by the native runtime."""


class NativeAgentRuntime:
    """Minimal native agent runtime with optional long-term memory."""

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        max_steps: int = 5,
        *,
        memory_store: MemoryStore | None = None,
        memory_scope: MemoryScope = MemoryScope.PROJECT,
        memory_scope_id: str | None = None,
        memory_top_k: int = 5,
        memory_min_score: float = 0.0,
        context_max_chars: int = 4_000,
    ) -> None:
        self._provider = provider
        self._tool_registry = tool_registry
        self._max_steps = max_steps
        self._memory_store = memory_store
        self._memory_scope = memory_scope
        self._memory_scope_id = memory_scope_id
        self._memory_top_k = memory_top_k
        self._memory_min_score = memory_min_score
        self._memory_retriever = (
            MemoryRetriever(memory_store) if memory_store is not None else None
        )
        self._memory_writer = MemoryWriter(memory_store) if memory_store is not None else None
        self._context_composer = ContextComposer(max_context_chars=context_max_chars)

    def run(
        self,
        user_input: str,
        config: RunConfig | None = None,
    ) -> AgentRunResult:
        max_steps = config.max_steps if config is not None else self._max_steps
        state = AgentState(user_input=user_input)

        try:
            self._initialize_messages_with_memory(state, user_input)

            for step in range(max_steps):
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
                    self._write_memory_from_user_input(state, user_input)
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
                            "Provider returned empty response without tool calls "
                            "or final answer."
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
                        self._write_memory_from_user_input(state, user_input)
                        self._write_failure_memory(
                            state,
                            failure_reason=error_message,
                            failure_type="runtime_error",
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
                    self._write_memory_from_user_input(state, user_input)
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
                    safe_arguments = _sanitize_trace_arguments(tool_call.arguments)

                    state.trace_events.append(
                        TraceEvent(
                            event_type="tool_call",
                            step=step,
                            data={
                                "tool_call_id": tool_call.id,
                                "tool_name": tool_call.name,
                                "arguments": safe_arguments,
                            },
                        )
                    )
                    state.trace_events.append(
                        TraceEvent(
                            event_type="permission_check",
                            step=step,
                            data={
                                "tool_call_id": tool_call.id,
                                "tool_name": tool_call.name,
                                "arguments": safe_arguments,
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
                                "payload": tool_result.payload,
                                "safe_detail": tool_result.safe_detail,
                            },
                        )
                    )

                    if _is_security_denial(tool_result):
                        state.trace_events.append(
                            TraceEvent(
                                event_type="permission_denied",
                                step=step,
                                data={
                                    "tool_call_id": tool_call.id,
                                    "tool_name": tool_result.tool_name,
                                    "error_code": tool_result.error_code,
                                    "reason": tool_result.payload.get("reason"),
                                    "safe_detail": tool_result.safe_detail,
                                },
                            )
                        )

                    if not tool_result.success:
                        self._write_failure_memory(
                            state,
                            failure_reason=tool_result.error_message
                            or tool_result.safe_detail.get("message", "")
                            or f"{tool_result.tool_name} failed",
                            failure_type="tool_failed",
                        )

            state.trace_events.append(
                TraceEvent(
                    event_type="runtime_stop",
                    step=max_steps,
                    data={"reason": "max_steps"},
                )
            )
            self._write_memory_from_user_input(state, user_input)
            return AgentRunResult(
                final_answer=None,
                stopped_reason="max_steps",
                steps=max_steps,
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
            self._write_failure_memory(
                state,
                failure_reason=str(error),
                failure_type="runtime_error",
            )
            return AgentRunResult(
                final_answer=None,
                stopped_reason="error",
                steps=state.step_index + 1,
                tool_results=list(state.tool_results),
                trace_events=list(state.trace_events),
                error_message=str(error),
            )

    def _initialize_messages_with_memory(
        self,
        state: AgentState,
        user_input: str,
    ) -> None:
        if self._memory_retriever is None:
            state.messages.append(ModelMessage(role="user", content=user_input))
            return

        state.trace_events.append(
            TraceEvent(
                event_type="memory_recall_started",
                step=0,
                data={
                    "query": user_input,
                    "scope": self._memory_scope.value,
                    "scope_id": self._memory_scope_id,
                    "top_k": self._memory_top_k,
                    "min_score": self._memory_min_score,
                },
            )
        )

        memory_results = self._memory_retriever.retrieve(
            user_input,
            top_k=self._memory_top_k,
            scope_filter=self._memory_scope,
            scope_id=self._memory_scope_id,
            min_score=self._memory_min_score,
        )

        composed = self._context_composer.compose(
            user_input=user_input,
            memory_results=memory_results,
        )
        state.messages.extend(composed.messages)

        state.trace_events.append(
            TraceEvent(
                event_type="memory_recall_result",
                step=0,
                data={
                    "result_count": len(memory_results),
                    "memory_count": composed.memory_count,
                    "original_context_chars": composed.original_context_chars,
                    "final_context_chars": composed.final_context_chars,
                    "truncated": composed.truncated,
                    "records": [
                        _memory_result_trace_payload(result)
                        for result in memory_results
                    ],
                },
            )
        )

    def _write_memory_from_user_input(
        self,
        state: AgentState,
        user_input: str,
    ) -> None:
        if self._memory_writer is None:
            return

        state.trace_events.append(
            TraceEvent(
                event_type="memory_write_attempted",
                step=state.step_index,
                data={
                    "source": "user_input",
                    "scope": self._memory_scope.value,
                    "scope_id": self._memory_scope_id,
                },
            )
        )

        decision = self._memory_writer.maybe_write_from_user_input(
            user_input,
            scope=self._memory_scope,
            scope_id=self._memory_scope_id,
        )
        self._append_memory_write_decision_trace(
            state,
            source="user_input",
            decision_should_write=decision.should_write,
            decision_reason=decision.reason,
            decision_metadata=decision.metadata,
        )

    def _write_failure_memory(
        self,
        state: AgentState,
        *,
        failure_reason: str,
        failure_type: str,
    ) -> None:
        if self._memory_writer is None:
            return

        state.trace_events.append(
            TraceEvent(
                event_type="memory_write_attempted",
                step=state.step_index,
                data={
                    "source": failure_type,
                    "scope": self._memory_scope.value,
                    "scope_id": self._memory_scope_id,
                },
            )
        )

        decision = self._memory_writer.maybe_write_failure_memory(
            failure_reason,
            failure_type=failure_type,
            scope=self._memory_scope,
            scope_id=self._memory_scope_id,
        )
        self._append_memory_write_decision_trace(
            state,
            source=failure_type,
            decision_should_write=decision.should_write,
            decision_reason=decision.reason,
            decision_metadata=decision.metadata,
        )

    def _append_memory_write_decision_trace(
        self,
        state: AgentState,
        *,
        source: str,
        decision_should_write: bool,
        decision_reason: str,
        decision_metadata: dict[str, Any],
    ) -> None:
        event_type: TraceEventType = (
            "memory_write_completed" if decision_should_write else "memory_write_skipped"
        )
        state.trace_events.append(
            TraceEvent(
                event_type=event_type,
                step=state.step_index,
                data={
                    "source": source,
                    "reason": decision_reason,
                    "metadata": decision_metadata,
                },
            )
        )


def _memory_result_trace_payload(result: MemorySearchResult) -> dict[str, Any]:
    record = result.record
    return {
        "record_id": record.id,
        "type": record.type.value,
        "scope": record.scope.value,
        "scope_id": record.scope_id,
        "score": result.score,
        "rank": result.rank,
        "reason": result.reason,
    }


def _is_security_denial(tool_result: ToolResult) -> bool:
    return tool_result.error_code in SECURITY_DENIAL_ERROR_CODES


def _sanitize_trace_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive path-like arguments before writing trace events."""

    sanitized: dict[str, Any] = {}

    for key, value in arguments.items():
        if key in PATH_ARGUMENT_KEYS and isinstance(value, str):
            sanitized[key] = _sanitize_path_argument(value)
        else:
            sanitized[key] = value

    return sanitized


def _sanitize_path_argument(value: str) -> str:
    if _looks_like_sensitive_absolute_path(value):
        return ""
    return value


def _looks_like_sensitive_absolute_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("~")
        or _looks_like_windows_absolute_path(value)
    )


def _looks_like_windows_absolute_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] in {"\\", "/"}

