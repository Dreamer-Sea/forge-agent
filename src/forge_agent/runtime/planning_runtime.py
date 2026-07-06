from __future__ import annotations

from dataclasses import dataclass

from forge_agent.planning import (
    Plan,
    PlanDecision,
    Planner,
    PlanStep,
    ReplanOutcome,
    ReplanPolicy,
    ReplanReason,
    SimplePlanner,
    StepStatus,
)
from forge_agent.providers.base import ModelMessage, ModelProvider
from forge_agent.runtime.base import RunConfig, RunResult
from forge_agent.runtime.events import TraceEvent
from forge_agent.runtime.native_runtime import _sanitize_trace_arguments
from forge_agent.runtime.state import AgentState
from forge_agent.tools.base import ToolResult
from forge_agent.tools.registry import ToolRegistry


class PlanningRunResult(RunResult):
    """Result returned by PlanningRuntime."""


@dataclass(frozen=True)
class StepExecutionResult:
    """Internal result produced after executing one plan step."""

    success: bool
    model_steps: int
    final_answer: str | None = None
    tool_result: ToolResult | None = None
    error_message: str | None = None


class PlanningRuntime:
    """Plan-Execute-Replan runtime built on top of the provider/tool loop."""

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        planner: Planner | None = None,
        replan_policy: ReplanPolicy | None = None,
        max_steps: int = 5,
    ) -> None:
        self._provider = provider
        self._tool_registry = tool_registry
        self._planner = planner or SimplePlanner()
        self._replan_policy = replan_policy or ReplanPolicy()
        self._max_steps = max_steps

    def run(
        self,
        user_input: str,
        config: RunConfig | None = None,
    ) -> PlanningRunResult:
        max_steps = config.max_steps if config is not None else self._max_steps
        state = AgentState(user_input=user_input)

        try:
            plan = self._planner.create_plan(user_input)
            state.plan = plan
            state.messages.append(ModelMessage(role="user", content=user_input))
            self._record_plan_created(state, plan)

            model_steps_used = 0
            step_answers: list[str] = []

            while model_steps_used < max_steps:
                if plan.is_completed():
                    return self._complete_plan(state, plan, model_steps_used, step_answers)

                step = self._planner.select_next_step(plan, state)
                if step is None:
                    return self._fail_planning(
                        state=state,
                        plan=plan,
                        step_index=model_steps_used,
                        error_message="No executable plan step is available.",
                    )

                state.current_step_id = step.id
                plan.mark_step_running(step.id)
                self._record_plan_step_started(state, plan, step, model_steps_used)

                remaining_steps = max_steps - model_steps_used
                execution = self._execute_step(
                    state=state,
                    step=step,
                    start_step=model_steps_used,
                    max_model_steps=remaining_steps,
                )
                model_steps_used += execution.model_steps

                if execution.success:
                    plan.mark_step_succeeded(step.id)
                    if execution.final_answer is not None:
                        step_answers.append(execution.final_answer)
                    self._record_plan_step_completed(
                        state,
                        plan,
                        step,
                        model_steps_used,
                        execution,
                    )
                    continue

                if model_steps_used >= max_steps and execution.tool_result is None:
                    self._record_plan_step_failed(
                        state,
                        plan,
                        step,
                        model_steps_used,
                        execution.error_message or "Plan step exceeded max steps.",
                    )
                    state.trace_events.append(
                        TraceEvent(
                            event_type="runtime_stop",
                            step=model_steps_used,
                            data={"reason": "max_steps"},
                        )
                    )
                    return PlanningRunResult(
                        final_answer=None,
                        stopped_reason="max_steps",
                        steps=model_steps_used,
                        tool_results=list(state.tool_results),
                        trace_events=list(state.trace_events),
                        error_message=execution.error_message,
                    )

                if execution.tool_result is None:
                    error_message = execution.error_message or "Plan step failed."
                    plan.mark_step_failed(step.id, error_message)
                    self._record_plan_step_failed(
                        state,
                        plan,
                        step,
                        model_steps_used,
                        error_message,
                    )
                    return self._fail_planning(
                        state=state,
                        plan=plan,
                        step_index=model_steps_used,
                        error_message=error_message,
                    )

                outcome = self._replan_policy.evaluate_after_step(
                    state=state,
                    plan=plan,
                    step=step,
                    tool_result=execution.tool_result,
                )
                self._record_plan_step_failed(
                    state,
                    plan,
                    step,
                    model_steps_used,
                    step.failure_reason or execution.error_message or "Plan step failed.",
                )

                if outcome.decision == PlanDecision.STOP:
                    return self._stop_after_replan_limit(
                        state=state,
                        plan=plan,
                        step=step,
                        step_index=model_steps_used,
                        outcome=outcome,
                    )

                if outcome.decision == PlanDecision.REPLAN:
                    self._apply_replan(
                        state=state,
                        plan=plan,
                        failed_step=step,
                        outcome=outcome,
                        step_index=model_steps_used,
                    )
                    continue

                return self._fail_planning(
                    state=state,
                    plan=plan,
                    step_index=model_steps_used,
                    error_message=outcome.message,
                )

            state.trace_events.append(
                TraceEvent(
                    event_type="runtime_stop",
                    step=model_steps_used,
                    data={"reason": "max_steps"},
                )
            )
            return PlanningRunResult(
                final_answer=None,
                stopped_reason="max_steps",
                steps=model_steps_used,
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
            return PlanningRunResult(
                final_answer=None,
                stopped_reason="error",
                steps=state.step_index + 1,
                tool_results=list(state.tool_results),
                trace_events=list(state.trace_events),
                error_message=str(error),
            )

    def _execute_step(
        self,
        state: AgentState,
        step: PlanStep,
        start_step: int,
        max_model_steps: int,
    ) -> StepExecutionResult:
        if max_model_steps <= 0:
            return StepExecutionResult(
                success=False,
                model_steps=0,
                error_message="No model steps remain for plan step execution.",
            )

        state.messages.append(
            ModelMessage(
                role="user",
                content=(
                    "Execute the current plan step.\n"
                    f"Step id: {step.id}\n"
                    f"Description: {step.description}\n"
                    f"Expected outcome: {step.expected_outcome}\n"
                    f"Tool hint: {step.tool_hint or 'none'}\n"
                    f"Evidence required: {step.evidence_required}"
                ),
            )
        )

        for offset in range(max_model_steps):
            runtime_step = start_step + offset
            state.step_index = runtime_step

            state.trace_events.append(
                TraceEvent(
                    event_type="model_call",
                    step=runtime_step,
                    data={
                        "plan_step_id": step.id,
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
                    step=runtime_step,
                    data={
                        "plan_step_id": step.id,
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
                        step=runtime_step,
                        data={
                            "plan_step_id": step.id,
                            "final_answer": response.final_answer,
                        },
                    )
                )
                return StepExecutionResult(
                    success=True,
                    model_steps=offset + 1,
                    final_answer=response.final_answer,
                )

            if not response.has_tool_calls:
                if response.content is None:
                    return StepExecutionResult(
                        success=False,
                        model_steps=offset + 1,
                        error_message=(
                            "Provider returned empty response without tool calls "
                            "or final answer."
                        ),
                    )

                state.trace_events.append(
                    TraceEvent(
                        event_type="final_answer",
                        step=runtime_step,
                        data={
                            "plan_step_id": step.id,
                            "final_answer": response.content,
                        },
                    )
                )
                return StepExecutionResult(
                    success=True,
                    model_steps=offset + 1,
                    final_answer=response.content,
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
                        step=runtime_step,
                        data={
                            "plan_step_id": step.id,
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_call.name,
                            "arguments": safe_arguments,
                        },
                    )
                )
                state.trace_events.append(
                    TraceEvent(
                        event_type="permission_check",
                        step=runtime_step,
                        data={
                            "plan_step_id": step.id,
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
                        step=runtime_step,
                        data={
                            "plan_step_id": step.id,
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_result.tool_name,
                            "success": tool_result.success,
                            "error_code": tool_result.error_code,
                            "payload": tool_result.payload,
                            "safe_detail": tool_result.safe_detail,
                        },
                    )
                )

                if not tool_result.success:
                    return StepExecutionResult(
                        success=False,
                        model_steps=offset + 1,
                        tool_result=tool_result,
                        error_message=tool_result.error_message,
                    )

        return StepExecutionResult(
            success=False,
            model_steps=max_model_steps,
            error_message="Plan step did not finish before max steps was reached.",
        )

    def _record_plan_created(self, state: AgentState, plan: Plan) -> None:
        state.trace_events.append(
            TraceEvent(
                event_type="plan_created",
                step=0,
                data={
                    "plan_id": plan.id,
                    "step_count": len(plan.steps),
                    "steps": [
                        {
                            "id": step.id,
                            "description": step.description,
                            "expected_outcome": step.expected_outcome,
                            "depends_on": step.depends_on,
                            "tool_hint": step.tool_hint,
                            "evidence_required": step.evidence_required,
                        }
                        for step in plan.steps
                    ],
                },
            )
        )

    def _record_plan_step_started(
        self,
        state: AgentState,
        plan: Plan,
        step: PlanStep,
        step_index: int,
    ) -> None:
        state.trace_events.append(
            TraceEvent(
                event_type="plan_step_started",
                step=step_index,
                data={
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "description": step.description,
                    "status": step.status,
                },
            )
        )

    def _record_plan_step_completed(
        self,
        state: AgentState,
        plan: Plan,
        step: PlanStep,
        step_index: int,
        execution: StepExecutionResult,
    ) -> None:
        state.trace_events.append(
            TraceEvent(
                event_type="plan_step_completed",
                step=step_index,
                data={
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "status": step.status,
                    "final_answer": execution.final_answer,
                },
            )
        )

    def _record_plan_step_failed(
        self,
        state: AgentState,
        plan: Plan,
        step: PlanStep,
        step_index: int,
        reason: str,
    ) -> None:
        state.trace_events.append(
            TraceEvent(
                event_type="plan_step_failed",
                step=step_index,
                data={
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "status": step.status,
                    "reason": reason,
                },
            )
        )

    def _record_replan_triggered(
        self,
        state: AgentState,
        plan: Plan,
        failed_step: PlanStep,
        outcome: ReplanOutcome,
        step_index: int,
    ) -> None:
        state.trace_events.append(
            TraceEvent(
                event_type="replan_triggered",
                step=step_index,
                data={
                    "plan_id": plan.id,
                    "failed_step_id": failed_step.id,
                    "reason": outcome.reason,
                    "message": outcome.message,
                    "replacement_step_ids": [
                        replacement.id for replacement in outcome.replacement_steps
                    ],
                    "replan_count": state.replan_count,
                },
            )
        )

    def _record_plan_completed(
        self,
        state: AgentState,
        plan: Plan,
        step_index: int,
    ) -> None:
        state.trace_events.append(
            TraceEvent(
                event_type="plan_completed",
                step=step_index,
                data={
                    "plan_id": plan.id,
                    "status": plan.status,
                    "step_count": len(plan.steps),
                },
            )
        )

    def _apply_replan(
        self,
        state: AgentState,
        plan: Plan,
        failed_step: PlanStep,
        outcome: ReplanOutcome,
        step_index: int,
    ) -> None:
        state.replan_count += 1
        self._make_replacement_step_ids_unique(plan, outcome.replacement_steps)
        failed_step.mark_skipped(
            reason=f"Replanned after failure: {failed_step.failure_reason}"
        )
        plan.steps.extend(outcome.replacement_steps)
        plan.mark_running()
        self._record_replan_triggered(
            state=state,
            plan=plan,
            failed_step=failed_step,
            outcome=outcome,
            step_index=step_index,
        )

    def _make_replacement_step_ids_unique(
        self,
        plan: Plan,
        replacement_steps: list[PlanStep],
    ) -> None:
        existing_step_ids = {step.id for step in plan.steps}

        for replacement in replacement_steps:
            if replacement.id not in existing_step_ids:
                existing_step_ids.add(replacement.id)
                continue

            base_id = replacement.id
            suffix = 1
            while f"{base_id}_{suffix}" in existing_step_ids:
                suffix += 1

            replacement.id = f"{base_id}_{suffix}"
            existing_step_ids.add(replacement.id)

    def _complete_plan(
        self,
        state: AgentState,
        plan: Plan,
        step_index: int,
        step_answers: list[str],
    ) -> PlanningRunResult:
        plan.mark_succeeded()
        final_answer = self._compose_final_answer(step_answers)
        self._record_plan_completed(state, plan, step_index)
        state.trace_events.append(
            TraceEvent(
                event_type="runtime_stop",
                step=step_index,
                data={"reason": "completed"},
            )
        )
        return PlanningRunResult(
            final_answer=final_answer,
            stopped_reason="completed",
            steps=step_index,
            tool_results=list(state.tool_results),
            trace_events=list(state.trace_events),
        )

    def _compose_final_answer(self, step_answers: list[str]) -> str:
        if not step_answers:
            return "Plan completed."

        if len(step_answers) == 1:
            return step_answers[0]

        return "\n".join(
            f"{index}. {answer}" for index, answer in enumerate(step_answers, start=1)
        )

    def _stop_after_replan_limit(
        self,
        state: AgentState,
        plan: Plan,
        step: PlanStep,
        step_index: int,
        outcome: ReplanOutcome,
    ) -> PlanningRunResult:
        if step.status != StepStatus.FAILED:
            step.mark_failed(outcome.message)

        self._record_plan_step_failed(
            state,
            plan,
            step,
            step_index,
            outcome.message,
        )
        state.trace_events.append(
            TraceEvent(
                event_type="runtime_stop",
                step=step_index,
                data={
                    "reason": "replan_limit_reached",
                    "replan_reason": outcome.reason or ReplanReason.RETRY_LIMIT_REACHED,
                },
            )
        )
        return PlanningRunResult(
            final_answer=None,
            stopped_reason="replan_limit_reached",
            steps=step_index,
            tool_results=list(state.tool_results),
            trace_events=list(state.trace_events),
            error_message=outcome.message,
        )

    def _fail_planning(
        self,
        state: AgentState,
        plan: Plan,
        step_index: int,
        error_message: str,
    ) -> PlanningRunResult:
        plan.mark_failed()
        state.trace_events.append(
            TraceEvent(
                event_type="runtime_stop",
                step=step_index,
                data={
                    "reason": "planning_failed",
                    "error_message": error_message,
                },
            )
        )
        return PlanningRunResult(
            final_answer=None,
            stopped_reason="planning_failed",
            steps=step_index,
            tool_results=list(state.tool_results),
            trace_events=list(state.trace_events),
            error_message=error_message,
        )


__all__ = ["PlanningRunResult", "PlanningRuntime"]
