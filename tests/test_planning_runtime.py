from __future__ import annotations

from forge_agent.planning import ReplanPolicy
from forge_agent.providers.base import ModelMessage, ProviderResponse, ToolCall
from forge_agent.runtime import (
    AgentRuntime,
    PlanningRunResult,
    PlanningRuntime,
    RunConfig,
)
from forge_agent.tools.registry import ToolRegistry


class FinalAnswerProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, object]],
    ) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(final_answer=f"completed step {self.calls}")


class FailingToolThenFinalProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, object]],
    ) -> ProviderResponse:
        self.calls += 1

        if self.calls == 1:
            return ProviderResponse(
                content="Need to call a tool.",
                tool_calls=[
                    ToolCall(
                        id="call_missing_tool",
                        name="missing_tool",
                        arguments={},
                    )
                ],
            )

        return ProviderResponse(final_answer="Recovered safely.")


def event_types(result: PlanningRunResult) -> list[str]:
    return [event.event_type for event in result.trace_events]


def test_planning_runtime_runs_simple_task_to_completion() -> None:
    provider = FinalAnswerProvider()
    runtime = PlanningRuntime(
        provider=provider,
        tool_registry=ToolRegistry(),
        max_steps=3,
    )

    result = runtime.run("Explain planning runtime")

    assert result.stopped_reason == "completed"
    assert result.final_answer == "completed step 1"
    assert result.steps == 1
    assert provider.calls == 1

    types = event_types(result)
    assert "plan_created" in types
    assert "plan_step_started" in types
    assert "plan_step_completed" in types
    assert "plan_completed" in types


def test_planning_runtime_progresses_multiple_steps() -> None:
    provider = FinalAnswerProvider()
    runtime = PlanningRuntime(
        provider=provider,
        tool_registry=ToolRegistry(),
        max_steps=5,
    )

    result = runtime.run("Collect requirements then summarize risks")

    assert result.stopped_reason == "completed"
    assert result.steps == 2
    assert provider.calls == 2
    assert result.final_answer == "1. completed step 1\n2. completed step 2"

    completed_step_events = [
        event
        for event in result.trace_events
        if event.event_type == "plan_step_completed"
    ]
    assert [event.data["step_id"] for event in completed_step_events] == [
        "step_1",
        "step_2",
    ]


def test_planning_runtime_records_tool_failure_and_replans() -> None:
    provider = FailingToolThenFinalProvider()
    runtime = PlanningRuntime(
        provider=provider,
        tool_registry=ToolRegistry(),
        max_steps=5,
    )

    result = runtime.run("Run a tool")

    assert result.stopped_reason == "completed"
    assert result.final_answer == "Recovered safely."
    assert result.steps == 2
    assert len(result.tool_results) == 1
    assert result.tool_results[0].error_code == "unknown_tool"

    types = event_types(result)
    assert "plan_step_failed" in types
    assert "replan_triggered" in types
    assert "plan_completed" in types


def test_planning_runtime_stops_when_replan_limit_is_reached() -> None:
    provider = FailingToolThenFinalProvider()
    runtime = PlanningRuntime(
        provider=provider,
        tool_registry=ToolRegistry(),
        replan_policy=ReplanPolicy(max_replans=0),
        max_steps=5,
    )

    result = runtime.run("Run a tool")

    assert result.stopped_reason == "replan_limit_reached"
    assert result.final_answer is None
    assert result.error_message is not None
    assert len(result.tool_results) == 1
    assert result.tool_results[0].error_code == "unknown_tool"

    types = event_types(result)
    assert "plan_step_failed" in types
    assert "runtime_stop" in types


def test_planning_runtime_implements_agent_runtime_protocol() -> None:
    runtime = PlanningRuntime(
        provider=FinalAnswerProvider(),
        tool_registry=ToolRegistry(),
    )

    assert isinstance(runtime, AgentRuntime)


def test_planning_runtime_respects_run_config_max_steps() -> None:
    provider = FinalAnswerProvider()
    runtime = PlanningRuntime(
        provider=provider,
        tool_registry=ToolRegistry(),
        max_steps=5,
    )

    result = runtime.run(
        "Collect requirements then summarize risks",
        config=RunConfig(max_steps=1),
    )

    assert result.stopped_reason == "max_steps"
    assert result.steps == 1
