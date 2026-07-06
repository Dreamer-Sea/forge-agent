from __future__ import annotations

from forge_agent.providers.base import ModelMessage, ProviderResponse, ToolCall
from forge_agent.runtime import PlanningRuntime
from forge_agent.tools.registry import ToolRegistry


class TraceFinalProvider:
    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, object]],
    ) -> ProviderResponse:
        return ProviderResponse(final_answer="step done")


class TraceFailureProvider:
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
                content="Need a missing tool.",
                tool_calls=[
                    ToolCall(
                        id="call_missing",
                        name="missing_tool",
                        arguments={},
                    )
                ],
            )

        return ProviderResponse(final_answer="fallback done")


def test_planning_trace_contains_successful_lifecycle_events() -> None:
    runtime = PlanningRuntime(
        provider=TraceFinalProvider(),
        tool_registry=ToolRegistry(),
        max_steps=3,
    )

    result = runtime.run("Explain planning runtime")

    event_types = [event.event_type for event in result.trace_events]

    assert event_types.index("plan_created") < event_types.index(
        "plan_step_started"
    )
    assert event_types.index("plan_step_started") < event_types.index(
        "plan_step_completed"
    )
    assert event_types.index("plan_step_completed") < event_types.index(
        "plan_completed"
    )
    assert event_types[-1] == "runtime_stop"


def test_planning_trace_records_plan_metadata() -> None:
    runtime = PlanningRuntime(
        provider=TraceFinalProvider(),
        tool_registry=ToolRegistry(),
        max_steps=3,
    )

    result = runtime.run("Collect requirements then summarize risks")

    plan_created = next(
        event for event in result.trace_events if event.event_type == "plan_created"
    )

    assert plan_created.data["step_count"] == 2
    assert [step["id"] for step in plan_created.data["steps"]] == [
        "step_1",
        "step_2",
    ]


def test_planning_trace_records_failure_and_replan_metadata() -> None:
    runtime = PlanningRuntime(
        provider=TraceFailureProvider(),
        tool_registry=ToolRegistry(),
        max_steps=5,
    )

    result = runtime.run("Run a tool")

    failed_event = next(
        event
        for event in result.trace_events
        if event.event_type == "plan_step_failed"
    )
    replan_event = next(
        event
        for event in result.trace_events
        if event.event_type == "replan_triggered"
    )

    assert failed_event.data["step_id"] == "answer"
    assert failed_event.data["reason"] == "Unknown tool: missing_tool"
    assert replan_event.data["failed_step_id"] == "answer"
    assert replan_event.data["reason"] == "tool_failed"
    assert replan_event.data["replan_count"] == 1
    assert replan_event.data["replacement_step_ids"] == ["recover_answer"]


def test_planning_trace_tags_model_and_tool_events_with_plan_step_id() -> None:
    runtime = PlanningRuntime(
        provider=TraceFailureProvider(),
        tool_registry=ToolRegistry(),
        max_steps=5,
    )

    result = runtime.run("Run a tool")

    model_call = next(
        event for event in result.trace_events if event.event_type == "model_call"
    )
    tool_result = next(
        event for event in result.trace_events if event.event_type == "tool_result"
    )

    assert model_call.data["plan_step_id"] == "answer"
    assert tool_result.data["plan_step_id"] == "answer"
