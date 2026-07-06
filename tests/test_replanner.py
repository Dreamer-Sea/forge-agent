from forge_agent.planning import (
    Plan,
    PlanDecision,
    PlanStep,
    ReplanPolicy,
    ReplanReason,
    StepStatus,
)
from forge_agent.runtime.state import AgentState
from forge_agent.tools.base import ToolResult


def assert_step_status(step: PlanStep, expected: StepStatus) -> None:
    assert step.status == expected


def test_replan_policy_replans_after_tool_failure() -> None:
    policy = ReplanPolicy(max_replans=2)
    state = AgentState(user_input="run a tool")
    step = PlanStep(
        id="run_tool",
        description="Run a tool",
        expected_outcome="Tool succeeds",
    )
    plan = Plan(user_input=state.user_input, steps=[step])
    tool_result = ToolResult(
        tool_name="demo_tool",
        success=False,
        error_code="tool_error",
        error_message="tool failed",
    )

    outcome = policy.evaluate_after_step(state, plan, step, tool_result)

    assert outcome.decision == PlanDecision.REPLAN
    assert outcome.reason == ReplanReason.TOOL_FAILED
    assert outcome.replacement_steps[0].id == "recover_run_tool"
    assert_step_status(step, StepStatus.FAILED)
    assert step.failure_reason == "tool failed"


def test_replan_policy_replans_to_safe_explanation_on_permission_denied() -> None:
    policy = ReplanPolicy(max_replans=2)
    state = AgentState(user_input="write outside workspace")
    step = PlanStep(
        id="write_file",
        description="Write outside workspace",
        expected_outcome="File is written",
    )
    plan = Plan(user_input=state.user_input, steps=[step])
    tool_result = ToolResult(
        tool_name="write_file",
        success=False,
        error_code="permission_denied",
        error_message="permission denied",
    )

    outcome = policy.evaluate_after_step(state, plan, step, tool_result)

    assert outcome.decision == PlanDecision.REPLAN
    assert outcome.reason == ReplanReason.PERMISSION_DENIED
    assert outcome.replacement_steps[0].id == "explain_permission_denial"
    assert "do not retry" in outcome.message.lower()
    assert_step_status(step, StepStatus.FAILED)


def test_replan_policy_replans_when_required_evidence_is_empty() -> None:
    policy = ReplanPolicy(max_replans=2)
    state = AgentState(user_input="answer with evidence")
    step = PlanStep(
        id="retrieve_context",
        description="Retrieve context",
        expected_outcome="Context is available",
        evidence_required=True,
    )
    plan = Plan(user_input=state.user_input, steps=[step])
    tool_result = ToolResult(
        tool_name="search_knowledge_base",
        success=True,
        payload={"results": []},
    )

    outcome = policy.evaluate_after_step(state, plan, step, tool_result)

    assert outcome.decision == PlanDecision.REPLAN
    assert outcome.reason == ReplanReason.EMPTY_RETRIEVAL
    assert outcome.replacement_steps[0].id == "answer_without_evidence"
    assert_step_status(step, StepStatus.FAILED)
    assert step.failure_reason == "evidence required but no retrieval result was found"


def test_replan_policy_continues_when_step_observation_is_valid() -> None:
    policy = ReplanPolicy(max_replans=2)
    state = AgentState(user_input="answer with evidence")
    step = PlanStep(
        id="retrieve_context",
        description="Retrieve context",
        expected_outcome="Context is available",
        evidence_required=True,
    )
    plan = Plan(user_input=state.user_input, steps=[step])
    tool_result = ToolResult(
        tool_name="search_knowledge_base",
        success=True,
        payload={"results": [{"source": "docs/planning.md"}]},
    )

    outcome = policy.evaluate_after_step(state, plan, step, tool_result)

    assert outcome.decision == PlanDecision.CONTINUE
    assert outcome.reason is None
    assert outcome.replacement_steps == []
    assert_step_status(step, StepStatus.PENDING)


def test_replan_policy_stops_when_replan_limit_is_reached() -> None:
    policy = ReplanPolicy(max_replans=2)
    state = AgentState(user_input="run a tool", replan_count=2)
    step = PlanStep(
        id="run_tool",
        description="Run a tool",
        expected_outcome="Tool succeeds",
    )
    plan = Plan(user_input=state.user_input, steps=[step])
    tool_result = ToolResult(
        tool_name="demo_tool",
        success=False,
        error_code="tool_error",
        error_message="tool failed",
    )

    outcome = policy.evaluate_after_step(state, plan, step, tool_result)

    assert outcome.decision == PlanDecision.STOP
    assert outcome.reason == ReplanReason.RETRY_LIMIT_REACHED
    assert outcome.replacement_steps == []
    assert_step_status(step, StepStatus.PENDING)


def test_replan_policy_rejects_negative_max_replans() -> None:
    try:
        ReplanPolicy(max_replans=-1)
    except ValueError as error:
        assert str(error) == "max_replans must be non-negative"
    else:
        raise AssertionError("Expected ValueError for negative max_replans")
