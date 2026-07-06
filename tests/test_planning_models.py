import pytest

from forge_agent.planning import Plan, PlanStatus, PlanStep, StepStatus


def assert_step_status(step: PlanStep, expected: StepStatus) -> None:
    assert step.status == expected


def assert_plan_status(plan: Plan, expected: PlanStatus) -> None:
    assert plan.status == expected


def test_plan_can_contain_multiple_steps() -> None:
    plan = Plan(
        user_input="summarize workspace guard and check security boundaries",
        steps=[
            PlanStep(
                id="step_1",
                description="Retrieve workspace guard context",
                expected_outcome="Relevant workspace guard context is collected",
            ),
            PlanStep(
                id="step_2",
                description="Summarize security boundaries",
                expected_outcome="Security boundaries are summarized",
                depends_on=["step_1"],
            ),
        ],
    )

    assert plan.user_input == "summarize workspace guard and check security boundaries"
    assert len(plan.steps) == 2
    assert_plan_status(plan, PlanStatus.PENDING)


def test_plan_step_default_status_is_pending() -> None:
    step = PlanStep(
        description="Answer the user question",
        expected_outcome="A final answer is produced",
    )

    assert_step_status(step, StepStatus.PENDING)
    assert step.failure_reason is None


def test_plan_step_can_transition_from_pending_to_running_to_succeeded() -> None:
    step = PlanStep(
        description="Inspect workspace",
        expected_outcome="Workspace files are inspected",
    )

    step.mark_running()
    assert_step_status(step, StepStatus.RUNNING)

    step.mark_succeeded()
    assert_step_status(step, StepStatus.SUCCEEDED)
    assert step.failure_reason is None


def test_failed_step_can_record_failure_reason() -> None:
    step = PlanStep(
        description="Read protected file",
        expected_outcome="File content is read",
    )

    step.mark_failed("permission denied")

    assert_step_status(step, StepStatus.FAILED)
    assert step.failure_reason == "permission denied"


def test_plan_can_detect_all_steps_completed() -> None:
    plan = Plan(
        user_input="complete all steps",
        steps=[
            PlanStep(
                id="step_1",
                description="First step",
                expected_outcome="First result",
            ),
            PlanStep(
                id="step_2",
                description="Second step",
                expected_outcome="Second result",
            ),
        ],
    )

    assert plan.is_completed() is False

    plan.mark_step_succeeded("step_1")
    assert plan.is_completed() is False
    assert_plan_status(plan, PlanStatus.RUNNING)

    plan.mark_step_succeeded("step_2")
    assert plan.is_completed() is True
    assert_plan_status(plan, PlanStatus.SUCCEEDED)


def test_plan_can_return_next_executable_step() -> None:
    plan = Plan(
        user_input="retrieve then answer",
        steps=[
            PlanStep(
                id="retrieve",
                description="Retrieve context",
                expected_outcome="Context is available",
            ),
            PlanStep(
                id="answer",
                description="Answer with context",
                expected_outcome="Grounded answer is produced",
                depends_on=["retrieve"],
            ),
        ],
    )

    next_step = plan.next_executable_step()
    assert next_step is not None
    assert next_step.id == "retrieve"

    plan.mark_step_succeeded("retrieve")

    next_step = plan.next_executable_step()
    assert next_step is not None
    assert next_step.id == "answer"

    plan.mark_step_succeeded("answer")

    assert plan.next_executable_step() is None


def test_unknown_step_raises_error() -> None:
    plan = Plan(
        user_input="unknown step",
        steps=[
            PlanStep(
                id="known",
                description="Known step",
                expected_outcome="Known result",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Unknown plan step: missing"):
        plan.mark_step_succeeded("missing")
