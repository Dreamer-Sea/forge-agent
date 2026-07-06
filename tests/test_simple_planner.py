from forge_agent.planning import Planner, PlanningContext, SimplePlanner, StepStatus


def test_simple_planner_creates_single_answer_step_for_plain_task() -> None:
    planner = SimplePlanner()

    plan = planner.create_plan("Explain what a planning runtime is")

    assert plan.user_input == "Explain what a planning runtime is"
    assert len(plan.steps) == 1

    step = plan.steps[0]
    assert step.id == "answer"
    assert step.description == "Explain what a planning runtime is"
    assert step.expected_outcome == "A direct answer is produced."
    assert step.status == StepStatus.PENDING
    assert step.tool_hint is None
    assert step.evidence_required is False


def test_simple_planner_splits_compound_task_into_multiple_steps() -> None:
    planner = SimplePlanner()

    plan = planner.create_plan(
        "Collect requirements then propose design and summarize risks"
    )

    assert [step.id for step in plan.steps] == ["step_1", "step_2", "step_3"]
    assert [step.description for step in plan.steps] == [
        "Collect requirements",
        "propose design",
        "summarize risks",
    ]


def test_simple_planner_creates_rag_steps_for_knowledge_base_task() -> None:
    planner = SimplePlanner()

    plan = planner.create_plan("总结知识库中的 Workspace Guard 设计")

    assert [step.id for step in plan.steps] == [
        "retrieve_context",
        "answer_with_context",
    ]

    retrieve_step = plan.steps[0]
    answer_step = plan.steps[1]

    assert retrieve_step.tool_hint == "search_knowledge_base"
    assert retrieve_step.evidence_required is True
    assert answer_step.depends_on == ["retrieve_context"]


def test_simple_planner_can_use_context_to_create_rag_plan() -> None:
    planner = SimplePlanner()

    plan = planner.create_plan(
        "Summarize the design",
        context=PlanningContext(knowledge_base_enabled=True),
    )

    assert [step.id for step in plan.steps] == [
        "retrieve_context",
        "answer_with_context",
    ]


def test_simple_planner_creates_file_steps_for_workspace_task() -> None:
    planner = SimplePlanner()

    plan = planner.create_plan("Read file README.md and summarize it")

    assert [step.id for step in plan.steps] == [
        "inspect_workspace",
        "execute_operation",
        "summarize_result",
    ]
    assert plan.steps[0].tool_hint == "list_files"
    assert plan.steps[1].depends_on == ["inspect_workspace"]
    assert plan.steps[2].depends_on == ["execute_operation"]


def test_simple_planner_selects_next_executable_step() -> None:
    planner = SimplePlanner()
    plan = planner.create_plan("Read file README.md and summarize it")
    state = __import__(
        "forge_agent.runtime.state",
        fromlist=["AgentState"],
    ).AgentState(user_input=plan.user_input)

    next_step = planner.select_next_step(plan, state)

    assert next_step is not None
    assert next_step.id == "inspect_workspace"

    plan.mark_step_succeeded("inspect_workspace")
    next_step = planner.select_next_step(plan, state)

    assert next_step is not None
    assert next_step.id == "execute_operation"


def test_simple_planner_satisfies_planner_protocol() -> None:
    planner: Planner = SimplePlanner()

    plan = planner.create_plan("Explain task planning")

    assert len(plan.steps) == 1
    assert plan.steps[0].id == "answer"
