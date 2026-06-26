from __future__ import annotations

import pytest

from forge_agent.evals import EvalCase, EvalDataset, EvalRunner, EvalRunOutput
from forge_agent.observability import TraceRecorder

pytestmark = pytest.mark.anyio


class FakeEvalExecutor:
    async def run_case(
        self,
        case: EvalCase,
        recorder: TraceRecorder,
    ) -> EvalRunOutput:
        recorder.record_model_call(name="fake-provider")

        if "error" in case.id:
            raise RuntimeError("simulated runtime failure")

        if "tool" in case.id:
            recorder.record_tool_call(name="read_file")
            recorder.record_tool_result(
                name="read_file",
                payload={"source": "README.md"},
            )

        return EvalRunOutput(
            final_answer="Runtime uses Tool and README.md sources.",
            stopped_reason="completed",
            sources=["README.md"],
        )


async def test_eval_runner_checks_expected_tool() -> None:
    case = EvalCase(
        id="tool_001",
        input="Read README.",
        expected_tools=["read_file"],
    )

    result = await EvalRunner(FakeEvalExecutor()).run_case(case)

    assert result.status == "passed"
    assert result.observed_tools == ["read_file"]
    assert result.missing_expected_tools == []


async def test_eval_runner_marks_missing_tool_failed() -> None:
    case = EvalCase(
        id="no_tool_001",
        input="Should call search.",
        expected_tools=["search_knowledge_base"],
    )

    result = await EvalRunner(FakeEvalExecutor()).run_case(case)

    assert result.status == "failed"
    assert result.failure_reasons == ["missing_expected_tool"]
    assert result.missing_expected_tools == ["search_knowledge_base"]


async def test_eval_runner_checks_expected_contains() -> None:
    case = EvalCase(
        id="tool_002",
        input="Explain runtime.",
        expected_tools=["read_file"],
        expected_contains=["Runtime", "Tool"],
    )

    result = await EvalRunner(FakeEvalExecutor()).run_case(case)

    assert result.status == "passed"
    assert result.missing_expected_texts == []


async def test_eval_runner_marks_case_failed_with_reason() -> None:
    case = EvalCase(
        id="tool_003",
        input="Explain runtime.",
        expected_tools=["read_file"],
        expected_contains=["不存在的模块X"],
    )

    result = await EvalRunner(FakeEvalExecutor()).run_case(case)

    assert result.status == "failed"
    assert "missing_expected_text" in result.failure_reasons
    assert result.missing_expected_texts == ["不存在的模块X"]


async def test_eval_runner_continues_after_case_error() -> None:
    dataset = EvalDataset(
        cases=[
            EvalCase(
                id="error_001",
                input="Trigger runtime error.",
                expected_contains=["anything"],
            ),
            EvalCase(
                id="tool_004",
                input="Read README.",
                expected_tools=["read_file"],
                expected_contains=["Runtime"],
            ),
        ]
    )

    suite = await EvalRunner(FakeEvalExecutor()).run_dataset(dataset)

    assert suite.case_count == 2
    assert suite.results[0].status == "error"
    assert suite.results[0].failure_reasons == ["runtime_error"]
    assert suite.results[0].error_message == "simulated runtime failure"
    assert suite.results[1].status == "passed"


async def test_eval_runner_checks_expected_sources() -> None:
    case = EvalCase(
        id="tool_005",
        input="Read README.",
        expected_tools=["read_file"],
        expected_sources=["README.md"],
    )

    result = await EvalRunner(FakeEvalExecutor()).run_case(case)

    assert result.status == "passed"
    assert result.observed_sources
    assert result.missing_expected_sources == []


async def test_eval_runner_marks_wrong_source_failed() -> None:
    case = EvalCase(
        id="tool_006",
        input="Read README.",
        expected_tools=["read_file"],
        expected_sources=["docs/missing.md"],
    )

    result = await EvalRunner(FakeEvalExecutor()).run_case(case)

    assert result.status == "failed"
    assert "wrong_source" in result.failure_reasons
    assert result.missing_expected_sources == ["docs/missing.md"]


async def test_eval_runner_checks_expected_stopped_reason() -> None:
    case = EvalCase(
        id="tool_007",
        input="Read README.",
        expected_tools=["read_file"],
        expected_stopped_reason="completed",
    )

    result = await EvalRunner(FakeEvalExecutor()).run_case(case)

    assert result.status == "passed"
    assert result.stopped_reason == "completed"


async def test_eval_runner_records_trace_for_each_case() -> None:
    case = EvalCase(
        id="tool_008",
        input="Read README.",
        expected_tools=["read_file"],
    )

    result = await EvalRunner(FakeEvalExecutor()).run_case(case)

    assert result.run_id
    assert {event.case_id for event in result.trace_events} == {"tool_008"}
    assert [event.event_type for event in result.trace_events] == [
        "model_call",
        "tool_call",
        "tool_result",
        "final_answer",
    ]
