"""Deterministic eval runner."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from forge_agent.evals.dataset import EvalCase, EvalDataset
from forge_agent.observability import TraceEvent, TraceRecorder

EvalCaseStatus = Literal["passed", "failed", "error"]
EvalFailureReason = Literal[
    "missing_expected_tool",
    "missing_expected_text",
    "wrong_source",
    "wrong_stopped_reason",
    "runtime_error",
]


class EvalRunOutput(BaseModel):
    """Normalized output returned by one agent eval execution."""

    model_config = ConfigDict(extra="forbid")

    final_answer: str
    stopped_reason: str = "completed"
    sources: list[str] = Field(default_factory=list)


class EvalCaseExecutor(Protocol):
    """Protocol implemented by concrete agent runtime adapters."""

    async def run_case(
        self,
        case: EvalCase,
        recorder: TraceRecorder,
    ) -> EvalRunOutput:
        """Execute one eval case and record trace events."""


class EvalResult(BaseModel):
    """Result for one eval case."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    case_id: str
    status: EvalCaseStatus
    input: str
    final_answer: str = ""
    stopped_reason: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    observed_tools: list[str] = Field(default_factory=list)
    expected_contains: list[str] = Field(default_factory=list)
    missing_expected_tools: list[str] = Field(default_factory=list)
    missing_expected_texts: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    observed_sources: list[str] = Field(default_factory=list)
    missing_expected_sources: list[str] = Field(default_factory=list)
    failure_reasons: list[EvalFailureReason] = Field(default_factory=list)
    error_message: str | None = None
    run_id: str
    trace_events: list[TraceEvent] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return whether the case passed."""
        return self.status == "passed"


class EvalSuiteResult(BaseModel):
    """Result for a full eval dataset run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    results: list[EvalResult]

    @property
    def case_count(self) -> int:
        """Return number of executed cases."""
        return len(self.results)

    @property
    def failed_cases(self) -> list[EvalResult]:
        """Return failed or errored cases."""
        return [result for result in self.results if not result.passed]

    @property
    def trace_events(self) -> list[TraceEvent]:
        """Return all trace events emitted by the suite."""
        events: list[TraceEvent] = []
        for result in self.results:
            events.extend(result.trace_events)
        return events


class EvalRunner:
    """Run deterministic eval cases against an agent executor."""

    def __init__(self, executor: EvalCaseExecutor) -> None:
        self.executor = executor

    async def run_dataset(self, dataset: EvalDataset) -> EvalSuiteResult:
        """Run every case in a dataset and continue after failures."""
        results: list[EvalResult] = []

        for case in dataset.cases:
            results.append(await self.run_case(case))

        return EvalSuiteResult(results=results)

    async def run_case(self, case: EvalCase) -> EvalResult:
        """Run one eval case."""
        recorder = TraceRecorder(case_id=case.id)

        try:
            output = await self.executor.run_case(case, recorder)
        except Exception as exc:  # noqa: BLE001 - eval must isolate case failures
            recorder.record(
                "final_answer",
                payload={
                    "answer": "",
                    "stopped_reason": "error",
                    "error": str(exc),
                },
            )
            return EvalResult(
                case_id=case.id,
                status="error",
                input=case.input,
                stopped_reason="error",
                expected_tools=case.expected_tools,
                expected_contains=case.expected_contains,
                expected_sources=case.expected_sources,
                failure_reasons=["runtime_error"],
                error_message=str(exc),
                run_id=recorder.run_id,
                trace_events=list(recorder.events),
            )

        recorder.record_final_answer(
            answer=output.final_answer,
            stopped_reason=output.stopped_reason,
        )

        return self._evaluate_case(
            case=case,
            output=output,
            recorder=recorder,
        )

    def _evaluate_case(
        self,
        *,
        case: EvalCase,
        output: EvalRunOutput,
        recorder: TraceRecorder,
    ) -> EvalResult:
        observed_tools = _observed_tools(recorder.events)
        missing_tools = [
            tool for tool in case.expected_tools if tool not in observed_tools
        ]

        missing_texts = [
            expected_text
            for expected_text in case.expected_contains
            if not _contains_text(output.final_answer, expected_text)
        ]

        observed_sources = sorted(
            {
                *output.sources,
                *_observed_sources(recorder.events),
            }
        )
        missing_sources = [
            source
            for source in case.expected_sources
            if source not in observed_sources
        ]

        failure_reasons: list[EvalFailureReason] = []

        if missing_tools:
            failure_reasons.append("missing_expected_tool")
        if missing_texts:
            failure_reasons.append("missing_expected_text")
        if missing_sources:
            failure_reasons.append("wrong_source")
        if (
            case.expected_stopped_reason is not None
            and output.stopped_reason != case.expected_stopped_reason
        ):
            failure_reasons.append("wrong_stopped_reason")

        status: EvalCaseStatus = "failed" if failure_reasons else "passed"

        return EvalResult(
            case_id=case.id,
            status=status,
            input=case.input,
            final_answer=output.final_answer,
            stopped_reason=output.stopped_reason,
            expected_tools=case.expected_tools,
            observed_tools=observed_tools,
            expected_contains=case.expected_contains,
            missing_expected_tools=missing_tools,
            missing_expected_texts=missing_texts,
            expected_sources=case.expected_sources,
            observed_sources=observed_sources,
            missing_expected_sources=missing_sources,
            failure_reasons=failure_reasons,
            run_id=recorder.run_id,
            trace_events=list(recorder.events),
        )


def _observed_tools(events: Iterable[TraceEvent]) -> list[str]:
    tools: list[str] = []

    for event in events:
        if event.event_type == "tool_call" and event.name is not None:
            tools.append(event.name)

    return tools


def _contains_text(answer: str, expected_text: str) -> bool:
    return expected_text.casefold() in answer.casefold()


def _observed_sources(events: Iterable[TraceEvent]) -> list[str]:
    sources: set[str] = set()

    for event in events:
        for value in _iter_payload_strings(event.payload):
            sources.add(value)

    return sorted(sources)


def _iter_payload_strings(payload: Any) -> Iterable[str]:
    if isinstance(payload, str):
        yield payload
        return

    if isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_payload_strings(value)
        return

    if isinstance(payload, list):
        for value in payload:
            yield from _iter_payload_strings(value)
        return
