from forge_agent.reflection import ReflectionConfig, VerificationResult
from forge_agent.runtime import ReflectionRuntime, RunConfig, RunResult
from forge_agent.runtime.state import AgentState


class StaticRuntime:
    def __init__(self, result: RunResult) -> None:
        self.result = result

    def run(
        self,
        user_input: str,
        config: RunConfig | None = None,
    ) -> RunResult:
        del user_input, config
        return self.result.model_copy(deep=True)


class SequenceVerifier:
    def __init__(self, results: list[VerificationResult]) -> None:
        self.results = results
        self.calls = 0

    def verify(
        self,
        result: RunResult,
        state: AgentState | None = None,
    ) -> VerificationResult:
        del result, state
        index = min(self.calls, len(self.results) - 1)
        self.calls += 1
        return self.results[index]


def event_types(result: RunResult) -> list[str]:
    return [event.event_type for event in result.trace_events]


def test_reflection_trace_records_successful_verification() -> None:
    base_result = RunResult(
        final_answer="Done.",
        stopped_reason="completed",
        steps=1,
    )
    verifier = SequenceVerifier(
        [
            VerificationResult(
                passed=True,
                decision="accept",
                reasons=["answer passed"],
                confidence=0.9,
            )
        ]
    )

    result = ReflectionRuntime(StaticRuntime(base_result), verifier).run("task")

    assert "reflection_started" in event_types(result)
    assert "verification_result" in event_types(result)
    assert "reflection_completed" in event_types(result)
    assert "reflection_failed" not in event_types(result)

    verification_events = [
        event for event in result.trace_events if event.event_type == "verification_result"
    ]
    assert verification_events[0].data["passed"] is True
    assert verification_events[0].data["decision"] == "accept"
    assert verification_events[0].data["confidence"] == 0.9


def test_reflection_trace_records_revision_request() -> None:
    base_result = RunResult(
        final_answer="Initial answer.",
        stopped_reason="completed",
        steps=1,
    )
    verifier = SequenceVerifier(
        [
            VerificationResult(
                passed=False,
                decision="revise",
                reasons=["missing citation"],
                missing_evidence=["citation"],
            ),
            VerificationResult(
                passed=True,
                decision="accept",
                reasons=["revision passed"],
            ),
        ]
    )

    result = ReflectionRuntime(
        StaticRuntime(base_result),
        verifier,
        ReflectionConfig(max_attempts=2, max_revisions=1),
    ).run("task")

    types = event_types(result)

    assert "reflection_started" in types
    assert "verification_result" in types
    assert "critique_generated" in types
    assert "revision_requested" in types
    assert "reflection_completed" in types

    revision_events = [
        event for event in result.trace_events if event.event_type == "revision_requested"
    ]
    assert revision_events[0].data["reasons"] == ["missing citation"]


def test_reflection_trace_records_failed_verification() -> None:
    base_result = RunResult(
        final_answer="Unsafe answer.",
        stopped_reason="completed",
        steps=1,
    )
    verifier = SequenceVerifier(
        [
            VerificationResult(
                passed=False,
                decision="abort",
                reasons=["permission denied"],
            )
        ]
    )

    result = ReflectionRuntime(StaticRuntime(base_result), verifier).run("task")

    types = event_types(result)

    assert "reflection_started" in types
    assert "verification_result" in types
    assert "critique_generated" in types
    assert "reflection_failed" in types
    assert result.stopped_reason == "verification_failed"

    failed_events = [
        event for event in result.trace_events if event.event_type == "reflection_failed"
    ]
    assert failed_events[0].data["reason"] == "verification_failed"
    assert failed_events[0].data["error_message"] == "permission denied"
