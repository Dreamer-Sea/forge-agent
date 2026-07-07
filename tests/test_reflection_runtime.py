from forge_agent.reflection import ReflectionConfig, VerificationResult
from forge_agent.runtime import ReflectionRunResult, ReflectionRuntime, RunConfig, RunResult
from forge_agent.runtime.events import TraceEvent
from forge_agent.runtime.state import AgentState


class StaticRuntime:
    def __init__(self, result: RunResult) -> None:
        self.result = result
        self.calls = 0

    def run(
        self,
        user_input: str,
        config: RunConfig | None = None,
    ) -> RunResult:
        del user_input, config
        self.calls += 1
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


def test_reflection_runtime_accepts_verified_result() -> None:
    base_result = RunResult(
        final_answer="Done.",
        stopped_reason="completed",
        steps=1,
    )
    runtime = StaticRuntime(base_result)
    verifier = SequenceVerifier(
        [
            VerificationResult(
                passed=True,
                decision="accept",
                reasons=["answer passed"],
            )
        ]
    )

    result = ReflectionRuntime(runtime, verifier).run("finish task")

    assert isinstance(result, ReflectionRunResult)
    assert result.final_answer == "Done."
    assert result.stopped_reason == "completed"
    assert len(result.reflection_attempts) == 1
    assert result.reflection_attempts[0].verification_result.passed is True
    assert runtime.calls == 1
    assert verifier.calls == 1


def test_reflection_runtime_returns_verification_failed_on_abort() -> None:
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

    result = ReflectionRuntime(StaticRuntime(base_result), verifier).run("write file")

    assert result.stopped_reason == "verification_failed"
    assert result.error_message == "permission denied"
    assert len(result.reflection_attempts) == 1


def test_reflection_runtime_can_revise_once_before_accepting() -> None:
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
                reasons=["missing evidence"],
                missing_evidence=["citation"],
            ),
            VerificationResult(
                passed=True,
                decision="accept",
                reasons=["revision passed"],
            ),
        ]
    )
    reflection_config = ReflectionConfig(max_attempts=2, max_revisions=1)

    result = ReflectionRuntime(
        StaticRuntime(base_result),
        verifier,
        reflection_config,
    ).run("answer with evidence")

    assert result.stopped_reason == "completed"
    assert result.final_answer is not None
    assert "Verification requested a revision." in result.final_answer
    assert len(result.reflection_attempts) == 2
    assert verifier.calls == 2


def test_reflection_runtime_stops_when_revision_limit_is_reached() -> None:
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
                reasons=["still unsupported"],
            )
        ]
    )
    reflection_config = ReflectionConfig(max_attempts=1, max_revisions=0)

    result = ReflectionRuntime(
        StaticRuntime(base_result),
        verifier,
        reflection_config,
    ).run("answer with evidence")

    assert result.stopped_reason == "reflection_limit_reached"
    assert result.error_message == "still unsupported"
    assert len(result.reflection_attempts) == 1


def test_reflection_runtime_can_retry_base_runtime() -> None:
    retry_result = RunResult(
        final_answer=None,
        stopped_reason="max_steps",
        steps=1,
    )
    accepted_result = RunResult(
        final_answer="Recovered answer.",
        stopped_reason="completed",
        steps=1,
        trace_events=[
            TraceEvent(
                event_type="runtime_stop",
                step=1,
                data={"reason": "completed"},
            )
        ],
    )

    class RetryingRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def run(
            self,
            user_input: str,
            config: RunConfig | None = None,
        ) -> RunResult:
            del user_input, config
            self.calls += 1
            if self.calls == 1:
                return retry_result.model_copy(deep=True)
            return accepted_result.model_copy(deep=True)

    runtime = RetryingRuntime()
    verifier = SequenceVerifier(
        [
            VerificationResult(
                passed=False,
                decision="retry",
                reasons=["runtime reached max_steps"],
            ),
            VerificationResult(
                passed=True,
                decision="accept",
                reasons=["retry passed"],
            ),
        ]
    )

    result = ReflectionRuntime(
        runtime,
        verifier,
        ReflectionConfig(max_attempts=2, max_revisions=0),
    ).run("retry task")

    assert runtime.calls == 2
    assert verifier.calls == 2
    assert result.stopped_reason == "completed"
    assert result.final_answer == "Recovered answer."
    assert len(result.reflection_attempts) == 2
