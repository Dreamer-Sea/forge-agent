# Reflection / Verification Runtime

`forge-agent` supports a reflection runtime wrapper that verifies an agent result
before returning it to the caller.

## Goal

The reflection layer upgrades the runtime flow from offline-only evaluation:

    agent run -> eval report

to runtime verification:

    final_answer -> verify -> accept / revise / retry / abort

## Core Components

### VerificationResult

`VerificationResult` is the machine-readable output of a verifier. It records:

- whether verification passed
- the next decision: `accept`, `revise`, `retry`, or `abort`
- failure reasons
- missing evidence
- unsupported claims
- confidence

### RuleVerifier

`RuleVerifier` checks generic runtime correctness:

- empty final answer
- non-completed stopped reason
- max step termination
- failed tool results
- permission-denied tool failures
- abnormal output such as tracebacks
- answers that claim success despite failed tools

### RagVerifier

`RagVerifier` checks citation groundedness for RAG answers:

- whether retrieved citations exist
- whether the final answer contains citations
- whether answer citations come from retrieved context
- whether empty retrieval avoids unsupported direct claims

### ReflectionRuntime

`ReflectionRuntime` wraps an existing runtime.

    base_runtime.run()
            ↓
    verifier.verify(result)
            ↓
    accept / revise / retry / abort

The wrapper does not replace `NativeAgentRuntime`, `PlanningRuntime`, or
`LangGraphAgentRuntime`. It composes with them and adds a verification gate.

## Trace Events

Reflection emits the following runtime trace events:

- `reflection_started`
- `verification_result`
- `critique_generated`
- `revision_requested`
- `reflection_completed`
- `reflection_failed`

These events make verification decisions visible in CLI output, eval traces, and
debug reports.

## CLI Usage

    uv run forge run "echo hello" --runtime reflection

## Eval Usage

    uv run forge eval examples/evals/reflection_runtime_eval.jsonl \
      --runtime reflection \
      --output examples/reports/reflection-report.md \
      --trace-out examples/reports/reflection-traces.jsonl

The eval report includes:

- `reflection_attempts`
- `verification_passed`
- `verification_reasons`
- `unsupported_claims`
- `missing_evidence`
