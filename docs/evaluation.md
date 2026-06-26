# Evaluation and Trace

`forge-agent` includes a lightweight evaluation loop for validating core agent platform behavior.

The goal is not to benchmark model intelligence. The goal is to make the demo paths repeatable, debuggable, and safe to refactor.

## What Eval Checks

Each JSONL eval case can define:

- `id`: stable case identifier.
- `input`: user task sent to the agent runtime.
- `expected_tools`: tools that should be called.
- `expected_contains`: substrings expected in the final answer.

This keeps the evaluation format simple enough for a demo while still catching regressions in runtime behavior, tool routing, RAG grounding, and output formatting.

## Demo Command

Run the Day 6 eval dataset:

    uv run forge eval examples/evals/agent_platform.jsonl

Expected output summary:

    case_count: 3
    success_rate: 100.00%
    tool_call_success_rate: 100.00%
    expected_contains_pass_rate: 100.00%
    failed_cases: 0
    trace_file: reports/traces.jsonl
    report_file: reports/eval-report.md

## Trace

Trace events make each agent run inspectable.

A typical trace includes:

- `model_call`
- `model_response`
- `tool_call`
- `permission_check`
- `tool_result`
- `final_answer`

For RAG cases, the trace should show the `search_knowledge_base` tool call. For file-tool cases, the trace should show file-related tools such as `list_files` and `read_file`.

## Report

The eval report summarizes:

- Total case count.
- Success rate.
- Tool-call success rate.
- Expected-content pass rate.
- Failed cases.
- Trace file location.

The report is intended for quick review during development and for interview/demo presentation.

## Design Trade-offs

The Day 6 evaluator intentionally uses deterministic checks:

- It checks exact tool names instead of judging open-ended reasoning quality.
- It checks expected substrings instead of using an LLM judge.
- It writes local JSON/Markdown reports instead of requiring an external observability backend.

This makes the demo easy to run from a clean checkout and reliable in CI.
