# Long-term Memory System

`forge-agent` supports a scoped long-term memory system that extends static RAG with dynamic, cross-run experience reuse.

The runtime flow is:

```text
memory recall -> context compose -> agent run -> memory write
```

## Why memory is separate from RAG

| Capability | RAG Knowledge Base | Long-term Memory |
|---|---|---|
| Primary purpose | Static documentation and project knowledge | Dynamic user/project/session experience |
| Data source | Markdown files and committed docs | Agent runs, user memory requests, reusable summaries, failure experience |
| Update pattern | Relatively stable | Continuously updated across runs |
| Runtime role | External knowledge grounding | Historical preference and experience reuse |

RAG answers questions from stable project documents. Memory recalls facts, preferences, task summaries, and reusable failure experience learned during previous runs.

## Memory types

`forge-agent` models three memory types:

| Type | Purpose | Example |
|---|---|---|
| `semantic` | Stable facts, preferences, rules | `项目默认使用 Python 3.13 和 uv` |
| `episodic` | Historical task or failure experience | `verification_failed: JsonlMemoryStore list shadowed built-in list in mypy` |
| `summary` | Reusable task or conversation summaries | `Completed the JSONL memory store and validated reload behavior` |

## Memory scopes

Memory records are isolated by scope and optional `scope_id`.

| Scope | Purpose |
|---|---|
| `global` | Shared global memory |
| `user` | User-level memory |
| `session` | Session-level memory, usually selected by `--session-id` |
| `project` | Project-level memory |

The CLI uses `session` scope when `--memory-path` is enabled for `forge run`.

## Core modules

```text
src/forge_agent/memory/
├── models.py          # MemoryRecord, MemoryType, MemoryScope, search/write models
├── store.py           # MemoryStore protocol
├── in_memory_store.py # Deterministic test-friendly store
├── jsonl_store.py     # Persistent local JSONL store
├── retriever.py       # MemoryRetriever
├── policy.py          # MemoryWritePolicy
└── writer.py          # MemoryWriter

src/forge_agent/context/
├── budget.py          # context budget enforcement
└── composer.py        # compose memory, RAG context, and user input

src/forge_agent/runtime/
├── native_runtime.py  # memory recall/write integration
└── events.py          # memory trace event types
```

## Write policy

Memory is not written blindly. `MemoryWritePolicy` accepts:

- explicit user memory requests such as `记住：...` or `remember ...`
- successful reusable task summaries
- `verification_failed` failure reasons
- reusable tool failure root causes
- stable project facts

It skips:

- temporary user input
- blank or overly short content
- sensitive-looking content such as passwords, API keys, secrets, and tokens
- duplicate memory already present in the same type/scope/scope_id boundary

## Store implementations

- `InMemoryStore`: deterministic test-friendly memory store for tests and runtime injection.
- `JsonlMemoryStore`: persistent local JSONL memory store for cross-run memory.

## Retrieval

The first implementation uses deterministic keyword overlap rather than embeddings. This keeps tests stable and makes local behavior easy to inspect.

`MemoryRetriever` supports:

- query
- `top_k`
- scope filter
- scope id filter
- memory type filter
- minimum score filter

## Context composition

`ContextComposer` merges:

1. recalled long-term memory
2. optional RAG context
3. current user input

It deduplicates memory records and applies a character budget before messages are passed to the model provider.

## Trace events

Memory behavior is observable through runtime trace events:

```text
memory_recall_started
memory_recall_result
memory_write_attempted
memory_write_skipped
memory_write_completed
```

These events make it possible to debug why a memory was recalled, written, or skipped.

## CLI usage

Write a memory:

```bash
uv run forge run "记住：我的项目默认使用 Python 3.13 和 uv" \
  --memory-path .memory \
  --session-id demo \
  --max-steps 1
```

Recall the memory in a later run:

```bash
uv run forge run "我的项目默认使用什么 Python 版本？" \
  --memory-path .memory \
  --session-id demo \
  --max-steps 1
```

List persisted memories:

```bash
uv run forge memory list \
  --memory-path .memory \
  --session-id demo
```

Search persisted memories:

```bash
uv run forge memory search "Python 3.13" \
  --memory-path .memory \
  --session-id demo
```

## Local storage layout

`JsonlMemoryStore` stores records by memory type:

```text
.memory/
├── semantic.jsonl
├── episodic.jsonl
└── summary.jsonl
```

The JSONL format is intentionally simple, readable, and suitable for local demos and tests.

## Verification

Run memory-specific tests:

```bash
uv run pytest -v \
  tests/test_memory_models.py \
  tests/test_memory_store_protocol.py \
  tests/test_in_memory_store.py \
  tests/test_jsonl_memory_store.py \
  tests/test_memory_retriever.py \
  tests/test_memory_writer.py \
  tests/test_memory_write_policy.py \
  tests/test_context_composer.py \
  tests/test_memory_runtime_integration.py \
  tests/test_memory_trace.py \
  tests/test_cli_memory.py
```

Run full project checks:

```bash
uv run pytest -v
uv run mypy src tests
uv run ruff check .
```
