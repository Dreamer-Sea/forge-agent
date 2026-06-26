# forge-agent Final Report

## Project Goal

`forge-agent` is a minimal Agent Platform demo designed to demonstrate the core engineering capabilities behind production-grade AI agents.

The goal is not to build a full SaaS product or a feature-rich agent framework. The goal is to show, in a compact and testable codebase, how an Agent Platform can be decomposed into clear engineering boundaries:

* Agent Runtime
* Model Provider abstraction
* Tool Registry
* RAG Tool
* Workspace Permission Guard
* Trace Events
* Eval Runner
* LangGraph Runtime Adapter
* CLI demo surface

The project is intended as a portfolio-quality engineering artifact. It is small enough to explain in an interview, while still showing the architecture shape of a real agent platform.

---

## Implemented Capabilities

### Runtime

Implemented a native agent runtime that supports:

* user input handling
* model-provider calls
* tool-call execution
* tool observations
* final answer generation
* max-step protection
* structured stop reasons
* structured run results

The runtime owns orchestration and does not directly embed provider-specific or tool-specific logic.

### Model Provider Abstraction

Implemented a provider boundary so the runtime depends on a model provider interface instead of a specific vendor SDK.

Implemented providers include:

* deterministic `FakeProvider`
* OpenAI-compatible provider extension point

The deterministic provider makes runtime, tool-calling, RAG, and eval behavior testable without network access or API keys.

### Tool System

Implemented a tool system with:

* tool abstraction
* tool schema
* argument validation
* structured `ToolResult`
* duplicate tool-name rejection
* unknown-tool structured errors
* built-in tools such as file tools, calculator, echo, and RAG search

The runtime delegates tool execution to `ToolRegistry` instead of branching on tool names.

### RAG

Implemented a local Markdown RAG pipeline:

* Markdown document loading
* heading-aware chunking
* source metadata preservation
* deterministic keyword retrieval
* grounded context building
* citation output
* RAG exposed as an agent tool

This keeps the demo reproducible and easy to validate. The current retriever can later be replaced by embeddings, hybrid search, vector databases, or rerankers.

### Security and Permission

Implemented workspace-oriented safety controls:

* workspace guard
* path traversal blocking
* absolute path escape blocking
* symlink escape blocking
* read allowed by default
* write denied by default
* structured permission errors
* permission decisions recorded in trace events

The security model is intentionally demo-level, but it demonstrates the correct boundary: file access must go through workspace validation and permission checks.

### Observability

Implemented trace events for core runtime behavior:

* model calls
* tool calls
* tool results
* permission decisions
* eval case traces
* JSONL trace export

The trace system makes runtime behavior inspectable and supports debugging, eval analysis, and interview explanation.

### Evaluation

Implemented a deterministic eval runner that supports:

* JSONL eval dataset loading
* expected tool checks
* expected answer-content checks
* expected source checks
* expected stop-reason checks
* per-case result reporting
* Markdown report output
* JSON report output
* JSONL trace output

The eval system acts as a lightweight regression suite for agent behavior.

### LangGraph Adapter

Implemented a LangGraph runtime adapter while keeping the native runtime protocol stable.

The adapter demonstrates how graph-style orchestration can be added without making the whole project dependent on LangGraph-specific APIs.

### CLI

Implemented a CLI with the following main commands:

```bash
uv run forge --help
uv run forge run "Read README and summarize the architecture."
uv run forge rag index examples/knowledge_base
uv run forge eval examples/evals/agent_platform.jsonl
```

The CLI is the main demonstration and validation surface for the project.

---

## Demo Commands

### Demo 1: Tool Calling

```bash
uv run forge run "Read README and summarize the architecture."
```

Demonstrates:

* CLI task dispatch
* runtime orchestration
* provider response handling
* file-tool execution
* structured tool results
* final answer generation

### Demo 2: RAG with Citation

```bash
uv run forge rag index examples/knowledge_base
uv run forge run "According to the knowledge base, how does the permission system work?"
```

Demonstrates:

* local knowledge-base indexing
* Markdown loading
* chunking
* retrieval
* citation generation
* RAG-as-tool integration

### Demo 3: Eval and Trace

```bash
uv run forge eval examples/evals/agent_platform.jsonl
```

Demonstrates:

* JSONL eval execution
* expected-tool validation
* expected-content validation
* report generation
* trace export
* regression-style quality loop

---

## Test Summary

Final local quality baseline:

```text
pytest: 138 passed
mypy: success, no issues found
ruff check: all checks passed
ruff format: all files formatted
forge --help: exits successfully
```

The test suite covers the main platform matrix:

| Area        | Coverage                                                             |
| ----------- | -------------------------------------------------------------------- |
| Runtime     | tool execution, completion, max steps, tool errors, result shape     |
| Tool System | registration, duplicate rejection, unknown tools, validation errors  |
| RAG         | loader, chunker, retriever, context builder, RAG tool                |
| Permission  | workspace access, path traversal, write denial, trace recording      |
| Eval        | JSONL loading, expected tools, expected content, expected sources    |
| Trace       | model/tool events, JSONL export, run ID serialization                |
| CLI         | run command, RAG command, eval command, runtime selection            |
| LangGraph   | workflow routing, state preservation, trace events, runtime protocol |

---

## Eval Summary

The eval dataset validates deterministic platform behavior rather than subjective answer quality.

The current eval loop checks:

* whether the expected tool was called
* whether the final answer contains required content
* whether expected sources are present
* whether the runtime stopped for the expected reason
* whether traces are recorded for each case

This makes eval useful as a regression signal during refactoring and final project hardening.

---

## Key Design Decisions

### Native Runtime First

A native runtime makes the agent loop explicit and easy to inspect.

This avoids hiding core behavior behind a framework too early and makes the following easier to explain:

* model call lifecycle
* tool-call execution
* tool-result feedback
* stop conditions
* error handling
* trace emission

### Provider Independence

The runtime depends on a provider protocol instead of a specific model SDK.

This allows the same runtime to work with deterministic tests, fake providers, and real provider integrations.

### RAG as a Tool

RAG is exposed as a tool rather than being hardcoded into the runtime.

This keeps the runtime generic and makes RAG permission-checkable, traceable, and testable like any other platform capability.

### LangGraph as an Adapter

LangGraph is used as an optional runtime adapter.

This demonstrates framework integration without replacing the platform boundary.

### Permission Before File Access

File tools and RAG indexing paths go through workspace checks.

This prevents common path-escape risks and demonstrates the security boundary required by coding agents.

### Trace and Eval as First-Class Concerns

Trace and eval are part of the platform, not external afterthoughts.

This makes the demo more than a toy agent: behavior can be reproduced, inspected, and verified.

---

## Known Limitations

This project is a demo-level Agent Platform, not a production SaaS system.

Current limitations:

* no production-grade sandbox
* no OS-level isolation
* no multi-tenant authentication
* no persistent session store
* no distributed runtime
* no distributed tracing backend
* no OpenTelemetry integration
* no production secret management
* no audit log system
* no vector database
* no embedding-based retrieval
* no reranking
* no online eval service
* no human approval workflow
* no cost tracking
* no web UI
* no deployment manifests
* no full CI/CD pipeline included in the demo scope
* no multi-agent orchestration

These limitations are intentional. The project focuses on demonstrating core platform boundaries and validation strategy.

---

## Roadmap

### Runtime

* persistent multi-turn sessions
* streaming output
* interrupt and resume
* human-in-the-loop checkpoints
* richer runtime state inspection

### Security

* stronger sandbox isolation
* command allowlist policy
* write approval workflow
* secret redaction
* audit log persistence

### RAG

* embedding-based retrieval
* vector database backend
* hybrid search
* reranking
* citation quality scoring
* document freshness metadata

### Observability

* OpenTelemetry integration
* trace viewer
* structured logs
* latency metrics
* token and cost metrics

### Evaluation

* larger eval dataset
* regression eval suite
* failure clustering
* CI quality gates
* online eval dashboard

### Packaging

* GitHub Actions CI
* Docker image
* release artifacts
* example deployment profile

---

## Interview Talking Points

### 30-Second Version

I built `forge-agent`, a minimal Agent Platform demo. The focus is not a single agent application, but the platform capabilities behind agents: runtime orchestration, tool calling, RAG, permission control, trace events, eval runner, and a LangGraph adapter. It has CLI demos, tests, and eval cases to prove the key paths are reproducible and verifiable.

### 2-Minute Version

`forge-agent` is a minimal Agent Platform demo. At the bottom, I implemented a native Agent Runtime to demonstrate multi-step agent loops, tool calls, stop conditions, and error feedback. The model layer is abstracted through a `ModelProvider` protocol, so the runtime is not coupled to one vendor. The tool layer uses a `ToolRegistry` and structured schemas. RAG is implemented as a platform tool with local Markdown loading, heading-aware chunking, retrieval, and citations. For safety, it has a workspace guard and permission policy to block path traversal and unauthorized writes. For quality and observability, it includes trace events, JSONL trace export, and a deterministic eval runner. Finally, it integrates LangGraph as an optional runtime adapter, showing how workflow orchestration can be added without replacing the platform boundary.

### Deep-Dive Topics

#### Runtime

* The runtime loop calls the provider, executes requested tools, appends tool observations, and calls the provider again until completion or `max_steps`.
* `max_steps` prevents uncontrolled loops.
* Tool errors are returned as structured tool results so the model can observe and recover.
* Native runtime is the reference implementation; LangGraph runtime is an adapter for workflow orchestration.

#### RAG

* Loader reads Markdown documents.
* Chunker creates heading-aware chunks.
* Retriever finds relevant chunks.
* Context builder formats grounded context with citations.
* RAG is exposed through the tool system, so it can be traced and permission-checked.

#### Permission

* Workspace guard blocks path escape risks.
* Path traversal is stopped by resolving paths against the workspace root.
* Write is denied by default because coding agents can modify local state and require stricter controls.
* Permission denials are recorded in trace events.

#### Trace and Eval

* Trace records model calls, tool calls, tool results, permission decisions, and eval case behavior.
* Eval cases define user input and expected behavior.
* Tool-call checks validate routing.
* Expected-content checks validate final-answer behavior.

#### LangGraph

* LangGraph is an adapter rather than the core platform.
* This keeps the runtime protocol stable.
* Future human-in-the-loop steps can be added as explicit workflow nodes.

---

## Final Status

Final project state:

* Code: core platform path completed
* CLI: `run`, `rag index`, and `eval` available
* Runtime: native runtime and LangGraph adapter
* RAG: Markdown loading, chunking, retrieval, and citation
* Security: workspace guard and permission policy
* Observability: trace events and JSONL export
* Eval: JSONL dataset, runner, and reports
* Tests: pytest, mypy, Ruff, and formatting checks pass
* Docs: README and final report prepared
* Interview packaging: 30-second, 2-minute, and deep-dive versions prepared
