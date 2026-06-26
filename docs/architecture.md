# Architecture

`forge-agent` is a demo-level Agent Platform implemented as a local CLI application.

The project focuses on platform primitives behind agent systems rather than a single vertical application. Its core abstractions are runtime orchestration, model providers, tool execution, RAG, permission checks, traces, and evaluation.

## High-level Components

### CLI

The CLI is the user-facing entry point.

It exposes commands for:

- Running one agent task.
- Indexing a local knowledge base.
- Running JSONL eval cases.
- Inspecting stable demo output.

### Agent Runtime

The runtime controls the agent loop.

It is responsible for:

- Sending user tasks to a model provider.
- Receiving model responses and tool calls.
- Executing tools through the tool registry.
- Recording trace events.
- Producing a final answer.
- Stopping on completion, error, or max steps.

### Model Provider

The model provider abstracts model interaction.

The demo uses deterministic providers so tests and demo commands remain stable. Production systems could plug in OpenAI-compatible providers or other hosted/local models.

### Tool Registry

The tool registry exposes capabilities to the runtime.

Current demo tools include:

- File listing.
- File reading.
- Text echoing.
- Calculator-style deterministic tools.
- Local knowledge-base search.

### RAG Pipeline

The RAG pipeline indexes local Markdown files and exposes retrieval as a tool named `search_knowledge_base`.

The runtime does not hard-code RAG behavior. It calls the RAG tool when the model provider decides that retrieved context is needed.

### Permission and Workspace Guard

Tool execution is protected by permission checks.

File and knowledge-base operations must stay inside the configured workspace. This prevents path escape and keeps local demo execution auditable.

### Trace Recorder

Trace events make every run inspectable.

Typical events include:

- `model_call`
- `model_response`
- `tool_call`
- `permission_check`
- `tool_result`
- `final_answer`

### Eval Runner

The eval runner executes JSONL cases and checks:

- Expected tool calls.
- Expected substrings in final answers.
- Success rate.
- Trace/report generation.

## Runtime Flow

A normal agent run follows this flow:

1. User submits a task through the CLI.
2. CLI creates settings, tools, and runtime.
3. Runtime sends the task to the model provider.
4. Provider returns either a tool call or a final answer.
5. Runtime checks permissions before tool execution.
6. Tool result is recorded and sent back into the loop.
7. Runtime emits a final answer and trace events.

## Design Principles

The project follows four design principles:

- Keep the demo local and reproducible.
- Keep platform abstractions explicit.
- Make tool execution observable.
- Prefer deterministic tests over impressive but flaky demos.

## Non-goals

This project intentionally does not implement:

- Web UI.
- Multi-tenant SaaS.
- Production vector database.
- Docker sandbox.
- MCP marketplace.
- Distributed tracing backend.
