# forge-agent

A minimal, testable Agent Platform runtime built from scratch.

`forge-agent` demonstrates the core architecture behind a tool-calling AI agent: model provider abstraction, tool registry, native agent runtime loop, structured tool results, trace events, and a CLI entrypoint.

This project is intentionally small and implementation-focused. It is designed as a learning and portfolio project for understanding how an Agent Platform works internally, rather than a production-ready SaaS platform or a full-featured agent framework.

## Features

* Minimal native Agent Runtime
* Deterministic `FakeProvider` for testing and demos
* Provider abstraction independent of any specific LLM vendor
* Tool Calling flow with normalized `ToolCall` and `ToolResult`
* Tool Registry for registering and discovering tools by name
* Built-in demo tools:

  * `list_files`
  * `read_file`
  * `calculator`
  * `echo_text`
* Structured runtime result with:

  * final answer
  * stopped reason
  * step count
  * tool results
  * trace events
* CLI interface powered by Typer
* Strict quality checks with pytest, mypy, and Ruff
* Apache License 2.0

## Project Status

Current status: **Day 1 Agent Runtime + Tool Calling skeleton**

Implemented:

* `ModelProvider` abstraction
* `FakeProvider`
* `Tool` abstraction
* `ToolRegistry`
* Basic file tools
* Safe arithmetic calculator tool
* Echo text assessment tool
* `NativeAgentRuntime`
* In-memory `TraceEvent`
* CLI command: `forge run`

Not implemented yet:

* Real LLM provider integration
* RAG
* Long-term memory
* Permission sandbox
* LangGraph workflow
* Evaluation framework
* Observability export
* Deployment

## Architecture

The core runtime flow is:

```text
User Input
  -> AgentRuntime
  -> ModelProvider
  -> ToolCall
  -> ToolRegistry
  -> Tool.execute()
  -> ToolResult
  -> AgentRuntime
  -> ModelProvider
  -> Final Answer
```

The project separates the major platform concerns:

```text
src/forge_agent/
  cli/
    app.py

  providers/
    base.py
    fake.py

  runtime/
    events.py
    state.py
    native_runtime.py

  tools/
    base.py
    registry.py
    defaults.py
    file_tools.py
    calculator.py
    echo.py
```

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

Clone the repository:

```bash
git clone <your-repo-url>
cd forge-agent
```

Install dependencies:

```bash
uv sync
```

## Usage

Show CLI help:

```bash
uv run forge --help
```

Run the minimal tool-calling demo:

```bash
uv run forge run "list files and read README"
```

Example output:

```text
Final answer: I inspected the workspace using these tools: list_files, read_file. The README was requested successfully.
Stopped reason: completed
Steps: 2

Tool calls:
- list_files: SUCCESS
- read_file: SUCCESS

Trace events:
- model_call
- model_response
- tool_call
- tool_result
- tool_call
- tool_result
- model_call
- model_response
- final_answer
```

## Core Concepts

### ModelProvider

`ModelProvider` defines the interface between the runtime and the model layer.

The runtime does not depend on OpenAI, Anthropic, DeepSeek, or any other specific model vendor. Instead, providers return normalized internal objects such as `ProviderResponse` and `ToolCall`.

Current provider:

* `FakeProvider`: deterministic provider for tests and demos

### Tool

A tool is a platform capability that can be called by the Agent Runtime.

Each tool provides:

* name
* description
* JSON schema
* argument validation
* structured execution result

Example tools:

* `list_files`: list files under a directory
* `read_file`: read a UTF-8 text file
* `calculator`: evaluate a safe arithmetic expression
* `echo_text`: echo the provided text

### ToolRegistry

`ToolRegistry` manages tool registration and lookup.

It is responsible for:

* registering tools
* preventing duplicate tool names
* returning provider-facing tool schemas
* executing tools by name
* converting unknown tool calls into structured `ToolResult` errors

### NativeAgentRuntime

`NativeAgentRuntime` drives the minimal agent loop:

```text
model_call -> tool_call -> tool_result -> next model_call -> final_answer
```

It supports:

* max step control
* structured stop reasons
* tool error feedback
* runtime trace events

Stop reasons:

* `completed`
* `max_steps`
* `error`

### TraceEvent

`TraceEvent` records important runtime events in memory.

Current event types:

* `model_call`
* `model_response`
* `tool_call`
* `tool_result`
* `final_answer`
* `runtime_stop`

Trace export is intentionally not implemented yet. Future versions may export traces to JSONL, OpenTelemetry, or external observability systems.

## Development

Run tests:

```bash
uv run pytest -q
```

Run type checks:

```bash
uv run mypy src tests
```

Run lint checks:

```bash
uv run ruff check .
```

Run all quality checks:

```bash
uv run pytest -q
uv run mypy src tests
uv run ruff check .
```

## Adding a New Tool

To add a new tool:

1. Create a new file under `src/forge_agent/tools/`
2. Define a Pydantic argument model
3. Implement `schema()`
4. Implement `execute()`
5. Return a structured `ToolResult`
6. Register the tool in `create_default_tool_registry()`
7. Add unit tests for success and validation failure
8. Add runtime-level tests if the tool is part of the default runtime demo

Example tool structure:

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from forge_agent.tools.base import ToolResult, ToolSchema


class EchoTextArgs(BaseModel):
    text: str


class EchoTextTool:
    name = "echo_text"
    description = "Echo the provided text."

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=EchoTextArgs.model_json_schema(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        try:
            args = EchoTextArgs.model_validate(arguments)
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=True,
                payload={"text": args.text},
            )
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )
```

## Design Principles

This project follows several design principles:

* Keep model providers independent from the runtime
* Keep tools independent from provider-specific formats
* Convert tool failures into structured results instead of leaking exceptions to the CLI
* Make the runtime deterministic and testable without requiring a real LLM
* Prefer simple native abstractions before introducing larger frameworks
* Keep Day 1 scope focused on Agent Runtime and Tool Calling only

## Roadmap

Planned learning and implementation stages:

* Day 1: Agent Runtime + Tool Calling skeleton
* Day 2: Real model provider integration
* Day 3: Context engineering and message management
* Day 4: LangGraph workflow integration
* Day 5: Evaluation and trace export
* Day 6: Permission and safety controls
* Day 7: Final demo polish and portfolio packaging

## Requirements

* Python 3.13+
* uv
* pytest
* mypy
* Ruff
* Typer
* Pydantic

## License

This project is licensed under the Apache License 2.0. See [LICENSE](./LICENSE) for details.
