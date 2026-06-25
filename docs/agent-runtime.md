# Agent Runtime

Forge Agent supports pluggable runtime backends through a stable runtime abstraction.

The goal is to keep the platform-owned interfaces stable while allowing orchestration frameworks such as LangGraph to be used as replaceable adapters.

## Runtime Interface

All runtimes return the same `RunResult` shape:

- `final_answer`
- `stopped_reason`
- `steps`
- `tool_results`
- `tool_calls`
- `trace_events`
- `error_message`

All runtimes accept a `RunConfig` object that carries runtime-level configuration such as:

- `runtime_name`
- `max_steps`
- `workspace`
- `trace_enabled`

This keeps CLI, tests, tools, trace, and future evaluation code independent from a specific orchestration framework.

## NativeAgentRuntime

`NativeAgentRuntime` is the project-owned runtime.

Its responsibilities are:

- Execute the model/tool loop directly.
- Call the configured model provider.
- Expose tool schemas through the existing `ToolRegistry`.
- Execute tool calls through the existing tool system.
- Record `TraceEvent` entries for model calls, tool calls, permission checks, tool results, and runtime stops.
- Demonstrate understanding of the lower-level agent loop.

The native runtime is intentionally kept in the project because it makes the underlying agent mechanics explicit.

## LangGraphAgentRuntime

`LangGraphAgentRuntime` is a runtime adapter backed by LangGraph.

Its responsibilities are:

- Execute a LangGraph `StateGraph` workflow.
- Classify whether a task should use RAG.
- Route RAG tasks through a retrieval node.
- Route non-RAG tasks through a direct answer path.
- Reuse the existing `ToolRegistry`.
- Reuse the existing `SearchKnowledgeBaseTool`.
- Preserve the existing permission model.
- Emit the same project-level `TraceEvent` shape.
- Return the same `RunResult` shape as the native runtime.

The LangGraph runtime demonstrates workflow orchestration, state passing, node execution, and conditional routing without replacing the platform abstractions.

## Why LangGraph Is an Adapter

LangGraph is not the architectural center of this project.

The core platform abstractions remain:

- `AgentRuntime`
- `RunConfig`
- `RunResult`
- `ToolRegistry`
- `PermissionPolicy`
- `TraceEvent`
- RAG tools

LangGraph is used behind `LangGraphAgentRuntime` as one possible runtime backend.

This design avoids turning Forge Agent into a framework demo. The project can run with the native runtime, LangGraph runtime, or future runtime backends without changing the CLI contract or tool system.

## Runtime Selection

The CLI supports runtime selection:

```bash
uv run forge run "根据知识库回答 Permission system 如何工作" --runtime native
uv run forge run "根据知识库回答 Permission system 如何工作" --runtime langgraph
```

The default runtime is `native`.

Use `native` when:

- You want the simplest model/tool loop.
- You want to debug low-level agent behavior.
- The task does not need explicit workflow orchestration.

Use `langgraph` when:

- The task benefits from explicit state transitions.
- The workflow needs conditional routing.
- The runtime should model a graph of nodes and edges.
- Future features may require checkpointing, human-in-the-loop, or more complex orchestration.

## Current Day 4 Workflow

The LangGraph runtime currently implements this routing workflow:

```text
START
  |
  v
classify_task
  |
  v
should_use_rag?
  |-- rag    --> retrieve --> answer_with_context --> END
  |-- direct --> direct_tool_agent              --> END
```

The route is based on user input.

Inputs containing one of these trigger phrases use RAG:

- `knowledge base`
- `according to docs`
- `根据知识库`

All other inputs use the direct path.

## Design Boundary

LangGraph is a runtime backend, not a replacement for the project runtime abstraction.

The intended design boundary is:

```text
AgentRuntime
  |-- NativeAgentRuntime
  |-- LangGraphAgentRuntime
```

Both runtime implementations should keep sharing:

- Tool registry
- Permission policy
- Trace events
- RAG tool
- Stable run result shape

This is the main Day 4 design requirement: keep the self-owned runtime abstraction stable while using LangGraph as a replaceable adapter.
