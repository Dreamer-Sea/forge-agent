# Native Runtime vs LangChain

This document explains the boundary between the `forge-agent` native runtime and the LangChain integration.

The project intentionally keeps both:

- `forge-agent` native runtime for platform-level control;
- LangChain integration for ecosystem compatibility.

The goal is not to choose one forever. The goal is to understand where each layer is strongest.

## Summary

| Dimension | forge-agent Native Runtime | LangChain |
|---|---|---|
| Primary goal | Show the agent loop and platform boundary | Integrate models, tools, retrievers, and agent harnesses quickly |
| Runtime control | High: loop, tool execution, permission, trace, and eval are explicit | High-level abstraction, but lower-level behavior depends on framework internals |
| Testability | Deterministic with `FakeProvider` and native evals | Depends on agent setup, model behavior, and callback configuration |
| Tool ecosystem | Small but controlled native tools | Broad ecosystem of tool abstractions and integrations |
| Retrieval | Native RAG pipeline with source metadata and eval support | Standard retriever and document interfaces |
| Permission model | Owned by `forge-agent` | Must be bridged through tools or middleware |
| Trace model | Native trace events are explicit runtime artifacts | Usually callback or middleware based |
| Eval workflow | Deterministic platform evals and RAG evals | Possible, but usually needs additional harness design |
| Best use case | Learning and demonstrating agent platform internals | Building LLM applications quickly |
| Project role | Reference implementation | Optional ecosystem adapter |

## Why keep a native runtime

A native runtime makes the agent loop explicit.

It shows this lifecycle:

    task
      -> model call
      -> tool call
      -> tool validation
      -> permission check
      -> tool execution
      -> structured tool result
      -> trace event
      -> next model call or final answer

This is useful because agent platform engineering is not only about getting a model to call a tool. It is about controlling the runtime boundary around model behavior.

The native runtime makes these concerns visible:

- maximum step limits;
- model provider abstraction;
- tool registry ownership;
- tool input validation;
- permission policy;
- workspace boundary;
- structured tool errors;
- trace events;
- deterministic evals;
- runtime selection.

These are platform concerns. They should not disappear behind a framework wrapper.

## Why integrate LangChain

LangChain is valuable because it provides a common ecosystem for LLM applications.

It helps with:

- common tool interfaces;
- retriever and document abstractions;
- model provider integrations;
- agent harnesses;
- middleware;
- callbacks;
- reusable application patterns.

A platform project should be able to interoperate with this ecosystem. The Day 3 integration therefore exposes native tools and retrievers to LangChain instead of rewriting the project around LangChain.

## Tool calling comparison

### Native runtime

In the native runtime, tool calling is controlled by `forge-agent`.

The native flow owns:

- `ToolSchema`;
- `ToolRegistry`;
- tool argument validation;
- permission checks;
- `ToolResult`;
- trace events;
- runtime loop behavior.

The result is deterministic and easy to test.

### LangChain adapter

In the LangChain integration, a native tool is converted into a LangChain-compatible structured tool.

The adapter keeps the native tool as the execution authority:

    LangChain StructuredTool
            |
            v
    forge-agent tool.execute(...)
            |
            v
    forge-agent ToolResult

This preserves:

- tool name;
- tool description;
- structured input schema;
- structured success result;
- structured error result.

The adapter should not swallow tool errors. A failed tool call is still a meaningful result for an agent.

## Retriever comparison

### Native RAG pipeline

The native RAG pipeline owns:

- document loading;
- chunking;
- retrieval;
- ranking;
- context building;
- citations;
- RAG eval.

The project can evaluate retrieval quality through deterministic datasets instead of manually judging final answers.

### LangChain retriever adapter

The LangChain retriever adapter exposes native retrieval results as LangChain `Document` objects.

The adapter maps:

    SearchResult.chunk.content -> Document.page_content
    SearchResult metadata      -> Document.metadata

The important design rule is that metadata must not be lost.

The adapter keeps:

- `source`;
- `source_path`;
- `source_id`;
- `title`;
- `heading_path`;
- `chunk_id`;
- `ordinal`;
- `rank`;
- `score`;
- `retriever`.

This keeps LangChain documents citation-ready, debug-friendly, and eval-friendly.

## Middleware vs runtime hooks

LangChain middleware and native runtime hooks solve similar lifecycle problems.

| Concern | Native runtime / platform hook | LangChain middleware |
|---|---|---|
| Before model call | Build messages, inject context, enforce limits | `before_model` |
| After model call | Validate model output, decide next step | `after_model` |
| Tool call wrapper | Permission, validation, trace, error handling | `wrap_tool_call` |
| Model call wrapper | Retry, fallback, rate limit, trace | `wrap_model_call` |

The difference is ownership.

In `forge-agent`, these controls are platform-native and deterministic.

In LangChain, these controls are usually implemented through middleware, callbacks, or agent configuration.

A future full LangChain agent integration should bridge these two models explicitly.

## Permission, trace, and eval ownership

### Permission

Permission must remain a platform concern.

A LangChain tool wrapper should not bypass:

- workspace boundaries;
- write permissions;
- path validation;
- unsafe operation checks;
- structured denial errors.

The adapter should call native tools rather than directly accessing files, shell commands, or other resources.

### Trace

Trace events should remain explicit.

Native runtime trace events currently record important lifecycle operations. If a future LangChain agent runtime is added, it should map LangChain callbacks or middleware events into the existing trace event model.

### Eval

Eval should remain deterministic where possible.

The native eval workflow is useful because it tests platform behavior without depending on remote model randomness. LangChain agent evals can be added later, but they should not replace deterministic native evals.

## When to use the native runtime

Use the native runtime when the goal is to demonstrate or test:

- agent loop mechanics;
- tool execution semantics;
- permission and workspace safety;
- trace event design;
- deterministic runtime behavior;
- RAG quality evaluation;
- platform boundary decisions.

The native runtime is the best representation of the project's architecture.

## When to use LangChain

Use LangChain when the goal is to:

- integrate with a broad model ecosystem;
- reuse LangChain tools or retrievers;
- build an LLM application quickly;
- experiment with agent harnesses;
- connect to middleware and callback patterns;
- interoperate with teams already using LangChain.

LangChain is best treated as an integration layer for application assembly.

## Why not rewrite forge-agent as LangChain

Rewriting `forge-agent` as a LangChain-only demo would hide the platform engineering skills that the project is meant to demonstrate.

The project is intended to show that the engineer understands:

- Agent Runtime;
- Tool Execution;
- Permission Policy;
- Workspace Guard;
- Trace Events;
- Eval;
- RAG retrieval quality;
- framework integration boundaries.

LangChain is useful, but it is not a substitute for understanding these primitives.

## Final positioning

A concise way to describe the design:

    LangChain is the ecosystem adapter.
    forge-agent native runtime is the platform reference implementation.

This is the key Day 3 engineering message.
