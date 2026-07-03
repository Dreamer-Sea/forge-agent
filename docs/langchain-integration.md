# LangChain Integration

`forge-agent` integrates with LangChain as an optional ecosystem adapter.

The goal is not to rewrite `forge-agent` as a LangChain project. The goal is to expose selected native platform capabilities to the LangChain ecosystem while keeping the core runtime, tool registry, RAG pipeline, permission model, trace events, and eval workflow owned by `forge-agent`.

Architecture:

    forge-agent native abstractions
            |
            v
    LangChain Tool / Retriever ecosystem

## Why integrate with LangChain

LangChain provides a broad ecosystem for LLM applications:

- common tool abstractions;
- retriever and document interfaces;
- agent harnesses;
- model integrations;
- middleware and orchestration utilities.

`forge-agent` already has its own runtime loop, tool execution model, permission boundary, RAG pipeline, trace recorder, and deterministic eval workflow. The LangChain integration is therefore designed as an adapter layer instead of a replacement runtime.

This lets the project demonstrate two capabilities at the same time:

1. the ability to build and reason about lower-level agent platform primitives;
2. the ability to integrate those primitives with a mainstream LLM application ecosystem.

## Optional dependency boundary

LangChain support is optional.

Default installation keeps the native capabilities available without requiring LangChain:

- native runtime;
- tool registry;
- permission guard;
- RAG pipeline;
- trace events;
- eval workflow.

LangChain-specific commands and adapters require the optional extra:

    uv sync --extra langchain

This boundary is intentional:

- LangChain changes quickly, so it should not become a hard dependency of the core runtime.
- Native tests and demos should remain deterministic and lightweight.
- Platform-level primitives should stay usable even when the ecosystem adapter is not installed.
- Integration failure should produce a clear error instead of a low-level import traceback.

The adapter modules avoid top-level LangChain imports where possible. Missing optional dependencies are converted into a clear `LangChainIntegrationError`.

## Tool adapter design

The tool adapter converts a native `forge-agent` tool into a LangChain-compatible structured tool:

    forge-agent Tool
            |
            v
    LangChain StructuredTool

The adapter preserves the important parts of the native tool contract:

| Concern | Adapter behavior |
|---|---|
| Tool name | Uses the native `ToolSchema.name` |
| Description | Uses the native `ToolSchema.description` |
| Input schema | Converts the supported JSON Schema subset into a Pydantic args model |
| Execution | Calls the original `tool.execute(arguments=...)` |
| Error semantics | Returns the original structured `ToolResult` |
| Permission boundary | Does not bypass the original tool implementation |

The adapter does not treat a tool error as a Python exception by default. If the original tool returns `success=false`, that structured result is passed back to the LangChain caller.

This is important because tool errors are part of the agent feedback loop. A model or agent harness can inspect:

- `success`;
- `error_code`;
- `error_message`;
- `safe_detail`;
- `payload`.

Then it can decide how to recover.

## Retriever adapter design

The retriever adapter converts a native `forge-agent` knowledge base into a LangChain retriever:

    forge-agent KnowledgeBase.search(query)
            |
            v
    LangChain Document list

Each `SearchResult` is converted into a LangChain `Document`.

The adapter maps:

| forge-agent | LangChain |
|---|---|
| `chunk.content` | `Document.page_content` |
| chunk metadata | `Document.metadata` |
| result rank | `metadata["rank"]` |
| result score | `metadata["score"]` |
| adapter retriever name | `metadata["retriever"]` |

The metadata intentionally keeps citation and debug fields:

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

This is not just a convenience. If metadata is lost during framework integration, downstream citation, debug, trace, and eval workflows become weaker.

## CLI demo

The Day 3 CLI demo is adapter-level, not a full remote-model agent demo.

List LangChain-compatible tools:

    uv run forge langchain tools

Search a local knowledge base through the LangChain retriever adapter:

    uv run forge langchain rag "workspace guard permission" \
      --knowledge-base examples/knowledge_base \
      --retriever keyword \
      --top-k 3

The command returns LangChain-style document results while preserving source metadata:

- source;
- chunk_id;
- rank;
- score;
- retriever;
- heading.

## Current limitations

The first version intentionally stays small.

Current limitations:

1. The JSON Schema to Pydantic conversion supports only the common subset used by current `forge-agent` tools.
2. The CLI demo exposes adapter behavior, not a full LangChain agent loop.
3. The adapter does not yet implement LangChain middleware hooks for trace, retry, fallback, or policy interception.
4. The retriever adapter delegates retrieval to `KnowledgeBase.search`; it does not reimplement retrieval inside LangChain.
5. Tool trace events remain owned by the native runtime when tools are executed through the native runtime. A full LangChain agent integration would need explicit middleware or callbacks to bridge trace events.

These limitations are deliberate. Day 3 focuses on adapter boundaries and testability rather than remote model behavior.

## Next step: full LangChain Agent integration

A future iteration can add an agent-level integration:

    forge-agent ToolRegistry
            |
            v
    LangChain tools

    forge-agent KnowledgeBase
            |
            v
    LangChain retriever

    LangChain model + prompt + middleware
            |
            v
    LangChain agent harness

The next integration should preserve the same rules:

- LangChain remains optional.
- Native permission checks remain authoritative.
- Tool errors stay structured.
- RAG metadata remains citation-ready.
- Trace and eval hooks are explicit, not implicit.
- Native runtime remains the reference implementation for platform internals.

## Positioning

The integration demonstrates that `forge-agent` can work with LangChain without becoming dependent on LangChain.

LangChain is the ecosystem adapter.

`forge-agent` remains the platform reference implementation.
