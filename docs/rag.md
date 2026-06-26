# RAG Pipeline

`forge-agent` includes a local Markdown-based RAG pipeline for demonstrating grounded agent answers with citations.

The goal is not to build a production vector database. The goal is to show the platform shape of a retrieval-augmented agent: document loading, chunking, retrieval, answer grounding, citation, and traceability.

## Responsibilities

The RAG pipeline is responsible for:

- Loading Markdown documents from a local knowledge base.
- Splitting documents into searchable chunks.
- Retrieving relevant chunks for a user query.
- Passing retrieved context to the agent runtime as tool output.
- Producing answers that include citations.
- Recording retrieval activity in trace events.

## Demo Command

Index the example knowledge base:

    uv run forge rag index examples/knowledge_base

Ask a grounded question:

    uv run forge run "According to the knowledge base, how does the permission system work?"

Expected output summary:

    runtime: native
    tools_used: search_knowledge_base
    stopped_reason: completed
    final_answer: <grounded answer with citations>

## Citation Contract

A RAG answer must include enough source information for a reviewer to verify the answer.

For the Day 6 demo, citations are file-based and chunk-based. A citation may include:

- Source file path, such as `security.md`.
- Markdown heading path, such as `Security > Permission System`.
- Chunk identifier, such as `markdown:security.md:chunk:2:<hash>`.
- A compact source marker emitted by the knowledge-base search tool.

Example citation shape:

    [source: security.md#Security > Permission System markdown:security.md:chunk:2:<hash>]

This format is intentionally simple. It keeps the demo local, deterministic, and easy to inspect.

## Runtime Flow

A typical RAG run follows this flow:

1. The user asks a question that requires project knowledge.
2. The model provider requests the `search_knowledge_base` tool.
3. The runtime checks permissions before executing the tool.
4. The knowledge-base search tool returns matching chunks with source metadata.
5. The runtime sends the retrieved context back to the model provider.
6. The final answer includes grounded content and citations.

## Traceability

A RAG run should make retrieval observable.

The trace should show:

- A model call or workflow route.
- A `search_knowledge_base` tool call.
- A permission check before tool execution.
- A tool result containing retrieved context.
- A final answer generated from retrieved context.

## Design Trade-offs

This implementation favors determinism and readability over retrieval sophistication.

It intentionally avoids:

- Production vector databases.
- External embedding services.
- Multi-tenant indexing.
- Distributed retrieval.
- Complex reranking.

These are valid production concerns, but they are outside the Day 6 demo scope.
