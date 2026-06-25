# Day 6 Demo Scope

Day 6 focuses on packaging `forge-agent` into an end-to-end Agent Platform demo that is easy to run, explain, and evaluate.

The goal is not to add more platform features. The goal is to stabilize three demo paths and document the design clearly enough for GitHub visitors, interviewers, and reviewers.

## Demo Scope

### Demo 1: Tool Calling

Shows that the agent runtime can:

- Receive a natural-language task.
- Ask a model provider for tool calls.
- Execute tools through the tool registry.
- Return a structured final answer.
- Report runtime metadata such as tools used and stop reason.

Primary command:

    uv run forge run "Read the project README and summarize the architecture."

### Demo 2: RAG + Citation

Shows that the platform can:

- Index local Markdown knowledge base documents.
- Retrieve relevant chunks.
- Ground an answer in retrieved context.
- Return citations so the answer is auditable.

Primary commands:

    uv run forge rag index examples/knowledge_base
    uv run forge run "According to the knowledge base, how does the permission system work?"

### Demo 3: Eval + Trace

Shows that the platform can:

- Run deterministic evaluation cases.
- Execute cases through the runtime.
- Record traces for debugging.
- Generate a readable evaluation report.

Primary command:

    uv run forge eval examples/evals/agent_platform.jsonl

## Non-goals

Day 6 intentionally does not include:

- Web UI.
- Multi-tenant SaaS platform.
- Production-grade vector database.
- Docker-based sandbox.
- MCP marketplace.
- Distributed tracing backend.
- Complex long-term memory.
- Production authentication and authorization.
- Additional model provider integrations.

## Success Criteria

Day 6 is complete when the following commands pass from a clean checkout:

    uv sync
    uv run pytest -q
    uv run mypy src tests
    uv run ruff check .
    uv run forge --help
    uv run forge run "Read the project README and summarize the architecture."
    uv run forge rag index examples/knowledge_base
    uv run forge run "According to the knowledge base, how does the permission system work?"
    uv run forge eval examples/evals/agent_platform.jsonl

## Review Checklist

Before moving to the next Day 6 stage, verify:

- The demo scope is clear.
- The non-goals are explicit.
- No new platform features are introduced.
- The three demo paths match the Day 6 plan.
- The success criteria are executable from the command line.
