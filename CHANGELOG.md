# Changelog

All notable changes to this project are documented in this file.

This project follows a learning-demo release style rather than a production package release process.

## [0.7.0] - 2026-06-26

### Added

* Added final README packaging for the Agent Platform demo.
* Added final project report under `docs/final-report.md`.
* Added release notes through `CHANGELOG.md`.
* Documented final demo commands for tool calling, RAG with citations, and eval/trace.
* Documented final test matrix and quality baseline.
* Documented interview talking points for runtime, RAG, permission, trace/eval, and LangGraph.
* Documented known limitations and roadmap for production-grade extension.

### Changed

* Reframed the project from a Day 1 runtime skeleton to a completed Agent Platform demo.
* Updated project positioning around platform-level abstractions:

  * Native Agent Runtime
  * Model Provider abstraction
  * Tool Registry
  * RAG Tool
  * Workspace Permission Guard
  * Trace Events
  * Eval Runner
  * LangGraph Runtime Adapter
* Cleaned up outdated milestone language from README.
* Aligned README commands with the current CLI surface:

  * `uv run forge run`
  * `uv run forge rag index`
  * `uv run forge eval`

### Verified

Final local quality gate:

```text
uv run pytest -v
uv run mypy src tests
uv run ruff check .
uv run ruff format --check .
uv run forge --help
```

Verified baseline:

```text
pytest: 138 passed
mypy: success, no issues found
ruff check: all checks passed
ruff format: all files formatted
forge --help: exits successfully
```

## [0.6.0] - 2026-06-25

### Added

* Added end-to-end demo packaging for the Agent Platform.
* Added architecture diagrams and documentation for runtime, RAG, evaluation, trace, and security.
* Added demo commands for tool calling, RAG, and eval.
* Added final demo scope documentation.

### Changed

* Improved README and documentation for demo-oriented review.
* Clarified demo limitations and platform boundaries.

## [0.5.0] - 2026-06-25

### Added

* Added evaluation runner for deterministic JSONL eval cases.
* Added eval metrics and reporting.
* Added trace export support.
* Added eval CLI command.
* Added tests for eval runner, metrics, reports, and trace export.

### Changed

* Improved agent behavior verification through expected tools, expected content, expected sources, and stop reasons.

## [0.4.0] - 2026-06-24

### Added

* Added LangGraph runtime adapter.
* Added workflow routing for direct agent and RAG paths.
* Added runtime selection support.
* Added tests for native and LangGraph runtime compatibility.

### Changed

* Introduced runtime protocol to keep native and framework-backed runtimes interchangeable.

## [0.3.0] - 2026-06-23

### Added

* Added workspace guard for local file access.
* Added permission policy.
* Added structured permission errors.
* Added trace events for permission decisions.
* Added tests for path traversal, absolute path escape, symlink escape, and denied writes.

### Changed

* File tools now access files through workspace validation and permission checks.

## [0.2.0] - 2026-06-23

### Added

* Added local Markdown RAG pipeline.
* Added Markdown document loader.
* Added heading-aware chunker.
* Added deterministic keyword retriever.
* Added grounded context builder with citations.
* Added RAG tool for agent runtime integration.
* Added example local knowledge base.
* Added RAG CLI command.
* Added RAG tests and retrieval-quality assessment documentation.

## [0.1.0] - 2026-06-22

### Added

* Added initial native Agent Runtime.
* Added model provider abstraction.
* Added deterministic fake provider.
* Added tool abstraction and tool registry.
* Added basic built-in tools.
* Added CLI command for running an agent task.
* Added structured runtime and tool results.
* Added initial trace event model.
* Added pytest, mypy, and Ruff quality setup.
* Added Apache License 2.0.
