from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, cast

import typer

from forge_agent.cli.langchain import langchain_app
from forge_agent.evals import EvalDataset, EvalReport, EvalRunner, RuntimeEvalExecutor
from forge_agent.integrations.langgraph import LangGraphAgentRuntime
from forge_agent.memory import JsonlMemoryStore, MemoryScope, MemoryStore, MemoryType
from forge_agent.observability import JsonlTraceExporter
from forge_agent.providers.fake import FakeProvider
from forge_agent.rag.evals import (
    RerankerName,
    run_rag_retrieval_eval,
    write_rag_retrieval_json_report,
    write_rag_retrieval_markdown_report,
)
from forge_agent.rag.knowledge_base import KnowledgeBase, RetrieverType
from forge_agent.runtime import RunConfig, RuntimeName
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.runtime.planning_runtime import PlanningRuntime
from forge_agent.runtime.reflection_runtime import ReflectionRuntime
from forge_agent.security import ToolError, Workspace
from forge_agent.tools.defaults import create_default_tool_registry
from forge_agent.tools.registry import ToolRegistry

app = typer.Typer(help="A minimal Agent Platform demo CLI.")
rag_app = typer.Typer(help="RAG commands.")
memory_app = typer.Typer(help="Long-term memory commands.")
app.add_typer(rag_app, name="rag")
app.add_typer(memory_app, name="memory")
app.add_typer(langchain_app, name="langchain")


@app.callback()
def callback() -> None:
    """Forge Agent CLI."""


DEFAULT_KNOWLEDGE_BASE_PATH = Path("examples/knowledge_base")


@app.command()
def run(
    task: str,
    knowledge_base: Annotated[
        Path,
        typer.Option(
            "--knowledge-base",
            "-k",
            help="Path to a local Markdown knowledge base.",
        ),
    ] = DEFAULT_KNOWLEDGE_BASE_PATH,
    runtime_name: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime backend to use: native, langgraph, planning, or reflection.",
        ),
    ] = "native",
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Maximum model steps for the selected runtime.",
        ),
    ] = 5,
    memory_path: Annotated[
        Path | None,
        typer.Option(
            "--memory-path",
            help="Path to a local JSONL memory directory.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Session id used to isolate session-scoped memories.",
        ),
    ] = None,
    memory_top_k: Annotated[
        int,
        typer.Option(
            "--memory-top-k",
            help="Maximum number of memories recalled before a run.",
        ),
    ] = 5,
) -> None:
    """Run one agent task."""

    selected_runtime = _validate_runtime_name(runtime_name)

    workspace = Workspace(Path.cwd())
    if memory_top_k <= 0:
        raise typer.BadParameter(
            "memory-top-k must be greater than 0",
            param_hint="--memory-top-k",
        )
    memory_store = _load_memory_store_if_requested(
        memory_path,
        workspace=workspace,
    )
    memory_scope_id = _normalize_optional_text(session_id)
    local_knowledge_base = _load_knowledge_base_if_exists(
        knowledge_base,
        workspace=workspace,
    )
    safe_knowledge_base_path = _safe_path_if_inside(
        knowledge_base,
        workspace=workspace,
    )

    registry = create_default_tool_registry(
        knowledge_base=local_knowledge_base,
        workspace=workspace,
        safe_knowledge_base_path=safe_knowledge_base_path,
    )
    runtime = _create_runtime(
        runtime_name=selected_runtime,
        registry=registry,
        memory_store=memory_store,
        memory_scope_id=memory_scope_id,
        memory_top_k=memory_top_k,
    )

    result = runtime.run(task, config=RunConfig(max_steps=max_steps))

    tool_names = [tool_result.tool_name for tool_result in result.tool_results]
    tools_used = ", ".join(dict.fromkeys(tool_names)) if tool_names else "none"

    typer.echo(f"runtime: {selected_runtime}")
    if memory_store is not None:
        typer.echo("memory: enabled")
        typer.echo(f"memory_scope: {MemoryScope.SESSION.value}")
        typer.echo(f"memory_session_id: {memory_scope_id or 'default'}")
    typer.echo(f"tools_used: {tools_used}")
    typer.echo(f"stopped_reason: {result.stopped_reason}")
    typer.echo(f"final_answer: {result.final_answer or ''}")
    typer.echo(f"steps: {result.steps}")

    if result.error_message is not None:
        typer.echo(f"error: {result.error_message}")

    typer.echo("")
    typer.echo("Tool calls:")

    if not result.tool_results:
        typer.echo("- none")
    else:
        for tool_result in result.tool_results:
            status = "SUCCESS" if tool_result.success else "FAILED"
            typer.echo(f"- {tool_result.tool_name}: {status}")

    typer.echo("")
    typer.echo("Trace events:")
    for event in result.trace_events:
        typer.echo(f"- {event.event_type}")


@app.command("eval")
def eval_command(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to a JSONL eval dataset."),
    ],
    knowledge_base: Annotated[
        Path,
        typer.Option(
            "--knowledge-base",
            "-k",
            help="Path to a local Markdown knowledge base.",
        ),
    ] = DEFAULT_KNOWLEDGE_BASE_PATH,
    runtime_name: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime backend to use: native, langgraph, planning, or reflection.",
        ),
    ] = "native",
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Path to write the markdown eval report.",
        ),
    ] = Path("reports/eval-report.md"),
    trace_out: Annotated[
        Path,
        typer.Option(
            "--trace-out",
            help="Path to write JSONL trace events.",
        ),
    ] = Path("reports/traces.jsonl"),
    json_output: Annotated[
        Path | None,
        typer.Option(
            "--json-output",
            help="Optional path to write the JSON eval report.",
        ),
    ] = None,
) -> None:
    """Run deterministic agent evals from a JSONL dataset."""

    selected_runtime = _validate_runtime_name(runtime_name)

    try:
        dataset = EvalDataset.load_jsonl(dataset_path)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    workspace = Workspace(Path.cwd())
    local_knowledge_base = _load_knowledge_base_if_exists(
        knowledge_base,
        workspace=workspace,
    )
    safe_knowledge_base_path = _safe_path_if_inside(
        knowledge_base,
        workspace=workspace,
    )

    registry = create_default_tool_registry(
        knowledge_base=local_knowledge_base,
        workspace=workspace,
        safe_knowledge_base_path=safe_knowledge_base_path,
    )
    runtime = _create_runtime(
        runtime_name=selected_runtime,
        registry=registry,
    )

    executor = RuntimeEvalExecutor(
        runtime=runtime,
        runtime_name=selected_runtime,
    )
    suite = asyncio.run(EvalRunner(executor).run_dataset(dataset))

    JsonlTraceExporter(trace_out).export(suite.trace_events)

    report = EvalReport.from_suite(
        suite,
        trace_file=trace_out,
    )
    report.write_markdown(output)

    if json_output is not None:
        report.write_json(json_output)

    metrics = report.metrics

    typer.echo(f"case_count: {metrics.case_count}")
    typer.echo(f"success_rate: {_format_cli_rate(metrics.success_rate)}")
    typer.echo(f"tool_call_success_rate: {_format_cli_rate(metrics.tool_call_success_rate)}")
    typer.echo(
        f"expected_contains_pass_rate: {_format_cli_rate(metrics.expected_contains_pass_rate)}"
    )
    typer.echo(f"failed_cases: {metrics.failed_case_count}")
    typer.echo(f"trace_file: {trace_out}")
    typer.echo(f"report_file: {output}")


@rag_app.command("index")
def rag_index(path: Path) -> None:
    """Index a local Markdown knowledge base."""

    workspace = Workspace(Path.cwd())

    try:
        resolved_path = workspace.resolve_user_path(
            path,
            tool_name="rag_index",
        )
    except ToolError as error:
        typer.echo(f"Error: {error.message}", err=True)
        typer.echo(f"Reason: {error.reason}", err=True)
        typer.echo(f"Code: {error.error_code}", err=True)
        raise typer.Exit(code=1) from error

    if not resolved_path.exists():
        typer.echo(
            f"Error: Path does not exist: {workspace.safe_display(resolved_path)}",
            err=True,
        )
        raise typer.Exit(code=1)

    if not resolved_path.is_dir():
        typer.echo(
            f"Error: Path is not a directory: {workspace.safe_display(resolved_path)}",
            err=True,
        )
        raise typer.Exit(code=1)

    knowledge_base = KnowledgeBase.from_directory(resolved_path)
    safe_path = workspace.safe_display(resolved_path)

    typer.echo(f"Knowledge base: {safe_path}")
    typer.echo(f"Documents: {len(knowledge_base.index.documents)}")
    typer.echo(f"Chunks: {len(knowledge_base.index.chunks)}")
    typer.echo("")
    typer.echo("Sources:")

    for document in knowledge_base.index.documents:
        typer.echo(f"- {document.metadata.relative_path}: {document.metadata.title}")


@rag_app.command("search")
def rag_search(
    query: Annotated[
        str,
        typer.Argument(help="Query to search in the local Markdown knowledge base."),
    ],
    knowledge_base: Annotated[
        Path,
        typer.Option(
            "--knowledge-base",
            "-k",
            help="Path to a local Markdown knowledge base.",
        ),
    ] = DEFAULT_KNOWLEDGE_BASE_PATH,
    retriever_type: Annotated[
        str,
        typer.Option(
            "--retriever",
            help="Retriever backend to use: keyword, vector, or hybrid.",
        ),
    ] = "keyword",
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="Maximum number of retrieval results.",
        ),
    ] = 3,
) -> None:
    """Search a local Markdown knowledge base."""
    selected_retriever = _validate_retriever_type(retriever_type)

    if top_k <= 0:
        raise typer.BadParameter(
            "top-k must be greater than 0",
            param_hint="--top-k",
        )

    workspace = Workspace(Path.cwd())
    try:
        resolved_path = workspace.resolve_user_path(
            knowledge_base,
            tool_name="rag_search",
        )
    except ToolError as error:
        typer.echo(f"Error: {error.message}", err=True)
        typer.echo(f"Reason: {error.reason}", err=True)
        typer.echo(f"Code: {error.error_code}", err=True)
        raise typer.Exit(code=1) from error

    if not resolved_path.exists():
        typer.echo(
            f"Error: Path does not exist: {workspace.safe_display(resolved_path)}",
            err=True,
        )
        raise typer.Exit(code=1)

    if not resolved_path.is_dir():
        typer.echo(
            f"Error: Path is not a directory: {workspace.safe_display(resolved_path)}",
            err=True,
        )
        raise typer.Exit(code=1)

    local_knowledge_base = KnowledgeBase.from_directory(
        resolved_path,
        retriever_type=selected_retriever,
        context_max_chunks=top_k,
        default_top_k=top_k,
    )
    search = local_knowledge_base.search(query, top_k=top_k)

    typer.echo(f"Knowledge base: {workspace.safe_display(resolved_path)}")
    typer.echo(f"Retriever: {selected_retriever}")
    typer.echo(f"Query: {query}")
    typer.echo(f"Results: {len(search.results)}")

    if not search.results:
        typer.echo("")
        typer.echo("No results.")
        return

    typer.echo("")
    typer.echo("Context:")
    typer.echo(search.built_context.context)

    typer.echo("")
    typer.echo("Sources:")
    for result in search.results:
        heading = " > ".join(result.chunk.metadata.heading_path)
        typer.echo(
            f"- #{result.rank} score={result.score:.4f} "
            f"source={result.chunk.metadata.relative_path} "
            f"heading={heading}"
        )


@rag_app.command("eval")
def rag_eval(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to a RAG retrieval JSONL eval dataset."),
    ],
    knowledge_base: Annotated[
        Path,
        typer.Option(
            "--knowledge-base",
            "-k",
            help="Path to a local Markdown knowledge base.",
        ),
    ] = DEFAULT_KNOWLEDGE_BASE_PATH,
    retriever_type: Annotated[
        str,
        typer.Option(
            "--retriever",
            help="Retriever backend to use: keyword, vector, or hybrid.",
        ),
    ] = "keyword",
    reranker_name: Annotated[
        str,
        typer.Option(
            "--reranker",
            help="Reranker to use: none, identity, or keyword-overlap.",
        ),
    ] = "none",
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            help="Override each eval case top-k retrieval depth.",
        ),
    ] = None,
    top_n: Annotated[
        int | None,
        typer.Option(
            "--top-n",
            help="Keep only the top-n final results after optional reranking.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Write a Markdown RAG retrieval eval report.",
        ),
    ] = None,
    json_output: Annotated[
        Path | None,
        typer.Option(
            "--json-output",
            help="Write a JSON RAG retrieval eval result file.",
        ),
    ] = None,
) -> None:
    """Run deterministic RAG retrieval evals."""
    selected_retriever = _validate_retriever_type(retriever_type)
    selected_reranker = _validate_reranker_name(reranker_name)

    if top_k is not None and top_k <= 0:
        raise typer.BadParameter(
            "top-k must be greater than 0",
            param_hint="--top-k",
        )
    if top_n is not None and top_n <= 0:
        raise typer.BadParameter(
            "top-n must be greater than 0",
            param_hint="--top-n",
        )

    workspace = Workspace(Path.cwd())
    try:
        resolved_dataset_path = workspace.resolve_user_path(
            dataset_path,
            tool_name="rag_eval",
        )
        resolved_knowledge_base_path = workspace.resolve_user_path(
            knowledge_base,
            tool_name="rag_eval",
        )
        resolved_output_path = (
            workspace.resolve_user_path(output, tool_name="rag_eval")
            if output is not None
            else None
        )
        resolved_json_output_path = (
            workspace.resolve_user_path(json_output, tool_name="rag_eval")
            if json_output is not None
            else None
        )
    except ToolError as error:
        typer.echo(f"Error: {error.message}", err=True)
        typer.echo(f"Reason: {error.reason}", err=True)
        typer.echo(f"Code: {error.error_code}", err=True)
        raise typer.Exit(code=1) from error

    if not resolved_dataset_path.exists():
        typer.echo(
            f"Error: Dataset does not exist: {workspace.safe_display(resolved_dataset_path)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if not resolved_dataset_path.is_file():
        typer.echo(
            f"Error: Dataset is not a file: {workspace.safe_display(resolved_dataset_path)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if not resolved_knowledge_base_path.exists():
        typer.echo(
            f"Error: Knowledge base does not exist: "
            f"{workspace.safe_display(resolved_knowledge_base_path)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if not resolved_knowledge_base_path.is_dir():
        typer.echo(
            f"Error: Knowledge base is not a directory: "
            f"{workspace.safe_display(resolved_knowledge_base_path)}",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        suite = run_rag_retrieval_eval(
            dataset_path=resolved_dataset_path,
            knowledge_base_path=resolved_knowledge_base_path,
            retriever_type=selected_retriever,
            reranker_name=selected_reranker,
            top_k=top_k,
            top_n=top_n,
        )
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    metrics = suite.summary

    typer.echo("RAG retrieval eval")
    typer.echo(f"Dataset: {workspace.safe_display(resolved_dataset_path)}")
    typer.echo(f"Knowledge base: {workspace.safe_display(resolved_knowledge_base_path)}")
    typer.echo(f"Retriever: {suite.retriever_type}")
    typer.echo(f"Reranker: {suite.reranker_name}")
    typer.echo(f"Cases: {metrics.total_cases}")
    typer.echo(f"Answerable cases: {metrics.answerable_cases}")
    typer.echo(f"No-answer cases: {metrics.no_answer_cases}")
    typer.echo(f"source_hit_rate: {_format_cli_rate(metrics.source_hit_rate)}")
    typer.echo(f"recall_at_k: {_format_cli_rate(metrics.recall_at_k)}")
    typer.echo(f"mrr: {_format_cli_rate(metrics.mrr)}")
    typer.echo(f"term_hit_rate: {_format_cli_rate(metrics.term_hit_rate)}")
    typer.echo(f"no_answer_accuracy: {_format_cli_rate(metrics.no_answer_accuracy)}")
    typer.echo(f"citation_presence_rate: {_format_cli_rate(metrics.citation_presence_rate)}")
    typer.echo("")
    typer.echo("Cases:")
    for evaluation in suite.evaluations:
        retrieved_sources = ", ".join(evaluation.retrieved_sources) or "none"
        typer.echo(
            f"- {evaluation.case_id}: "
            f"source_hit={evaluation.source_hit} "
            f"recall_at_k={evaluation.recall_at_k:.2f} "
            f"mrr={evaluation.reciprocal_rank:.2f} "
            f"term_hit_rate={_format_cli_rate(evaluation.term_hit_rate)} "
            f"no_answer_correct={evaluation.no_answer_correct} "
            f"sources={retrieved_sources}"
        )

    if resolved_output_path is not None:
        write_rag_retrieval_markdown_report(suite, resolved_output_path)
        typer.echo(f"Markdown report written: {workspace.safe_display(resolved_output_path)}")

    if resolved_json_output_path is not None:
        write_rag_retrieval_json_report(suite, resolved_json_output_path)
        typer.echo(f"JSON report written: {workspace.safe_display(resolved_json_output_path)}")



@memory_app.command("list")
def memory_list(
    memory_path: Annotated[
        Path,
        typer.Option(
            "--memory-path",
            help="Path to a local JSONL memory directory.",
        ),
    ] = Path(".memory"),
    scope_name: Annotated[
        str | None,
        typer.Option(
            "--scope",
            help="Optional memory scope: global, user, session, or project.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Optional session id filter for session-scoped memories.",
        ),
    ] = None,
    memory_type_name: Annotated[
        str | None,
        typer.Option(
            "--type",
            help="Optional memory type: semantic, episodic, or summary.",
        ),
    ] = None,
) -> None:
    """List persisted long-term memories."""

    workspace = Workspace(Path.cwd())
    store = _load_memory_store(memory_path, workspace=workspace)
    scope = _validate_optional_memory_scope(scope_name)
    memory_type = _validate_optional_memory_type(memory_type_name)
    scope_id = _normalize_optional_text(session_id)

    records = store.list(
        scope=scope,
        scope_id=scope_id,
        type_filter=memory_type,
    )

    typer.echo(f"Memory path: {workspace.safe_display(memory_path)}")
    typer.echo(f"Records: {len(records)}")
    if not records:
        typer.echo("")
        typer.echo("No memories.")
        return

    typer.echo("")
    typer.echo("Memories:")
    for record in records:
        scope_display = (
            f"{record.scope.value}:{record.scope_id}"
            if record.scope_id is not None
            else record.scope.value
        )
        typer.echo(
            f"- id={record.id} type={record.type.value} "
            f"scope={scope_display} source={record.source}"
        )
        typer.echo(f"  {record.content}")


@memory_app.command("search")
def memory_search(
    query: Annotated[
        str,
        typer.Argument(help="Query to search in long-term memory."),
    ],
    memory_path: Annotated[
        Path,
        typer.Option(
            "--memory-path",
            help="Path to a local JSONL memory directory.",
        ),
    ] = Path(".memory"),
    scope_name: Annotated[
        str | None,
        typer.Option(
            "--scope",
            help="Optional memory scope: global, user, session, or project.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Optional session id filter for session-scoped memories.",
        ),
    ] = None,
    memory_type_name: Annotated[
        str | None,
        typer.Option(
            "--type",
            help="Optional memory type: semantic, episodic, or summary.",
        ),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="Maximum number of search results.",
        ),
    ] = 5,
) -> None:
    """Search persisted long-term memories."""

    if top_k <= 0:
        raise typer.BadParameter(
            "top-k must be greater than 0",
            param_hint="--top-k",
        )

    workspace = Workspace(Path.cwd())
    store = _load_memory_store(memory_path, workspace=workspace)
    scope = _validate_optional_memory_scope(scope_name)
    memory_type = _validate_optional_memory_type(memory_type_name)
    scope_id = _normalize_optional_text(session_id)

    results = store.search(
        query,
        scope=scope,
        top_k=top_k,
        scope_id=scope_id,
        type_filter=memory_type,
    )

    typer.echo(f"Memory path: {workspace.safe_display(memory_path)}")
    typer.echo(f"Query: {query}")
    typer.echo(f"Results: {len(results)}")
    if not results:
        typer.echo("")
        typer.echo("No memories found.")
        return

    typer.echo("")
    typer.echo("Memories:")
    for result in results:
        record = result.record
        scope_display = (
            f"{record.scope.value}:{record.scope_id}"
            if record.scope_id is not None
            else record.scope.value
        )
        typer.echo(
            f"- #{result.rank} score={result.score:.4f} "
            f"type={record.type.value} scope={scope_display} "
            f"source={record.source}"
        )
        typer.echo(f"  {record.content}")


def _load_memory_store_if_requested(
    memory_path: Path | None,
    *,
    workspace: Workspace,
) -> JsonlMemoryStore | None:
    if memory_path is None:
        return None
    return _load_memory_store(memory_path, workspace=workspace)


def _load_memory_store(
    memory_path: Path,
    *,
    workspace: Workspace,
) -> JsonlMemoryStore:
    try:
        resolved_path = workspace.resolve_user_path(
            memory_path,
            tool_name="memory",
        )
    except ToolError as error:
        typer.echo(f"Error: {error.message}", err=True)
        typer.echo(f"Reason: {error.reason}", err=True)
        typer.echo(f"Code: {error.error_code}", err=True)
        raise typer.Exit(code=1) from error

    return JsonlMemoryStore(resolved_path)


def _validate_optional_memory_scope(scope_name: str | None) -> MemoryScope | None:
    normalized = _normalize_optional_text(scope_name)
    if normalized is None:
        return None
    return _validate_memory_scope(normalized)


def _validate_memory_scope(scope_name: str) -> MemoryScope:
    normalized = scope_name.strip().lower()
    if normalized in {"global", "user", "session", "project"}:
        return cast(MemoryScope, normalized)

    raise typer.BadParameter(
        f"Unknown memory scope: {scope_name}.\n"
        "Supported memory scopes: global, user, session, project.",
        param_hint="--scope",
    )


def _validate_optional_memory_type(memory_type_name: str | None) -> MemoryType | None:
    normalized = _normalize_optional_text(memory_type_name)
    if normalized is None:
        return None
    return _validate_memory_type(normalized)


def _validate_memory_type(memory_type_name: str) -> MemoryType:
    normalized = memory_type_name.strip().lower()
    if normalized in {"semantic", "episodic", "summary"}:
        return cast(MemoryType, normalized)

    raise typer.BadParameter(
        f"Unknown memory type: {memory_type_name}.\n"
        "Supported memory types: semantic, episodic, summary.",
        param_hint="--type",
    )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    return normalized

def _create_runtime(
    *,
    runtime_name: RuntimeName,
    registry: ToolRegistry,
    memory_store: MemoryStore | None = None,
    memory_scope_id: str | None = None,
    memory_top_k: int = 5,
) -> NativeAgentRuntime | LangGraphAgentRuntime | PlanningRuntime | ReflectionRuntime:
    if runtime_name == "native":
        return NativeAgentRuntime(
            provider=FakeProvider(),
            tool_registry=registry,
            max_steps=5,
            memory_store=memory_store,
            memory_scope=MemoryScope.SESSION,
            memory_scope_id=memory_scope_id,
            memory_top_k=memory_top_k,
        )
    if runtime_name == "planning":
        return PlanningRuntime(
            provider=FakeProvider(),
            tool_registry=registry,
            max_steps=5,
        )
    if runtime_name == "reflection":
        base_runtime = NativeAgentRuntime(
            provider=FakeProvider(),
            tool_registry=registry,
            max_steps=5,
            memory_store=memory_store,
            memory_scope=MemoryScope.SESSION,
            memory_scope_id=memory_scope_id,
            memory_top_k=memory_top_k,
        )
        return ReflectionRuntime(base_runtime)

    return LangGraphAgentRuntime(tool_registry=registry)


def _validate_runtime_name(runtime_name: str) -> RuntimeName:
    normalized = runtime_name.strip().lower()
    if normalized == "native":
        return "native"
    if normalized == "langgraph":
        return "langgraph"
    if normalized == "planning":
        return "planning"
    if normalized == "reflection":
        return "reflection"

    raise typer.BadParameter(
        f"Unknown runtime: {runtime_name}.\n"
        "Supported runtimes: native, langgraph, planning, reflection.",
        param_hint="--runtime",
    )


def _validate_retriever_type(retriever_type: str) -> RetrieverType:
    normalized = retriever_type.strip().lower()
    if normalized in {"keyword", "vector", "hybrid"}:
        return cast(RetrieverType, normalized)

    raise typer.BadParameter(
        f"Unknown retriever: {retriever_type}. Supported retrievers: keyword, vector, hybrid.",
        param_hint="--retriever",
    )


def _validate_reranker_name(reranker_name: str) -> RerankerName:
    normalized = reranker_name.strip().lower()
    if normalized in {"none", "identity", "keyword-overlap"}:
        return cast(RerankerName, normalized)
    raise typer.BadParameter(
        f"Unknown reranker: {reranker_name}.\n"
        "Supported rerankers: none, identity, keyword-overlap.",
        param_hint="--reranker",
    )


def _load_knowledge_base_if_exists(
    path: Path,
    *,
    workspace: Workspace,
) -> KnowledgeBase | None:
    try:
        resolved_path = workspace.resolve_user_path(
            path,
            tool_name="knowledge_base",
        )
    except ToolError:
        return None

    if not resolved_path.exists():
        return None

    if not resolved_path.is_dir():
        return None

    return KnowledgeBase.from_directory(resolved_path)


def _safe_path_if_inside(
    path: Path,
    *,
    workspace: Workspace,
) -> str | None:
    try:
        resolved_path = workspace.resolve_user_path(
            path,
            tool_name="knowledge_base",
        )
    except ToolError:
        return None

    return workspace.safe_display(resolved_path)


def _format_cli_rate(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    app()
