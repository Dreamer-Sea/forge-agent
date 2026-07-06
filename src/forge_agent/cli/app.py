from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, cast

import typer

from forge_agent.cli.langchain import langchain_app
from forge_agent.evals import EvalDataset, EvalReport, EvalRunner, RuntimeEvalExecutor
from forge_agent.integrations.langgraph import LangGraphAgentRuntime
from forge_agent.observability import JsonlTraceExporter
from forge_agent.providers.fake import FakeProvider
from forge_agent.rag.evals import (
    RerankerName,
    run_rag_retrieval_eval,
    write_rag_retrieval_json_report,
    write_rag_retrieval_markdown_report,
)
from forge_agent.rag.knowledge_base import KnowledgeBase, RetrieverType
from forge_agent.runtime import RuntimeName
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.security import ToolError, Workspace
from forge_agent.tools.defaults import create_default_tool_registry
from forge_agent.tools.registry import ToolRegistry

app = typer.Typer(help="A minimal Agent Platform demo CLI.")
rag_app = typer.Typer(help="RAG commands.")
app.add_typer(rag_app, name="rag")
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
            help="Runtime backend to use: native or langgraph.",
        ),
    ] = "native",
) -> None:
    """Run one agent task."""

    selected_runtime = _validate_runtime_name(runtime_name)

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

    result = runtime.run(task)

    tool_names = [tool_result.tool_name for tool_result in result.tool_results]
    tools_used = ", ".join(dict.fromkeys(tool_names)) if tool_names else "none"

    typer.echo(f"runtime: {selected_runtime}")
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
            help="Runtime backend to use: native or langgraph.",
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


def _create_runtime(
    *,
    runtime_name: RuntimeName,
    registry: ToolRegistry,
) -> NativeAgentRuntime | LangGraphAgentRuntime:
    if runtime_name == "native":
        return NativeAgentRuntime(
            provider=FakeProvider(),
            tool_registry=registry,
            max_steps=5,
        )

    return LangGraphAgentRuntime(tool_registry=registry)


def _validate_runtime_name(runtime_name: str) -> RuntimeName:
    normalized = runtime_name.strip().lower()

    if normalized == "native":
        return "native"

    if normalized == "langgraph":
        return "langgraph"

    raise typer.BadParameter(
        f"Unknown runtime: {runtime_name}. Supported runtimes: native, langgraph.",
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
