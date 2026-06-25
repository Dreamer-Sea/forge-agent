from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from forge_agent.evals import EvalDataset, EvalReport, EvalRunner, RuntimeEvalExecutor
from forge_agent.integrations.langgraph import LangGraphAgentRuntime
from forge_agent.observability import JsonlTraceExporter
from forge_agent.providers.fake import FakeProvider
from forge_agent.rag.knowledge_base import KnowledgeBase
from forge_agent.runtime import RuntimeName
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.security import ToolError, Workspace
from forge_agent.tools.defaults import create_default_tool_registry
from forge_agent.tools.registry import ToolRegistry

app = typer.Typer(help="A minimal Agent Platform demo CLI.")
rag_app = typer.Typer(help="RAG commands.")
app.add_typer(rag_app, name="rag")


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

    typer.echo(f"Runtime: {selected_runtime}")
    typer.echo(f"Final answer: {result.final_answer or ''}")
    typer.echo(f"Stopped reason: {result.stopped_reason}")
    typer.echo(f"Steps: {result.steps}")

    if result.error_message is not None:
        typer.echo(f"Error: {result.error_message}")

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
    typer.echo(
        "tool_call_success_rate: "
        f"{_format_cli_rate(metrics.tool_call_success_rate)}"
    )
    typer.echo(
        "expected_contains_pass_rate: "
        f"{_format_cli_rate(metrics.expected_contains_pass_rate)}"
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
        typer.echo(
            f"- {document.metadata.relative_path}: {document.metadata.title}"
        )


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
