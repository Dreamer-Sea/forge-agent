from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from forge_agent.providers.fake import FakeProvider
from forge_agent.rag.knowledge_base import KnowledgeBase
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.tools.defaults import create_default_tool_registry

app = typer.Typer(help="A minimal Agent Platform demo CLI.")
rag_app = typer.Typer(help="RAG commands.")
app.add_typer(rag_app, name="rag")


@app.callback()
def callback() -> None:
    """Forge Agent CLI."""


DEFAULT_KNOWLEDGE_BASE_PATH = Path("examples/knowledge_base")


@app.command()
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
) -> None:
    """Run one agent task."""

    local_knowledge_base = _load_knowledge_base_if_exists(knowledge_base)
    registry = create_default_tool_registry(knowledge_base=local_knowledge_base)
    runtime = NativeAgentRuntime(
        provider=FakeProvider(),
        tool_registry=registry,
        max_steps=5,
    )

    result = runtime.run(task)

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


@rag_app.command("index")
def rag_index(path: Path) -> None:
    """Index a local Markdown knowledge base."""

    knowledge_base = KnowledgeBase.from_directory(path)

    typer.echo(f"Knowledge base: {path}")
    typer.echo(f"Documents: {len(knowledge_base.index.documents)}")
    typer.echo(f"Chunks: {len(knowledge_base.index.chunks)}")
    typer.echo("")
    typer.echo("Sources:")

    for document in knowledge_base.index.documents:
        typer.echo(
            f"- {document.metadata.relative_path}: {document.metadata.title}"
        )


def _load_knowledge_base_if_exists(path: Path) -> KnowledgeBase | None:
    if not path.exists():
        return None

    if not path.is_dir():
        return None

    return KnowledgeBase.from_directory(path)


def main() -> None:
    app()
