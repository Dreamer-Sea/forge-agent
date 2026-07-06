from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from forge_agent.integrations.langchain.errors import LangChainIntegrationError
from forge_agent.integrations.langchain.retrievers import (
    forge_knowledge_base_to_langchain_retriever,
)
from forge_agent.integrations.langchain.tools import forge_registry_to_langchain_tools
from forge_agent.rag.knowledge_base import KnowledgeBase, RetrieverType
from forge_agent.security import ToolError, Workspace
from forge_agent.tools.defaults import create_default_tool_registry

langchain_app = typer.Typer(help="LangChain adapter demo commands.")


@langchain_app.command("tools")
def langchain_tools() -> None:
    """List forge-agent tools converted to LangChain-compatible tools."""
    registry = create_default_tool_registry()

    try:
        tools = forge_registry_to_langchain_tools(registry)
    except LangChainIntegrationError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo("LangChain-compatible tools:")
    for tool in tools:
        typer.echo(f"- {tool.name}: {tool.description}")


@langchain_app.command("rag")
def langchain_rag(
    query: Annotated[
        str,
        typer.Argument(help="Query to search through the LangChain retriever adapter."),
    ],
    knowledge_base: Annotated[
        Path,
        typer.Option(
            "--knowledge-base",
            "-k",
            help="Path to a local Markdown knowledge base.",
        ),
    ] = Path("examples/knowledge_base"),
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
            help="Maximum number of LangChain Document results.",
        ),
    ] = 3,
) -> None:
    """Search a forge-agent knowledge base through a LangChain retriever."""
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
            tool_name="langchain_rag",
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

    try:
        retriever = forge_knowledge_base_to_langchain_retriever(
            local_knowledge_base,
            top_k=top_k,
            retriever_name=selected_retriever,
        )
        documents = retriever.invoke(query)
    except LangChainIntegrationError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Knowledge base: {workspace.safe_display(resolved_path)}")
    typer.echo(f"Retriever: {selected_retriever}")
    typer.echo(f"Query: {query}")
    typer.echo(f"Documents: {len(documents)}")

    if not documents:
        typer.echo("")
        typer.echo("No documents.")
        return

    typer.echo("")
    typer.echo("Documents:")
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        heading_path = " > ".join(metadata.get("heading_path", []))
        score = float(metadata.get("score", 0.0))
        typer.echo(
            f"- #{index} rank={metadata.get('rank')} "
            f"score={score:.4f} "
            f"source={metadata.get('source')} "
            f"chunk_id={metadata.get('chunk_id')} "
            f"retriever={metadata.get('retriever')}"
        )
        if heading_path:
            typer.echo(f"  heading={heading_path}")


def _validate_retriever_type(retriever_type: str) -> RetrieverType:
    normalized = retriever_type.strip().lower()
    if normalized in {"keyword", "vector", "hybrid"}:
        return cast(RetrieverType, normalized)

    raise typer.BadParameter(
        f"Unknown retriever: {retriever_type}. Supported retrievers: keyword, vector, hybrid.",
        param_hint="--retriever",
    )
