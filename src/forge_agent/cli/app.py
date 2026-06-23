from __future__ import annotations

import typer

from forge_agent.providers.fake import FakeProvider
from forge_agent.runtime.native_runtime import NativeAgentRuntime
from forge_agent.tools.defaults import create_default_tool_registry

app = typer.Typer(help="A minimal Agent Platform demo CLI.")


@app.callback()
def callback() -> None:
    """Forge Agent CLI."""


@app.command()
def run(task: str) -> None:
    """Run one agent task."""

    registry = create_default_tool_registry()
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


def main() -> None:
    app()