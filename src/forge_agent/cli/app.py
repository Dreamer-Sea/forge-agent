from __future__ import annotations

import typer

app = typer.Typer(help="A minimal Agent Platform demo CLI.")


@app.callback()
def callback() -> None:
    """Forge Agent CLI."""
    pass


@app.command()
def run(task: str) -> None:
    """Run one agent task."""
    typer.echo(f"Task: {task}")
    typer.echo("Status: Day 1 skeleton ready")


def main() -> None:
    app()