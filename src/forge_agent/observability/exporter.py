"""Trace exporters."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from forge_agent.observability.events import TraceEvent


class JsonlTraceExporter:
    """Export trace events to JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def export(self, events: Iterable[TraceEvent]) -> Path:
        """Write trace events to a JSONL file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w", encoding="utf-8") as file:
            for event in events:
                file.write(event.model_dump_json())
                file.write("\n")

        return self.path

    def append(self, events: Iterable[TraceEvent]) -> Path:
        """Append trace events to a JSONL file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as file:
            for event in events:
                file.write(event.model_dump_json())
                file.write("\n")

        return self.path