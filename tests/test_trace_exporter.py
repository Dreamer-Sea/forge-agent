from __future__ import annotations

import json
from pathlib import Path

from forge_agent.observability import JsonlTraceExporter, TraceEvent, TraceRecorder


def test_trace_records_model_and_tool_events() -> None:
    recorder = TraceRecorder(run_id="run-001", case_id="case-001")

    recorder.record_model_call(
        name="fake-provider",
        step_index=0,
        payload={"input": "hello"},
    )
    recorder.record_tool_call(
        name="read_file",
        step_index=1,
        payload={"path": "README.md"},
    )

    events = recorder.events

    assert len(events) == 2
    assert events[0].run_id == "run-001"
    assert events[0].case_id == "case-001"
    assert events[0].event_type == "model_call"
    assert events[0].name == "fake-provider"
    assert events[1].event_type == "tool_call"
    assert events[1].name == "read_file"


def test_trace_exporter_writes_jsonl(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    events = [
        TraceEvent(
            run_id="run-001",
            case_id="case-001",
            event_type="model_call",
            name="fake-provider",
            payload={"input": "hello"},
        ),
        TraceEvent(
            run_id="run-001",
            case_id="case-001",
            event_type="final_answer",
            payload={"answer": "done", "stopped_reason": "completed"},
        ),
    ]

    exported_path = JsonlTraceExporter(trace_path).export(events)

    assert exported_path == trace_path

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "model_call"
    assert json.loads(lines[1])["event_type"] == "final_answer"


def test_trace_event_is_json_serializable() -> None:
    event = TraceEvent(
        run_id="run-001",
        case_id="case-001",
        event_type="tool_result",
        name="read_file",
        payload={
            "ok": True,
            "items": [1, "two", None],
            "nested": {"path": "README.md"},
        },
    )

    decoded = json.loads(event.model_dump_json())

    assert decoded["run_id"] == "run-001"
    assert decoded["case_id"] == "case-001"
    assert decoded["event_type"] == "tool_result"
    assert decoded["payload"]["nested"]["path"] == "README.md"


def test_trace_exporter_includes_run_id(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    recorder = TraceRecorder(run_id="run-123", case_id="trace_001")

    recorder.record_model_call(name="fake-provider")
    recorder.record_tool_call(name="read_file", payload={"path": "README.md"})
    recorder.record_final_answer(answer="Runtime and Tool summary", stopped_reason="completed")

    JsonlTraceExporter(trace_path).export(recorder.events)

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert {row["run_id"] for row in rows} == {"run-123"}
    assert {row["case_id"] for row in rows} == {"trace_001"}
    assert [row["event_type"] for row in rows] == [
        "model_call",
        "tool_call",
        "final_answer",
    ]
