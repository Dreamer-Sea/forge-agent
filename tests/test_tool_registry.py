from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.security import Workspace
from forge_agent.tools.calculator import CalculatorTool
from forge_agent.tools.file_tools import ListFilesTool, ReadFileTool
from forge_agent.tools.registry import ToolRegistry


def test_tool_registry_registers_tool() -> None:
    registry = ToolRegistry()
    tool = ListFilesTool()

    registry.register(tool)

    assert registry.get("list_files") is tool


def test_tool_registry_rejects_duplicate_tool_name() -> None:
    registry = ToolRegistry()
    tool = ListFilesTool()

    registry.register(tool)

    with pytest.raises(ValueError, match="Tool already registered"):
        registry.register(tool)


def test_tool_registry_returns_unknown_tool_error() -> None:
    registry = ToolRegistry()

    result = registry.execute(
        name="missing_tool",
        arguments={},
        tool_call_id="call_missing",
    )

    assert result.success is False
    assert result.tool_name == "missing_tool"
    assert result.tool_call_id == "call_missing"
    assert result.error_code == "unknown_tool"


def test_tool_validates_arguments() -> None:
    tool = ReadFileTool()

    result = tool.execute(arguments={})

    assert result.success is False
    assert result.error_code == "validation_error"


def test_list_files_tool_lists_directory(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# forge-agent", encoding="utf-8")

    tool = ListFilesTool(workspace=Workspace(tmp_path))
    result = tool.execute(arguments={"path": "."})

    assert result.success is True
    assert result.payload["path"] == "."
    assert result.payload["files"] == ["README.md"]


def test_read_file_tool_reads_file(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# forge-agent", encoding="utf-8")

    tool = ReadFileTool(workspace=Workspace(tmp_path))
    result = tool.execute(arguments={"path": "README.md"})

    assert result.success is True
    assert result.payload["path"] == "README.md"
    assert result.payload["content"] == "# forge-agent"


def test_calculator_tool_evaluates_expression() -> None:
    tool = CalculatorTool()

    result = tool.execute(arguments={"expression": "1 + 2 * 3"})

    assert result.success is True
    assert result.payload["result"] == 7.0