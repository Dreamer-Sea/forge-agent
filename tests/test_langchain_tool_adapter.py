from __future__ import annotations

import pytest

from forge_agent.integrations.langchain.tools import (
    forge_registry_to_langchain_tools,
    forge_tool_to_langchain_tool,
)
from forge_agent.tools.calculator import CalculatorTool
from forge_agent.tools.registry import ToolRegistry

pytest.importorskip("langchain_core")


def test_forge_tool_converts_to_langchain_tool() -> None:
    tool = forge_tool_to_langchain_tool(CalculatorTool())

    assert tool.name == "calculator"
    assert tool.description == "Evaluate a safe arithmetic expression."


def test_langchain_tool_preserves_input_schema() -> None:
    tool = forge_tool_to_langchain_tool(CalculatorTool())

    schema = tool.args_schema.model_json_schema()

    assert "expression" in schema["properties"]
    assert "expression" in schema["required"]


def test_langchain_tool_executes_original_forge_tool() -> None:
    tool = forge_tool_to_langchain_tool(CalculatorTool())

    result = tool.invoke({"expression": "2 + 3 * 4"})

    assert result["tool_name"] == "calculator"
    assert result["success"] is True
    assert result["payload"]["result"] == 14.0


def test_langchain_tool_does_not_swallow_tool_error() -> None:
    tool = forge_tool_to_langchain_tool(CalculatorTool())

    result = tool.invoke({"expression": "2 +"})

    assert result["tool_name"] == "calculator"
    assert result["success"] is False
    assert result["error_code"] == "invalid_expression"
    assert result["error_message"]


def test_registry_converts_all_tools() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    tools = forge_registry_to_langchain_tools(registry)

    assert [tool.name for tool in tools] == ["calculator"]
