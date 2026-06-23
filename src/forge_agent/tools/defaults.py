from __future__ import annotations

from forge_agent.tools.calculator import CalculatorTool
from forge_agent.tools.echo import EchoTextTool
from forge_agent.tools.file_tools import ListFilesTool, ReadFileTool
from forge_agent.tools.registry import ToolRegistry


def create_default_tool_registry() -> ToolRegistry:
    """Create the default Day 1 tool registry."""

    registry = ToolRegistry()
    registry.register(ListFilesTool())
    registry.register(ReadFileTool())
    registry.register(CalculatorTool())
    registry.register(EchoTextTool())
    return registry