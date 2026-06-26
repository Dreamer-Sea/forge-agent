from __future__ import annotations

from typing import Any

from forge_agent.tools.base import Tool, ToolResult, ToolSchema


class ToolRegistry:
    """Registry for discovering and executing tools by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"Unknown tool: {name}") from error

    def schemas(self) -> list[ToolSchema]:
        return [tool.schema() for tool in self._tools.values()]

    def provider_schemas(self) -> list[dict[str, Any]]:
        return [schema.model_dump() for schema in self.schemas()]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        tool = self._tools.get(name)

        if tool is None:
            return ToolResult(
                tool_name=name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="unknown_tool",
                error_message=f"Unknown tool: {name}",
            )

        return tool.execute(arguments=arguments, tool_call_id=tool_call_id)
