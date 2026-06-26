from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from forge_agent.tools.base import ToolResult, ToolSchema


class EchoTextArgs(BaseModel):
    text: str


class EchoTextTool:
    name = "echo_text"
    description = "Echo the provided text."

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=EchoTextArgs.model_json_schema(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        try:
            args = EchoTextArgs.model_validate(arguments)

            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=True,
                payload={"text": args.text},
            )
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )
