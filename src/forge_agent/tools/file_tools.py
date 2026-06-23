from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from forge_agent.tools.base import ToolResult, ToolSchema


class ListFilesArgs(BaseModel):
    path: str = "."


class ReadFileArgs(BaseModel):
    path: str


class ListFilesTool:
    name = "list_files"
    description = "List files under a directory."

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=ListFilesArgs.model_json_schema(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        try:
            args = ListFilesArgs.model_validate(arguments)
            directory = Path(args.path)

            if not directory.exists():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="path_not_found",
                    error_message=f"Path does not exist: {args.path}",
                )

            if not directory.is_dir():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="not_a_directory",
                    error_message=f"Path is not a directory: {args.path}",
                )

            files = sorted(path.name for path in directory.iterdir())

            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=True,
                payload={"path": str(directory), "files": files},
            )
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )
        except OSError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="io_error",
                error_message=str(error),
            )


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file."

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=ReadFileArgs.model_json_schema(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        try:
            args = ReadFileArgs.model_validate(arguments)
            file_path = Path(args.path)

            if not file_path.exists():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="path_not_found",
                    error_message=f"Path does not exist: {args.path}",
                )

            if not file_path.is_file():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="not_a_file",
                    error_message=f"Path is not a file: {args.path}",
                )

            content = file_path.read_text(encoding="utf-8")

            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=True,
                payload={"path": str(file_path), "content": content},
            )
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )
        except OSError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="io_error",
                error_message=str(error),
            )