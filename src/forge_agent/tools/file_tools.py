from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from forge_agent.security import (
    PermissionAction,
    PermissionDeniedError,
    PermissionPolicy,
    ToolError,
    Workspace,
)
from forge_agent.tools.base import ToolResult, ToolSchema


class ListFilesArgs(BaseModel):
    path: str = "."


class ReadFileArgs(BaseModel):
    path: str


class WriteFileArgs(BaseModel):
    path: str
    content: str


class GuardedFileToolMixin:
    """Shared permission and workspace behavior for file tools."""

    workspace: Workspace
    permission_policy: PermissionPolicy

    def _check_permission(
        self,
        action: PermissionAction,
        *,
        tool_call_id: str | None,
        safe_detail: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        decision = self.permission_policy.check(action)

        if decision.allowed:
            return None

        error = PermissionDeniedError(
            tool_name=self.name,
            reason=decision.reason,
            safe_detail=safe_detail or {"action": action.value},
        )
        return self._error_to_result(error, tool_call_id=tool_call_id)

    def _error_to_result(
        self,
        error: ToolError,
        *,
        tool_call_id: str | None,
    ) -> ToolResult:
        return ToolResult(
            tool_name=error.tool_name,
            tool_call_id=tool_call_id,
            success=False,
            error_code=error.error_code,
            error_message=error.message,
            payload={"reason": error.reason},
            safe_detail=error.safe_detail,
        )

    @property
    def name(self) -> str:
        raise NotImplementedError


class ListFilesTool(GuardedFileToolMixin):
    name = "list_files"
    description = "List files under a directory."

    def __init__(
        self,
        workspace: Workspace | None = None,
        permission_policy: PermissionPolicy | None = None,
    ) -> None:
        self.workspace = workspace or Workspace(Path.cwd())
        self.permission_policy = permission_policy or PermissionPolicy()

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
            permission_error = self._check_permission(
                PermissionAction.READ_FILE,
                tool_call_id=tool_call_id,
                safe_detail={"path": args.path},
            )
            if permission_error is not None:
                return permission_error

            directory = self.workspace.resolve_user_path(
                args.path,
                tool_name=self.name,
            )
            safe_path = self.workspace.safe_display(directory)

            if not directory.exists():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="path_not_found",
                    error_message=f"Path does not exist: {safe_path}",
                    safe_detail={"path": safe_path},
                )

            if not directory.is_dir():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="not_a_directory",
                    error_message=f"Path is not a directory: {safe_path}",
                    safe_detail={"path": safe_path},
                )

            files = sorted(path.name for path in directory.iterdir())

            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=True,
                payload={"path": safe_path, "files": files},
            )
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )
        except ToolError as error:
            return self._error_to_result(error, tool_call_id=tool_call_id)
        except OSError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="io_error",
                error_message=str(error),
            )


class ReadFileTool(GuardedFileToolMixin):
    name = "read_file"
    description = "Read a UTF-8 text file."

    def __init__(
        self,
        workspace: Workspace | None = None,
        permission_policy: PermissionPolicy | None = None,
    ) -> None:
        self.workspace = workspace or Workspace(Path.cwd())
        self.permission_policy = permission_policy or PermissionPolicy()

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
            permission_error = self._check_permission(
                PermissionAction.READ_FILE,
                tool_call_id=tool_call_id,
                safe_detail={"path": args.path},
            )
            if permission_error is not None:
                return permission_error

            file_path = self.workspace.resolve_user_path(
                args.path,
                tool_name=self.name,
            )
            safe_path = self.workspace.safe_display(file_path)

            if not file_path.exists():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="path_not_found",
                    error_message=f"Path does not exist: {safe_path}",
                    safe_detail={"path": safe_path},
                )

            if not file_path.is_file():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="not_a_file",
                    error_message=f"Path is not a file: {safe_path}",
                    safe_detail={"path": safe_path},
                )

            content = file_path.read_text(encoding="utf-8")

            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=True,
                payload={"path": safe_path, "content": content},
            )
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )
        except ToolError as error:
            return self._error_to_result(error, tool_call_id=tool_call_id)
        except OSError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="io_error",
                error_message=str(error),
            )


class WriteFileTool(GuardedFileToolMixin):
    name = "write_file"
    description = "Write UTF-8 text content to a workspace file."

    def __init__(
        self,
        workspace: Workspace | None = None,
        permission_policy: PermissionPolicy | None = None,
    ) -> None:
        self.workspace = workspace or Workspace(Path.cwd())
        self.permission_policy = permission_policy or PermissionPolicy()

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=WriteFileArgs.model_json_schema(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        try:
            args = WriteFileArgs.model_validate(arguments)
            permission_error = self._check_permission(
                PermissionAction.WRITE_FILE,
                tool_call_id=tool_call_id,
                safe_detail={"path": args.path},
            )
            if permission_error is not None:
                return permission_error

            file_path = self.workspace.resolve_user_path(
                args.path,
                tool_name=self.name,
            )
            safe_path = self.workspace.safe_display(file_path)

            if not file_path.parent.exists():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="path_not_found",
                    error_message=f"Parent directory does not exist: {safe_path}",
                    safe_detail={"path": safe_path},
                )

            if file_path.exists() and not file_path.is_file():
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=tool_call_id,
                    success=False,
                    error_code="not_a_file",
                    error_message=f"Path is not a file: {safe_path}",
                    safe_detail={"path": safe_path},
                )

            file_path.write_text(args.content, encoding="utf-8")

            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=True,
                payload={
                    "path": safe_path,
                    "bytes_written": len(args.content.encode("utf-8")),
                },
            )
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )
        except ToolError as error:
            return self._error_to_result(error, tool_call_id=tool_call_id)
        except OSError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="io_error",
                error_message=str(error),
            )
