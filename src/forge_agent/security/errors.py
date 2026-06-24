"""Structured errors for tool execution and security failures."""

from __future__ import annotations

from typing import Any

PERMISSION_DENIED = "PERMISSION_DENIED"
PATH_OUTSIDE_WORKSPACE = "PATH_OUTSIDE_WORKSPACE"
PATH_TRAVERSAL_BLOCKED = "PATH_TRAVERSAL_BLOCKED"


class ToolError(Exception):
    """Base structured error returned by tools.

    Tool errors are expected runtime outcomes, not programmer bugs. They should
    be serializable so the agent loop, CLI, and trace layer can present them
    consistently.
    """

    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        reason: str,
        tool_name: str,
        safe_detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.reason = reason
        self.tool_name = tool_name
        self.safe_detail = safe_detail or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert the error to a JSON-serializable payload."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "safe_detail": self.safe_detail,
        }


class PermissionDeniedError(ToolError):
    """Raised when a tool action is rejected by PermissionPolicy."""

    def __init__(
        self,
        *,
        tool_name: str,
        reason: str,
        safe_detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            error_code=PERMISSION_DENIED,
            message="Permission denied.",
            reason=reason,
            tool_name=tool_name,
            safe_detail=safe_detail,
        )