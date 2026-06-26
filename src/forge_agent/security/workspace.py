"""Workspace guard for safe path access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forge_agent.security.errors import (
    PathOutsideWorkspaceError,
    PathTraversalBlockedError,
)


@dataclass(frozen=True)
class Workspace:
    """Restrict file access to a configured workspace root.

    This is not a full sandbox. It is a deterministic guardrail that prevents
    accidental or malicious path escape through traversal segments, absolute
    paths, and symlinks.
    """

    root: Path

    def __post_init__(self) -> None:
        resolved_root = self.root.expanduser().resolve(strict=False)
        object.__setattr__(self, "root", resolved_root)

    def resolve_user_path(
        self,
        user_path: str | Path,
        *,
        tool_name: str = "workspace",
    ) -> Path:
        """Resolve a user-provided path and ensure it stays inside workspace.

        Args:
            user_path: Path supplied by a user, model, or tool input.
            tool_name: Tool name used for structured error reporting.

        Returns:
            Absolute resolved path inside the workspace.

        Raises:
            PathTraversalBlockedError: If path contains ``..``.
            PathOutsideWorkspaceError: If resolved path escapes workspace.
        """
        path = Path(user_path)

        if self._contains_traversal(path):
            raise PathTraversalBlockedError(
                tool_name=tool_name,
                safe_detail={"path": self.safe_display(path)},
            )

        if path.is_absolute():
            candidate = path.resolve(strict=False)
        else:
            candidate = (self.root / path).resolve(strict=False)

        self.ensure_inside(candidate, tool_name=tool_name)
        return candidate

    def ensure_inside(
        self,
        path: Path,
        *,
        tool_name: str = "workspace",
    ) -> None:
        """Ensure a resolved path is inside the workspace root."""
        resolved_path = path.resolve(strict=False)

        try:
            resolved_path.relative_to(self.root)
        except ValueError as exc:
            raise PathOutsideWorkspaceError(
                tool_name=tool_name,
                safe_detail={"path": "<outside-workspace>"},
            ) from exc

    def safe_display(self, path: str | Path) -> str:
        """Return a path string safe for CLI output and trace events.

        Inside paths are shown relative to the workspace. Outside paths are
        redacted to avoid leaking sensitive absolute paths.
        """
        raw_path = Path(path)

        if not raw_path.is_absolute():
            return raw_path.as_posix()

        resolved_path = raw_path.resolve(strict=False)
        try:
            return resolved_path.relative_to(self.root).as_posix()
        except ValueError:
            return "<outside-workspace>"

    @staticmethod
    def _contains_traversal(path: Path) -> bool:
        """Return whether path contains traversal segments."""
        return ".." in path.parts
