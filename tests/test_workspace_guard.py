from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_agent.security import (
    PATH_OUTSIDE_WORKSPACE,
    PATH_TRAVERSAL_BLOCKED,
    PathOutsideWorkspaceError,
    PathTraversalBlockedError,
    Workspace,
)


def test_read_file_inside_workspace_allowed(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    readme = workspace_root / "README.md"
    readme.write_text("# Demo\n", encoding="utf-8")

    workspace = Workspace(workspace_root)

    resolved = workspace.resolve_user_path("README.md", tool_name="read_file")

    assert resolved == readme.resolve()


def test_read_file_outside_workspace_denied(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret\n", encoding="utf-8")

    workspace = Workspace(workspace_root)

    with pytest.raises(PathTraversalBlockedError) as exc_info:
        workspace.resolve_user_path("../secret.txt", tool_name="read_file")

    payload = exc_info.value.to_dict()
    assert payload["error_code"] == PATH_TRAVERSAL_BLOCKED
    assert payload["tool_name"] == "read_file"
    assert str(outside_file) not in json.dumps(payload)


def test_absolute_path_outside_workspace_denied(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret\n", encoding="utf-8")

    workspace = Workspace(workspace_root)

    with pytest.raises(PathOutsideWorkspaceError) as exc_info:
        workspace.resolve_user_path(outside_file, tool_name="read_file")

    payload = exc_info.value.to_dict()
    assert payload["error_code"] == PATH_OUTSIDE_WORKSPACE
    assert payload["safe_detail"] == {"path": "<outside-workspace>"}
    assert str(outside_file) not in json.dumps(payload)


def test_path_traversal_is_blocked(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()

    workspace = Workspace(workspace_root)

    with pytest.raises(PathTraversalBlockedError) as exc_info:
        workspace.resolve_user_path("docs/../../secret.txt", tool_name="read_file")

    payload = exc_info.value.to_dict()
    assert payload["error_code"] == PATH_TRAVERSAL_BLOCKED
    assert payload["safe_detail"] == {"path": "docs/../../secret.txt"}


def test_symlink_escape_is_blocked(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret\n", encoding="utf-8")

    symlink_path = workspace_root / "secret-link.txt"

    try:
        symlink_path.symlink_to(outside_file)
    except OSError:
        pytest.skip("Symlink is not supported in this environment.")

    workspace = Workspace(workspace_root)

    with pytest.raises(PathOutsideWorkspaceError) as exc_info:
        workspace.resolve_user_path("secret-link.txt", tool_name="read_file")

    payload = exc_info.value.to_dict()
    assert payload["error_code"] == PATH_OUTSIDE_WORKSPACE
    assert payload["safe_detail"] == {"path": "<outside-workspace>"}
    assert str(outside_file) not in json.dumps(payload)


def test_absolute_path_inside_workspace_allowed(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    readme = workspace_root / "README.md"
    readme.write_text("# Demo\n", encoding="utf-8")

    workspace = Workspace(workspace_root)

    resolved = workspace.resolve_user_path(readme, tool_name="read_file")

    assert resolved == readme.resolve()


def test_safe_display_redacts_outside_absolute_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"

    workspace = Workspace(workspace_root)

    assert workspace.safe_display(outside_file) == "<outside-workspace>"
