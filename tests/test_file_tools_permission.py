from __future__ import annotations

import json
from pathlib import Path

from forge_agent.security import (
    PATH_OUTSIDE_WORKSPACE,
    PATH_TRAVERSAL_BLOCKED,
    PERMISSION_DENIED,
    PermissionPolicy,
    Workspace,
)
from forge_agent.tools.file_tools import ListFilesTool, ReadFileTool, WriteFileTool


def test_read_file_tool_uses_workspace_guard(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    readme = workspace_root / "README.md"
    readme.write_text("# Demo\n", encoding="utf-8")

    tool = ReadFileTool(workspace=Workspace(workspace_root))

    result = tool.execute({"path": "README.md"})

    assert result.success is True
    assert result.payload["path"] == "README.md"
    assert result.payload["content"] == "# Demo\n"


def test_read_file_tool_blocks_path_traversal(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret\n", encoding="utf-8")

    tool = ReadFileTool(workspace=Workspace(workspace_root))

    result = tool.execute({"path": "../secret.txt"})

    payload = result.model_dump()
    assert result.success is False
    assert result.error_code == PATH_TRAVERSAL_BLOCKED
    assert str(outside_file) not in json.dumps(payload)


def test_read_file_tool_blocks_absolute_outside_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret\n", encoding="utf-8")

    tool = ReadFileTool(workspace=Workspace(workspace_root))

    result = tool.execute({"path": str(outside_file)})

    payload = result.model_dump()
    assert result.success is False
    assert result.error_code == PATH_OUTSIDE_WORKSPACE
    assert result.safe_detail == {"path": "<outside-workspace>"}
    assert str(outside_file) not in json.dumps(payload)


def test_list_files_tool_uses_workspace_guard(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (workspace_root / "docs").mkdir()

    tool = ListFilesTool(workspace=Workspace(workspace_root))

    result = tool.execute({"path": "."})

    assert result.success is True
    assert result.payload["path"] == "."
    assert result.payload["files"] == ["README.md", "docs"]


def test_write_file_requires_permission(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()

    tool = WriteFileTool(workspace=Workspace(workspace_root))

    result = tool.execute(
        {
            "path": "notes.md",
            "content": "hello\n",
        }
    )

    assert result.success is False
    assert result.error_code == PERMISSION_DENIED
    assert result.error_message == "Permission denied."
    assert result.payload["reason"] == "Write actions require explicit permission."
    assert not (workspace_root / "notes.md").exists()


def test_write_file_allowed_when_policy_enabled(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()

    tool = WriteFileTool(
        workspace=Workspace(workspace_root),
        permission_policy=PermissionPolicy(allow_write=True),
    )

    result = tool.execute(
        {
            "path": "notes.md",
            "content": "hello\n",
        }
    )

    assert result.success is True
    assert result.payload["path"] == "notes.md"
    assert result.payload["bytes_written"] == 6
    assert (workspace_root / "notes.md").read_text(encoding="utf-8") == "hello\n"


def test_write_file_still_uses_workspace_guard_when_allowed(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"

    tool = WriteFileTool(
        workspace=Workspace(workspace_root),
        permission_policy=PermissionPolicy(allow_write=True),
    )

    result = tool.execute(
        {
            "path": str(outside_file),
            "content": "secret\n",
        }
    )

    payload = result.model_dump()
    assert result.success is False
    assert result.error_code == PATH_OUTSIDE_WORKSPACE
    assert not outside_file.exists()
    assert str(outside_file) not in json.dumps(payload)


def test_file_tool_returns_structured_error_on_denial(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()

    tool = WriteFileTool(workspace=Workspace(workspace_root))

    result = tool.execute(
        {
            "path": "notes.md",
            "content": "hello\n",
        },
        tool_call_id="call_123",
    )

    assert result.model_dump() == {
        "tool_name": "write_file",
        "success": False,
        "tool_call_id": "call_123",
        "payload": {"reason": "Write actions require explicit permission."},
        "error_code": PERMISSION_DENIED,
        "error_message": "Permission denied.",
        "safe_detail": {"path": "notes.md"},
    }