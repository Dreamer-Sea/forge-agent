from __future__ import annotations

import json

from forge_agent.security import (
    PERMISSION_DENIED,
    PermissionAction,
    PermissionDeniedError,
    PermissionPolicy,
)


def test_permission_decision_allows_read_by_default() -> None:
    policy = PermissionPolicy()

    decision = policy.check(PermissionAction.READ_FILE)

    assert decision.allowed is True
    assert decision.action is PermissionAction.READ_FILE
    assert decision.policy_name == "default"
    assert "allowed" in decision.reason.lower()


def test_permission_decision_denies_write_by_default() -> None:
    policy = PermissionPolicy()

    decision = policy.check(PermissionAction.WRITE_FILE)

    assert decision.allowed is False
    assert decision.action is PermissionAction.WRITE_FILE
    assert decision.policy_name == "default"
    assert "explicit permission" in decision.reason.lower()


def test_permission_decision_allows_write_when_enabled() -> None:
    policy = PermissionPolicy(allow_write=True)

    decision = policy.check(PermissionAction.WRITE_FILE)

    assert decision.allowed is True
    assert decision.action is PermissionAction.WRITE_FILE
    assert "explicitly enabled" in decision.reason.lower()


def test_permission_decision_denies_shell_by_default() -> None:
    policy = PermissionPolicy()

    decision = policy.check(PermissionAction.SHELL)

    assert decision.allowed is False
    assert decision.action is PermissionAction.SHELL
    assert "disabled" in decision.reason.lower()


def test_tool_error_is_serializable() -> None:
    error = PermissionDeniedError(
        tool_name="write_file",
        reason="Write actions require explicit permission.",
        safe_detail={"path": "README.md"},
    )

    payload = error.to_dict()

    assert payload == {
        "error_code": PERMISSION_DENIED,
        "message": "Permission denied.",
        "reason": "Write actions require explicit permission.",
        "tool_name": "write_file",
        "safe_detail": {"path": "README.md"},
    }

    json.dumps(payload)