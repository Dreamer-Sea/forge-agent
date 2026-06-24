"""Permission policy for agent tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PermissionAction(StrEnum):
    """Supported permission actions for tools."""

    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    SEARCH_KB = "search_kb"
    SHELL = "shell"


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission policy check."""

    allowed: bool
    action: PermissionAction
    reason: str
    policy_name: str = "default"

    @classmethod
    def allow(
        cls,
        *,
        action: PermissionAction,
        reason: str,
        policy_name: str = "default",
    ) -> PermissionDecision:
        return cls(
            allowed=True,
            action=action,
            reason=reason,
            policy_name=policy_name,
        )

    @classmethod
    def deny(
        cls,
        *,
        action: PermissionAction,
        reason: str,
        policy_name: str = "default",
    ) -> PermissionDecision:
        return cls(
            allowed=False,
            action=action,
            reason=reason,
            policy_name=policy_name,
        )

    def to_dict(self) -> dict[str, str | bool]:
        """Convert the decision to a JSON-serializable payload."""
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "reason": self.reason,
            "policy_name": self.policy_name,
        }


@dataclass(frozen=True)
class PermissionPolicy:
    """Default platform-level permission policy.

    The default policy is intentionally conservative:
    - read_file is allowed by default;
    - search_kb is allowed, but later must still pass Workspace Guard;
    - write_file is denied unless explicitly enabled;
    - shell is denied unless explicitly enabled.
    """

    allow_write: bool = False
    allow_shell: bool = False
    policy_name: str = "default"

    def check(self, action: PermissionAction) -> PermissionDecision:
        """Return a decision for the requested action."""
        if action is PermissionAction.READ_FILE:
            return PermissionDecision.allow(
                action=action,
                reason="Read actions are allowed by default.",
                policy_name=self.policy_name,
            )

        if action is PermissionAction.SEARCH_KB:
            return PermissionDecision.allow(
                action=action,
                reason="Knowledge base search is allowed within guarded workspace.",
                policy_name=self.policy_name,
            )

        if action is PermissionAction.WRITE_FILE:
            if self.allow_write:
                return PermissionDecision.allow(
                    action=action,
                    reason="Write actions are explicitly enabled.",
                    policy_name=self.policy_name,
                )

            return PermissionDecision.deny(
                action=action,
                reason="Write actions require explicit permission.",
                policy_name=self.policy_name,
            )

        if action is PermissionAction.SHELL:
            if self.allow_shell:
                return PermissionDecision.allow(
                    action=action,
                    reason="Shell actions are explicitly enabled.",
                    policy_name=self.policy_name,
                )

            return PermissionDecision.deny(
                action=action,
                reason="Shell execution is disabled by default.",
                policy_name=self.policy_name,
            )