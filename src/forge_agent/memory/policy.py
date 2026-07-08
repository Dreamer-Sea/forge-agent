from __future__ import annotations

import re

from forge_agent.memory.models import MemoryScope, MemoryType, MemoryWriteDecision

_EXPLICIT_MEMORY_PATTERNS = (
    re.compile(
        r"^\s*(?:记住|请记住|记一下|保存|remember|please remember)\s*[:：]\s*(?P<content>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:记住|请记住|记一下|保存|remember|please remember)\s+(?P<content>.+)$",
        re.IGNORECASE,
    ),
)

_SENSITIVE_PATTERN = re.compile(
    "|".join(
        [
            r"password",
            r"passwd",
            r"api[\s_-]?key",
            r"access[\s_-]?key",
            r"secret",
            r"token",
            r"private[\s_-]?key",
            r"密码",
            r"口令",
            r"密钥",
            r"身份证",
            r"银行卡",
            r"信用卡",
        ]
    ),
    re.IGNORECASE,
)


class MemoryWritePolicy:
    """Deterministic first-pass policy for long-term memory writes."""

    def __init__(self, *, min_content_chars: int = 8) -> None:
        self._min_content_chars = min_content_chars

    def decide_from_user_input(
        self,
        user_input: str,
        *,
        scope: MemoryScope = MemoryScope.PROJECT,
        scope_id: str | None = None,
    ) -> MemoryWriteDecision:
        """Decide whether user input should become semantic memory."""

        content = self._extract_explicit_memory(user_input)
        if content is None:
            return MemoryWriteDecision(
                should_write=False,
                reason="user input does not contain an explicit memory request",
            )

        skip_reason = self._skip_reason(content)
        if skip_reason is not None:
            return MemoryWriteDecision(
                should_write=False,
                reason=skip_reason,
            )

        return MemoryWriteDecision(
            should_write=True,
            reason="explicit remember request",
            content=content,
            memory_type=MemoryType.SEMANTIC,
            scope=scope,
            scope_id=scope_id,
            metadata={"trigger": "explicit_user_request"},
            importance=0.8,
            confidence=0.9,
            source="user_input",
        )

    def decide_from_agent_result(
        self,
        summary: str,
        *,
        succeeded: bool,
        scope: MemoryScope = MemoryScope.PROJECT,
        scope_id: str | None = None,
    ) -> MemoryWriteDecision:
        """Decide whether a completed agent result should become summary memory."""

        content = summary.strip()
        if not succeeded:
            return MemoryWriteDecision(
                should_write=False,
                reason="agent result was not successful",
            )

        skip_reason = self._skip_reason(content)
        if skip_reason is not None:
            return MemoryWriteDecision(
                should_write=False,
                reason=skip_reason,
            )

        return MemoryWriteDecision(
            should_write=True,
            reason="successful reusable task summary",
            content=content,
            memory_type=MemoryType.SUMMARY,
            scope=scope,
            scope_id=scope_id,
            metadata={"trigger": "agent_result"},
            importance=0.6,
            confidence=0.75,
            source="agent_result",
        )

    def decide_failure_memory(
        self,
        failure_reason: str,
        *,
        failure_type: str = "tool_failed",
        scope: MemoryScope = MemoryScope.PROJECT,
        scope_id: str | None = None,
    ) -> MemoryWriteDecision:
        """Decide whether a failure should become episodic memory."""

        reason = failure_reason.strip()
        if not reason:
            return MemoryWriteDecision(
                should_write=False,
                reason="failure reason is blank",
            )

        content = f"{failure_type}: {reason}"
        skip_reason = self._skip_reason(content)
        if skip_reason is not None:
            return MemoryWriteDecision(
                should_write=False,
                reason=skip_reason,
            )

        importance = 0.9 if failure_type == "verification_failed" else 0.7

        return MemoryWriteDecision(
            should_write=True,
            reason=f"{failure_type} can be reused as failure memory",
            content=content,
            memory_type=MemoryType.EPISODIC,
            scope=scope,
            scope_id=scope_id,
            metadata={
                "trigger": "failure",
                "failure_type": failure_type,
            },
            importance=importance,
            confidence=0.8,
            source=failure_type,
        )

    def _extract_explicit_memory(self, user_input: str) -> str | None:
        for pattern in _EXPLICIT_MEMORY_PATTERNS:
            match = pattern.match(user_input)
            if match is not None:
                content = match.group("content").strip()
                return content or None

        return None

    def _skip_reason(self, content: str) -> str | None:
        if len(content.strip()) < self._min_content_chars:
            return "content is too short for long-term memory"
        if _SENSITIVE_PATTERN.search(content) is not None:
            return "content appears to contain sensitive information"
        return None
