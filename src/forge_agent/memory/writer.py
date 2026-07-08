from __future__ import annotations

from forge_agent.memory.models import MemoryRecord, MemoryScope, MemoryWriteDecision
from forge_agent.memory.policy import MemoryWritePolicy
from forge_agent.memory.store import MemoryStore


def _normalize_content(content: str) -> str:
    return " ".join(content.casefold().strip().split())


class MemoryWriter:
    """Applies write policy and persists accepted long-term memories."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        policy: MemoryWritePolicy | None = None,
    ) -> None:
        self._store = store
        self._policy = policy or MemoryWritePolicy()

    def maybe_write_from_user_input(
        self,
        user_input: str,
        *,
        scope: MemoryScope = MemoryScope.PROJECT,
        scope_id: str | None = None,
    ) -> MemoryWriteDecision:
        """Persist explicit user memory requests when policy allows it."""

        decision = self._policy.decide_from_user_input(
            user_input,
            scope=scope,
            scope_id=scope_id,
        )
        return self._persist_decision(decision)

    def maybe_write_from_agent_result(
        self,
        summary: str,
        *,
        succeeded: bool,
        scope: MemoryScope = MemoryScope.PROJECT,
        scope_id: str | None = None,
    ) -> MemoryWriteDecision:
        """Persist successful reusable task summaries when policy allows it."""

        decision = self._policy.decide_from_agent_result(
            summary,
            succeeded=succeeded,
            scope=scope,
            scope_id=scope_id,
        )
        return self._persist_decision(decision)

    def maybe_write_failure_memory(
        self,
        failure_reason: str,
        *,
        failure_type: str = "tool_failed",
        scope: MemoryScope = MemoryScope.PROJECT,
        scope_id: str | None = None,
    ) -> MemoryWriteDecision:
        """Persist reusable failure memories when policy allows it."""

        decision = self._policy.decide_failure_memory(
            failure_reason,
            failure_type=failure_type,
            scope=scope,
            scope_id=scope_id,
        )
        return self._persist_decision(decision)

    def _persist_decision(self, decision: MemoryWriteDecision) -> MemoryWriteDecision:
        if not decision.should_write:
            return decision

        record = decision.to_record()
        duplicate = self._find_duplicate(record)
        if duplicate is not None:
            return MemoryWriteDecision(
                should_write=False,
                reason=f"duplicate memory skipped: {duplicate.id}",
                metadata={
                    **decision.metadata,
                    "duplicate_record_id": duplicate.id,
                },
            )

        stored = self._store.add(record)
        return decision.model_copy(
            update={
                "metadata": {
                    **decision.metadata,
                    "record_id": stored.id,
                }
            }
        )

    def _find_duplicate(self, record: MemoryRecord) -> MemoryRecord | None:
        candidates = self._store.list(
            scope=record.scope,
            scope_id=record.scope_id,
            type_filter=record.type,
        )
        target_content = _normalize_content(record.content)

        for candidate in candidates:
            if _normalize_content(candidate.content) == target_content:
                return candidate

        return None
