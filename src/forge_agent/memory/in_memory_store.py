from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from forge_agent.memory.models import MemoryRecord, MemoryScope, MemorySearchResult, MemoryType

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]")


def _normalize_content(content: str) -> str:
    return " ".join(content.casefold().strip().split())


def _tokenize(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_PATTERN.findall(text)}


def _keyword_overlap_score(query: str, content: str) -> tuple[float, set[str]]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0, set()

    content_tokens = _tokenize(content)
    matched_tokens = query_tokens & content_tokens
    return len(matched_tokens) / len(query_tokens), matched_tokens


class InMemoryStore:
    """Deterministic in-memory implementation of MemoryStore."""

    def __init__(self, records: list[MemoryRecord] | None = None) -> None:
        self._records: dict[str, MemoryRecord] = {}

        for record in records or []:
            self.add(record)

    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Store a memory record.

        Duplicate records are detected by type, scope, scope_id, and normalized
        content. When a duplicate exists, the existing record is returned instead
        of creating another one.
        """

        duplicate = self._find_duplicate(record)
        if duplicate is not None:
            return duplicate

        self._records[record.id] = record
        return record

    def search(
        self,
        query: str,
        scope: MemoryScope | None = None,
        top_k: int = 5,
        *,
        scope_id: str | None = None,
        type_filter: MemoryType | None = None,
    ) -> list[MemorySearchResult]:
        """Search records with deterministic keyword overlap scoring."""

        if top_k <= 0:
            return []

        scored: list[tuple[MemoryRecord, float, set[str]]] = []

        for record in self._records.values():
            if not self._matches_filters(
                record,
                scope=scope,
                scope_id=scope_id,
                type_filter=type_filter,
            ):
                continue

            score, matched_tokens = _keyword_overlap_score(query, record.content)
            if score <= 0:
                continue

            scored.append((record, score, matched_tokens))

        scored.sort(
            key=lambda item: (
                -item[1],
                -item[0].importance,
                -item[0].confidence,
                item[0].created_at.isoformat(),
                item[0].id,
            )
        )

        results: list[MemorySearchResult] = []
        for rank, (record, score, matched_tokens) in enumerate(scored[:top_k], start=1):
            matched = ", ".join(sorted(matched_tokens))
            results.append(
                MemorySearchResult(
                    record=record,
                    score=score,
                    rank=rank,
                    reason=f"keyword overlap matched: {matched}",
                )
            )

        return results

    def list(
        self,
        scope: MemoryScope | None = None,
        *,
        scope_id: str | None = None,
        type_filter: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        """List records with optional scope, scope_id, and type filtering."""

        records = [
            record
            for record in self._records.values()
            if self._matches_filters(
                record,
                scope=scope,
                scope_id=scope_id,
                type_filter=type_filter,
            )
        ]

        return sorted(records, key=lambda record: (record.created_at.isoformat(), record.id))

    def update(self, record_id: str, patch: dict[str, Any]) -> MemoryRecord:
        """Patch one memory record and return the updated version."""

        if record_id not in self._records:
            raise KeyError(f"memory record not found: {record_id}")

        current = self._records[record_id]
        payload = current.model_dump()
        payload.update(patch)
        payload["id"] = record_id

        if "updated_at" not in patch:
            payload["updated_at"] = datetime.now(UTC)

        updated = MemoryRecord.model_validate(payload)

        duplicate = self._find_duplicate(updated, exclude_id=record_id)
        if duplicate is not None:
            raise ValueError(f"memory record duplicates existing record: {duplicate.id}")

        self._records[record_id] = updated
        return updated

    def delete(self, record_id: str) -> None:
        """Delete one memory record by id."""

        if record_id not in self._records:
            raise KeyError(f"memory record not found: {record_id}")

        del self._records[record_id]

    def _find_duplicate(
        self,
        record: MemoryRecord,
        *,
        exclude_id: str | None = None,
    ) -> MemoryRecord | None:
        target_content = _normalize_content(record.content)

        for existing in self._records.values():
            if exclude_id is not None and existing.id == exclude_id:
                continue

            if (
                existing.type == record.type
                and existing.scope == record.scope
                and existing.scope_id == record.scope_id
                and _normalize_content(existing.content) == target_content
            ):
                return existing

        return None

    def _matches_filters(
        self,
        record: MemoryRecord,
        *,
        scope: MemoryScope | None,
        scope_id: str | None,
        type_filter: MemoryType | None,
    ) -> bool:
        if scope is not None and record.scope != scope:
            return False
        if scope_id is not None and record.scope_id != scope_id:
            return False
        if type_filter is not None and record.type != type_filter:
            return False
        return True
