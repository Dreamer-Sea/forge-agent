from __future__ import annotations

from pathlib import Path
from typing import Any

from forge_agent.memory.in_memory_store import InMemoryStore
from forge_agent.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
)

type MemoryRecordList = list[MemoryRecord]
type MemorySearchResultList = list[MemorySearchResult]


class JsonlMemoryStore:
    """JSONL-backed implementation of MemoryStore.

    Records are stored by memory type:

    - semantic.jsonl
    - episodic.jsonl
    - summary.jsonl
    """

    def __init__(self, memory_path: Path | str) -> None:
        self._memory_path = Path(memory_path)
        self._memory_path.mkdir(parents=True, exist_ok=True)
        self._store = InMemoryStore(self._load_records())

    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Persist one memory record.

        Duplicate records are skipped by the underlying InMemoryStore. Only new
        records are appended to the JSONL file.
        """

        before_ids = {existing.id for existing in self._store.list()}
        stored = self._store.add(record)

        if stored.id not in before_ids:
            self._append(stored)

        return stored

    def search(
        self,
        query: str,
        scope: MemoryScope | None = None,
        top_k: int = 5,
        *,
        scope_id: str | None = None,
        type_filter: MemoryType | None = None,
    ) -> MemorySearchResultList:
        """Search memory records by deterministic keyword overlap."""

        return self._store.search(
            query,
            scope=scope,
            top_k=top_k,
            scope_id=scope_id,
            type_filter=type_filter,
        )

    def list(
        self,
        scope: MemoryScope | None = None,
        *,
        scope_id: str | None = None,
        type_filter: MemoryType | None = None,
    ) -> MemoryRecordList:
        """List memory records with optional filters."""

        return self._store.list(
            scope=scope,
            scope_id=scope_id,
            type_filter=type_filter,
        )

    def update(self, record_id: str, patch: dict[str, Any]) -> MemoryRecord:
        """Patch one memory record and rewrite JSONL files."""

        updated = self._store.update(record_id, patch)
        self._rewrite_all()
        return updated

    def delete(self, record_id: str) -> None:
        """Delete one memory record and rewrite JSONL files."""

        self._store.delete(record_id)
        self._rewrite_all()

    def _load_records(self) -> MemoryRecordList:
        records: MemoryRecordList = []

        for memory_type in MemoryType:
            path = self._file_for(memory_type)
            if not path.exists():
                continue

            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                records.append(MemoryRecord.model_validate_json(line))

        return records

    def _append(self, record: MemoryRecord) -> None:
        path = self._file_for(record.type)
        with path.open("a", encoding="utf-8") as file:
            file.write(record.model_dump_json())
            file.write("\n")

    def _rewrite_all(self) -> None:
        records_by_type: dict[MemoryType, MemoryRecordList] = {
            memory_type: [] for memory_type in MemoryType
        }

        for record in self._store.list():
            records_by_type[record.type].append(record)

        for memory_type, records in records_by_type.items():
            path = self._file_for(memory_type)
            if not records:
                if path.exists():
                    path.unlink()
                continue

            content = "\n".join(record.model_dump_json() for record in records)
            path.write_text(f"{content}\n", encoding="utf-8")

    def _file_for(self, memory_type: MemoryType) -> Path:
        return self._memory_path / f"{memory_type.value}.jsonl"
