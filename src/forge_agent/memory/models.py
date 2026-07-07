from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryType(StrEnum):
    """Supported long-term memory categories."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    SUMMARY = "summary"


class MemoryScope(StrEnum):
    """Isolation boundary for long-term memory."""

    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    PROJECT = "project"


class MemoryRecord(BaseModel):
    """A structured long-term memory entry."""

    id: str = Field(default_factory=lambda: f"mem_{uuid4().hex}")
    type: MemoryType
    scope: MemoryScope
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str
    scope_id: str | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory content must not be blank")
        return normalized

    @field_validator("source")
    @classmethod
    def source_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory source must not be blank")
        return normalized

    @model_validator(mode="after")
    def updated_at_must_not_precede_created_at(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        return self

    def belongs_to_scope(
        self,
        *,
        scope: MemoryScope | None = None,
        scope_id: str | None = None,
    ) -> bool:
        """Return whether this record matches the requested scope filter."""

        if scope is not None and self.scope != scope:
            return False
        if scope_id is not None and self.scope_id != scope_id:
            return False
        return True


class MemorySearchResult(BaseModel):
    """A ranked memory retrieval result."""

    record: MemoryRecord
    score: float = Field(ge=0.0)
    reason: str
    rank: int = Field(ge=1)


class MemoryWriteDecision(BaseModel):
    """Decision produced by a memory write policy."""

    should_write: bool
    reason: str
    content: str | None = None
    memory_type: MemoryType | None = None
    scope: MemoryScope | None = None
    scope_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "memory_write_policy"

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory write decision reason must not be blank")
        return normalized

    def to_record(self) -> MemoryRecord:
        """Convert an accepted write decision into a MemoryRecord."""

        if not self.should_write:
            raise ValueError("cannot convert a skipped memory write decision to a record")
        if self.content is None or self.memory_type is None or self.scope is None:
            raise ValueError("accepted memory write decision requires content, type, and scope")

        return MemoryRecord(
            type=self.memory_type,
            scope=self.scope,
            scope_id=self.scope_id,
            content=self.content,
            metadata=dict(self.metadata),
            importance=self.importance,
            confidence=self.confidence,
            source=self.source,
        )
