"""Dataset loader for deterministic agent eval cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class EvalDatasetError(ValueError):
    """Raised when an eval dataset cannot be loaded."""


class EvalCase(BaseModel):
    """One deterministic eval case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    expected_tools: list[str] = Field(default_factory=list)
    expected_contains: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_stopped_reason: str | None = None

    @field_validator("id", "input")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject strings that only contain whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("expected_tools", "expected_contains", "expected_sources")
    @classmethod
    def reject_blank_items(cls, value: list[str]) -> list[str]:
        """Reject blank expectation items."""
        cleaned: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("expectation items must not be blank")
            cleaned.append(stripped)
        return cleaned


class EvalDataset(BaseModel):
    """A collection of deterministic eval cases."""

    model_config = ConfigDict(extra="forbid")

    cases: list[EvalCase]

    @classmethod
    def load_jsonl(cls, path: str | Path) -> EvalDataset:
        """Load eval cases from a JSONL file with line-aware errors."""
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise EvalDatasetError(f"dataset not found: {dataset_path}")

        cases: list[EvalCase] = []

        with dataset_path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()

                if not line:
                    continue

                try:
                    payload: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvalDatasetError(
                        f"{dataset_path}:{line_number}: invalid json: {exc.msg}"
                    ) from exc

                if not isinstance(payload, dict):
                    raise EvalDatasetError(
                        f"{dataset_path}:{line_number}: expected json object"
                    )

                try:
                    cases.append(EvalCase.model_validate(payload))
                except ValidationError as exc:
                    raise EvalDatasetError(
                        f"{dataset_path}:{line_number}: invalid eval case: {exc}"
                    ) from exc

        return cls(cases=cases)