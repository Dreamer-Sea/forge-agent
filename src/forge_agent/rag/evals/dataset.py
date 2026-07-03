"""Dataset loader for RAG retrieval evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RagRetrievalEvalCase:
    """A deterministic RAG retrieval evaluation case."""

    case_id: str
    query: str
    expected_sources: tuple[str, ...]
    expected_terms: tuple[str, ...]
    top_k: int
    expect_no_answer: bool


def load_rag_retrieval_eval_cases(path: str | Path) -> tuple[RagRetrievalEvalCase, ...]:
    """Load RAG retrieval evaluation cases from a JSONL file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"RAG eval dataset not found: {dataset_path}")
    if not dataset_path.is_file():
        raise ValueError(f"RAG eval dataset is not a file: {dataset_path}")

    cases: list[RagRetrievalEvalCase] = []
    seen_case_ids: set[str] = set()

    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in RAG eval dataset at line {line_number}: {error.msg}"
            ) from error

        case = _parse_case(raw, line_number=line_number)

        if case.case_id in seen_case_ids:
            raise ValueError(f"Duplicate RAG eval case_id at line {line_number}: {case.case_id}")
        seen_case_ids.add(case.case_id)
        cases.append(case)

    if not cases:
        raise ValueError(f"RAG eval dataset is empty: {dataset_path}")

    return tuple(cases)


def _parse_case(raw: Any, *, line_number: int) -> RagRetrievalEvalCase:
    if not isinstance(raw, dict):
        raise ValueError(f"RAG eval case at line {line_number} must be an object")

    case_id = _required_str(raw, "case_id", line_number=line_number)
    query = _required_str(raw, "query", line_number=line_number)
    expected_sources = _required_str_tuple(
        raw,
        "expected_sources",
        line_number=line_number,
    )
    expected_terms = _required_str_tuple(
        raw,
        "expected_terms",
        line_number=line_number,
    )
    top_k = _required_positive_int(raw, "top_k", line_number=line_number)
    expect_no_answer = _required_bool(
        raw,
        "expect_no_answer",
        line_number=line_number,
    )

    if expect_no_answer and expected_sources:
        raise ValueError(
            f"RAG no-answer case at line {line_number} must not define expected_sources"
        )
    if expect_no_answer and expected_terms:
        raise ValueError(f"RAG no-answer case at line {line_number} must not define expected_terms")

    return RagRetrievalEvalCase(
        case_id=case_id,
        query=query,
        expected_sources=expected_sources,
        expected_terms=expected_terms,
        top_k=top_k,
        expect_no_answer=expect_no_answer,
    )


def _required_str(raw: dict[str, Any], key: str, *, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"RAG eval case at line {line_number} must define non-empty string {key}")
    return value


def _required_str_tuple(
    raw: dict[str, Any],
    key: str,
    *,
    line_number: int,
) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"RAG eval case at line {line_number} must define list {key}")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(
            f"RAG eval case at line {line_number} must define non-empty strings in {key}"
        )
    return tuple(value)


def _required_positive_int(
    raw: dict[str, Any],
    key: str,
    *,
    line_number: int,
) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"RAG eval case at line {line_number} must define positive integer {key}")
    return value


def _required_bool(raw: dict[str, Any], key: str, *, line_number: int) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"RAG eval case at line {line_number} must define bool {key}")
    return value
