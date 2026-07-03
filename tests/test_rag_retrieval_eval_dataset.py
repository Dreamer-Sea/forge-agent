from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path("examples/evals/rag_retrieval.jsonl")
KNOWLEDGE_BASE_PATH = Path("examples/knowledge_base")


def test_rag_retrieval_eval_dataset_exists() -> None:
    assert DATASET_PATH.exists()


def test_rag_retrieval_eval_dataset_has_valid_schema() -> None:
    cases = _load_cases()

    assert cases

    for case in cases:
        assert isinstance(case["case_id"], str)
        assert case["case_id"]
        assert isinstance(case["query"], str)
        assert case["query"]
        assert isinstance(case["expected_sources"], list)
        assert all(isinstance(source, str) for source in case["expected_sources"])
        assert isinstance(case["expected_terms"], list)
        assert all(isinstance(term, str) for term in case["expected_terms"])
        assert isinstance(case["top_k"], int)
        assert case["top_k"] > 0
        assert isinstance(case["expect_no_answer"], bool)


def test_rag_retrieval_eval_dataset_has_unique_case_ids() -> None:
    cases = _load_cases()
    case_ids = [case["case_id"] for case in cases]

    assert len(case_ids) == len(set(case_ids))


def test_rag_retrieval_eval_dataset_covers_required_scenarios() -> None:
    cases = _load_cases()
    case_ids = {case["case_id"] for case in cases}

    assert "rag_security_permission_system" in case_ids
    assert "rag_workspace_guard_path_escape" in case_ids
    assert "rag_runtime_agent_loop" in case_ids
    assert "rag_no_answer_billing_saga" in case_ids


def test_rag_retrieval_eval_expected_sources_exist() -> None:
    cases = _load_cases()

    for case in cases:
        for source in case["expected_sources"]:
            assert (KNOWLEDGE_BASE_PATH / source).exists(), source


def test_rag_retrieval_eval_dataset_has_no_answer_case() -> None:
    cases = _load_cases()
    no_answer_cases = [case for case in cases if case["expect_no_answer"]]

    assert no_answer_cases
    for case in no_answer_cases:
        assert case["expected_sources"] == []
        assert case["expected_terms"] == []


def _load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        DATASET_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        case = json.loads(line)
        assert isinstance(case, dict), line_number
        cases.append(case)

    return cases
