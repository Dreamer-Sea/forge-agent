from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.rag.evals import RagRetrievalEvalCase, load_rag_retrieval_eval_cases


def test_load_rag_retrieval_eval_cases_loads_jsonl() -> None:
    cases = load_rag_retrieval_eval_cases("examples/evals/rag_retrieval.jsonl")

    assert cases
    assert all(isinstance(case, RagRetrievalEvalCase) for case in cases)
    assert cases[0].case_id == "rag_security_permission_system"
    assert cases[0].expected_sources == ("security.md",)
    assert cases[0].top_k == 3


def test_load_rag_retrieval_eval_cases_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_rag_retrieval_eval_cases(tmp_path / "missing.jsonl")


def test_load_rag_retrieval_eval_cases_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rag.jsonl"
    dataset.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_rag_retrieval_eval_cases(dataset)


def test_load_rag_retrieval_eval_cases_rejects_duplicate_case_id(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rag.jsonl"
    dataset.write_text(
        "\n".join(
            [
                _case_json(case_id="duplicate"),
                _case_json(case_id="duplicate"),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate RAG eval case_id"):
        load_rag_retrieval_eval_cases(dataset)


def test_load_rag_retrieval_eval_cases_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rag.jsonl"
    dataset.write_text(_case_json(top_k=0), encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer top_k"):
        load_rag_retrieval_eval_cases(dataset)


def test_load_rag_retrieval_eval_cases_rejects_no_answer_expected_sources(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rag.jsonl"
    dataset.write_text(
        _case_json(expect_no_answer=True, expected_sources=["security.md"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not define expected_sources"):
        load_rag_retrieval_eval_cases(dataset)


def _case_json(
    *,
    case_id: str = "case_001",
    top_k: int = 3,
    expect_no_answer: bool = False,
    expected_sources: list[str] | None = None,
) -> str:
    sources = ["security.md"] if expected_sources is None else expected_sources
    expected_terms = [] if expect_no_answer else ["permission"]

    return (
        "{"
        f'"case_id":"{case_id}",'
        '"query":"How does permission work?",'
        f'"expected_sources":{sources!r},'
        f'"expected_terms":{expected_terms!r},'
        f'"top_k":{top_k},'
        f'"expect_no_answer":{str(expect_no_answer).lower()}'
        "}"
    ).replace("'", '"')
