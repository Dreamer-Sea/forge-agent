from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.evals.dataset import EvalDataset, EvalDatasetError


def test_eval_dataset_loads_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"runtime_001","input":"Agent Runtime 有哪些核心模块？",'
                    '"expected_tools":["search_knowledge_base"],'
                    '"expected_contains":["ModelProvider","ToolRegistry"]}'
                ),
                (
                    '{"id":"file_001","input":"Read README.",'
                    '"expected_tools":["read_file"],'
                    '"expected_contains":["Agent Platform"]}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvalDataset.load_jsonl(dataset_path)

    assert len(dataset.cases) == 2
    assert dataset.cases[0].id == "runtime_001"
    assert dataset.cases[0].expected_tools == ["search_knowledge_base"]
    assert dataset.cases[1].expected_contains == ["Agent Platform"]


def test_eval_dataset_rejects_missing_id(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        '{"input":"Agent Runtime 有哪些核心模块？"}',
        encoding="utf-8",
    )

    with pytest.raises(EvalDatasetError) as exc_info:
        EvalDataset.load_jsonl(dataset_path)

    assert "invalid eval case" in str(exc_info.value)
    assert "1" in str(exc_info.value)


def test_eval_dataset_reports_invalid_json_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"id":"ok_001","input":"valid"}',
                '{"id":"bad_001","input":',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalDatasetError) as exc_info:
        EvalDataset.load_jsonl(dataset_path)

    assert "invalid json" in str(exc_info.value)
    assert "2" in str(exc_info.value)


def test_eval_dataset_rejects_non_object_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        '["not", "an", "object"]',
        encoding="utf-8",
    )

    with pytest.raises(EvalDatasetError) as exc_info:
        EvalDataset.load_jsonl(dataset_path)

    assert "expected json object" in str(exc_info.value)


def test_eval_dataset_ignores_blank_lines(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        '\n\n{"id":"runtime_001","input":"hello"}\n\n',
        encoding="utf-8",
    )

    dataset = EvalDataset.load_jsonl(dataset_path)

    assert len(dataset.cases) == 1
    assert dataset.cases[0].id == "runtime_001"
