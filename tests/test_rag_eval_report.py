import json
from pathlib import Path

from forge_agent.rag.evals import (
    RagRetrievalEvalSuite,
    rag_retrieval_eval_suite_to_dict,
    render_rag_retrieval_markdown_report,
    run_rag_retrieval_eval,
    write_rag_retrieval_json_report,
    write_rag_retrieval_markdown_report,
)


def test_render_rag_retrieval_markdown_report_includes_summary_and_failures(
    tmp_path: Path,
) -> None:
    suite = _run_suite(tmp_path)

    markdown = render_rag_retrieval_markdown_report(suite)

    assert "# RAG Retrieval Evaluation Report" in markdown
    assert "## Configuration" in markdown
    assert "## Summary Metrics" in markdown
    assert "## Case Results" in markdown
    assert "## Failure Analysis" in markdown
    assert "source_hit_rate" in markdown
    assert "no_answer_accuracy" in markdown
    assert "no_answer_case" in markdown
    assert "expected no answer" in markdown


def test_write_rag_retrieval_markdown_report(
    tmp_path: Path,
) -> None:
    suite = _run_suite(tmp_path)
    output = tmp_path / "reports" / "rag-eval-report.md"

    write_rag_retrieval_markdown_report(suite, output)

    assert output.exists()
    assert "# RAG Retrieval Evaluation Report" in output.read_text(encoding="utf-8")


def test_write_rag_retrieval_json_report(
    tmp_path: Path,
) -> None:
    suite = _run_suite(tmp_path)
    output = tmp_path / "reports" / "rag-eval-results.json"

    write_rag_retrieval_json_report(suite, output)

    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["retriever"] == "keyword"
    assert data["reranker"] == "none"
    assert data["summary"]["total_cases"] == 2
    assert data["cases"][0]["case_id"] == "security_case"


def test_rag_retrieval_eval_suite_to_dict_is_json_serializable(
    tmp_path: Path,
) -> None:
    suite = _run_suite(tmp_path)

    data = rag_retrieval_eval_suite_to_dict(suite)

    assert json.loads(json.dumps(data))["summary"]["total_cases"] == 2


def _run_suite(tmp_path: Path) -> RagRetrievalEvalSuite:
    knowledge_base = tmp_path / "knowledge_base"
    dataset = tmp_path / "rag_eval.jsonl"
    knowledge_base.mkdir()
    (knowledge_base / "security.md").write_text(
        "# Security\n\n"
        "## Workspace Guard\n\n"
        "Workspace guard checks file tool permissions and blocks path escape.\n",
        encoding="utf-8",
    )
    dataset.write_text(
        "\n".join(
            [
                (
                    '{"case_id":"security_case",'
                    '"query":"workspace guard permission",'
                    '"expected_sources":["security.md"],'
                    '"expected_terms":["workspace guard","permission"],'
                    '"top_k":3,'
                    '"expect_no_answer":false}'
                ),
                (
                    '{"case_id":"no_answer_case",'
                    '"query":"workspace guard",'
                    '"expected_sources":[],'
                    '"expected_terms":[],'
                    '"top_k":3,'
                    '"expect_no_answer":true}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    return run_rag_retrieval_eval(
        dataset_path=dataset,
        knowledge_base_path=knowledge_base,
        retriever_type="keyword",
    )
