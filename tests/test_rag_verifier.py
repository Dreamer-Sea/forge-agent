from forge_agent.reflection import RagVerifier
from forge_agent.runtime.base import RunResult
from forge_agent.tools.base import ToolResult


def rag_tool_result(
    *,
    citations: list[dict[str, object]],
    chunks: list[dict[str, object]] | None = None,
    context: str = "",
    success: bool = True,
) -> ToolResult:
    return ToolResult(
        tool_name="search_knowledge_base",
        success=success,
        payload={
            "query": "Agent Runtime",
            "top_k": 3,
            "index_path": "knowledge_base",
            "context": context,
            "citations": citations,
            "chunks": chunks or [],
        },
        error_code=None if success else "knowledge_base_search_failed",
        error_message=None if success else "search failed",
    )


def test_rag_verifier_skips_non_rag_result() -> None:
    result = RunResult(
        final_answer="Normal non-RAG answer.",
        stopped_reason="completed",
        steps=1,
    )

    verification = RagVerifier().verify(result)

    assert verification.passed is True
    assert verification.decision == "accept"


def test_rag_verifier_accepts_answer_with_retrieved_citation() -> None:
    citation = "[source: agent-runtime.md#Agent Runtime > Components chunk-1]"
    result = RunResult(
        final_answer=(
            "Agent Runtime includes Agent Loop, Model Provider, and Tool Registry. "
            f"{citation}"
        ),
        stopped_reason="completed",
        steps=1,
        tool_results=[
            rag_tool_result(
                citations=[
                    {
                        "source": "agent-runtime.md",
                        "heading_path": ["Agent Runtime", "Components"],
                        "chunk_id": "chunk-1",
                        "citation": citation,
                    }
                ],
                chunks=[
                    {
                        "source": "agent-runtime.md",
                        "heading_path": ["Agent Runtime", "Components"],
                        "chunk_id": "chunk-1",
                        "content": (
                            "Agent Runtime includes Agent Loop, Model Provider, "
                            "Tool Registry, Tool Executor, and Trace Recorder."
                        ),
                    }
                ],
            )
        ],
    )

    verification = RagVerifier().verify(result)

    assert verification.passed is True
    assert verification.decision == "accept"


def test_rag_verifier_rejects_answer_without_citation() -> None:
    citation = "[source: security.md#Security > Permission System chunk-2]"
    result = RunResult(
        final_answer="Permission System checks runtime policy before tool execution.",
        stopped_reason="completed",
        steps=1,
        tool_results=[
            rag_tool_result(
                citations=[
                    {
                        "source": "security.md",
                        "heading_path": ["Security", "Permission System"],
                        "chunk_id": "chunk-2",
                        "citation": citation,
                    }
                ],
                chunks=[
                    {
                        "source": "security.md",
                        "heading_path": ["Security", "Permission System"],
                        "chunk_id": "chunk-2",
                        "content": (
                            "Permission System checks tool name, input arguments, "
                            "workspace path, user approval, and runtime policy."
                        ),
                    }
                ],
            )
        ],
    )

    verification = RagVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "revise"
    assert "final_answer does not contain citations" in verification.reasons
    assert "answer citations" in verification.missing_evidence


def test_rag_verifier_rejects_citation_not_in_retrieved_context() -> None:
    retrieved_citation = "[source: security.md#Security > Permission System chunk-2]"
    invented_citation = "[source: invented.md#Fake chunk-9]"
    result = RunResult(
        final_answer=(
            "Permission System checks runtime policy before tool execution. "
            f"{invented_citation}"
        ),
        stopped_reason="completed",
        steps=1,
        tool_results=[
            rag_tool_result(
                citations=[
                    {
                        "source": "security.md",
                        "heading_path": ["Security", "Permission System"],
                        "chunk_id": "chunk-2",
                        "citation": retrieved_citation,
                    }
                ],
                chunks=[
                    {
                        "source": "security.md",
                        "heading_path": ["Security", "Permission System"],
                        "chunk_id": "chunk-2",
                        "content": (
                            "Permission System checks tool name, input arguments, "
                            "workspace path, user approval, and runtime policy."
                        ),
                    }
                ],
            )
        ],
    )

    verification = RagVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "revise"
    assert (
        "final_answer contains citations not found in retrieved context"
        in verification.reasons
    )
    assert invented_citation in verification.unsupported_claims


def test_rag_verifier_rejects_empty_retrieval_with_strong_answer() -> None:
    result = RunResult(
        final_answer="Agent Runtime definitely uses a distributed scheduler.",
        stopped_reason="completed",
        steps=1,
        tool_results=[rag_tool_result(citations=[], chunks=[])],
    )

    verification = RagVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "revise"
    assert "RAG answer has no retrieved citations" in verification.reasons
    assert (
        "answer gives a direct claim despite empty retrieved citations"
        in verification.unsupported_claims
    )


def test_rag_verifier_allows_empty_retrieval_with_uncertainty_answer() -> None:
    result = RunResult(
        final_answer="没有检索到足够上下文，无法确认答案。",
        stopped_reason="completed",
        steps=1,
        tool_results=[rag_tool_result(citations=[], chunks=[])],
    )

    verification = RagVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "revise"
    assert "RAG answer has no retrieved citations" in verification.reasons
    assert verification.unsupported_claims == []


def test_rag_verifier_rejects_failed_rag_tool() -> None:
    result = RunResult(
        final_answer="Knowledge base search completed successfully.",
        stopped_reason="completed",
        steps=1,
        tool_results=[rag_tool_result(citations=[], success=False)],
    )

    verification = RagVerifier().verify(result)

    assert verification.passed is False
    assert verification.decision == "revise"
    assert "RAG tool execution failed" in verification.reasons
