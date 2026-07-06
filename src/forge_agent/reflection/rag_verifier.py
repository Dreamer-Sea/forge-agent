"""RAG groundedness verifier for citation-aware answers."""

from __future__ import annotations

import re
from typing import Any

from forge_agent.reflection.models import VerificationResult
from forge_agent.runtime.base import RunResult
from forge_agent.runtime.state import AgentState
from forge_agent.tools.base import ToolResult


class RagVerifier:
    """Verify whether a RAG answer is grounded in retrieved citations."""

    _CITATION_PATTERN = re.compile(r"\[source:\s+[^\]]+\]")

    def verify(
        self,
        result: RunResult,
        state: AgentState | None = None,
    ) -> VerificationResult:
        """Verify citation groundedness for search_knowledge_base results."""
        del state

        rag_results = self._rag_tool_results(result)
        if not rag_results:
            return VerificationResult(
                passed=True,
                decision="accept",
                reasons=["no RAG tool result found; skipped RAG verification"],
                confidence=1.0,
            )

        reasons: list[str] = []
        missing_evidence: list[str] = []
        unsupported_claims: list[str] = []

        final_answer = (result.final_answer or "").strip()
        if not final_answer:
            reasons.append("final_answer is empty")
            missing_evidence.append("answer text")

        failed_rag_results = [tool for tool in rag_results if not tool.success]
        if failed_rag_results:
            reasons.append("RAG tool execution failed")

        retrieved_citations = self._retrieved_citations(rag_results)
        retrieved_chunks = self._retrieved_chunks(rag_results)

        if not retrieved_citations:
            reasons.append("RAG answer has no retrieved citations")
            missing_evidence.append("retrieved citations")

            if self._looks_like_strong_answer(final_answer):
                unsupported_claims.append(
                    "answer gives a direct claim despite empty retrieved citations"
                )

        answer_citations = self._answer_citations(final_answer)
        if retrieved_citations and not answer_citations:
            reasons.append("final_answer does not contain citations")
            missing_evidence.append("answer citations")

        invalid_citations = sorted(answer_citations - retrieved_citations)
        if invalid_citations:
            reasons.append("final_answer contains citations not found in retrieved context")
            unsupported_claims.extend(invalid_citations)

        if retrieved_citations and not self._answer_uses_retrieved_content(
            final_answer,
            retrieved_chunks,
        ):
            reasons.append("final_answer does not appear to use retrieved context")
            unsupported_claims.append("answer is not grounded in retrieved chunks")

        if not reasons:
            return VerificationResult(
                passed=True,
                decision="accept",
                reasons=["RAG groundedness verification passed"],
                confidence=1.0,
            )

        return VerificationResult(
            passed=False,
            decision="revise",
            reasons=reasons,
            missing_evidence=missing_evidence,
            unsupported_claims=unsupported_claims,
            confidence=0.3,
        )

    @staticmethod
    def _rag_tool_results(result: RunResult) -> list[ToolResult]:
        return [
            tool
            for tool in result.tool_results
            if tool.tool_name == "search_knowledge_base"
        ]

    @classmethod
    def _answer_citations(cls, answer: str) -> set[str]:
        return set(cls._CITATION_PATTERN.findall(answer))

    @staticmethod
    def _retrieved_citations(rag_results: list[ToolResult]) -> set[str]:
        citations: set[str] = set()

        for tool in rag_results:
            raw_citations = tool.payload.get("citations", [])
            if not isinstance(raw_citations, list):
                continue

            for raw_citation in raw_citations:
                if not isinstance(raw_citation, dict):
                    continue

                citation = raw_citation.get("citation")
                if isinstance(citation, str) and citation:
                    citations.add(citation)

        return citations

    @staticmethod
    def _retrieved_chunks(rag_results: list[ToolResult]) -> list[str]:
        chunks: list[str] = []

        for tool in rag_results:
            raw_chunks = tool.payload.get("chunks", [])
            if isinstance(raw_chunks, list):
                chunks.extend(RagVerifier._chunk_contents(raw_chunks))

            raw_context = tool.payload.get("context")
            if isinstance(raw_context, str) and raw_context.strip():
                chunks.append(raw_context)

        return chunks

    @staticmethod
    def _chunk_contents(raw_chunks: list[Any]) -> list[str]:
        contents: list[str] = []

        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, dict):
                continue

            content = raw_chunk.get("content")
            if isinstance(content, str) and content.strip():
                contents.append(content)

        return contents

    @staticmethod
    def _answer_uses_retrieved_content(answer: str, retrieved_chunks: list[str]) -> bool:
        if not retrieved_chunks:
            return False

        normalized_answer = RagVerifier._normalize_text(answer)
        for chunk in retrieved_chunks:
            normalized_chunk = RagVerifier._normalize_text(chunk)
            if not normalized_chunk:
                continue

            chunk_terms = {
                term
                for term in normalized_chunk.split()
                if len(term) >= 4 and not term.startswith("[source:")
            }
            if not chunk_terms:
                continue

            matched_terms = [term for term in chunk_terms if term in normalized_answer]
            if len(matched_terms) >= min(3, len(chunk_terms)):
                return True

        return False

    @staticmethod
    def _looks_like_strong_answer(answer: str) -> bool:
        if not answer:
            return False

        weak_markers = (
            "不知道",
            "无法确认",
            "没有检索到",
            "not enough information",
            "cannot determine",
            "no retrieved context",
        )
        normalized = answer.lower()
        return not any(marker in normalized for marker in weak_markers)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()
