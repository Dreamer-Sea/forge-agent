from __future__ import annotations

from dataclasses import dataclass

from forge_agent.context.budget import enforce_context_budget
from forge_agent.memory.models import MemorySearchResult
from forge_agent.providers.base import ModelMessage


@dataclass(frozen=True)
class ContextComposerResult:
    """Composed prompt context and diagnostics."""

    messages: list[ModelMessage]
    memory_context: str
    rag_context: str | None
    original_context_chars: int
    final_context_chars: int
    truncated: bool
    memory_count: int


class ContextComposer:
    """Compose user input, retrieved memory, and optional RAG context."""

    def __init__(self, *, max_context_chars: int = 4_000) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than 0")
        self._max_context_chars = max_context_chars

    def compose(
        self,
        *,
        user_input: str,
        memory_results: list[MemorySearchResult] | None = None,
        rag_context: str | None = None,
    ) -> ContextComposerResult:
        """Return model messages containing layered context and user input."""

        normalized_user_input = user_input.strip()
        if not normalized_user_input:
            raise ValueError("user_input must not be blank")

        memory_context = self._format_memory_context(memory_results or [])
        normalized_rag_context = self._normalize_optional_context(rag_context)

        sections = self._build_context_sections(
            memory_context=memory_context,
            rag_context=normalized_rag_context,
        )
        combined_context = "\n\n".join(sections)
        budgeted = enforce_context_budget(
            combined_context,
            max_chars=self._max_context_chars,
        )

        messages: list[ModelMessage] = []
        if budgeted.text:
            messages.append(
                ModelMessage(
                    role="system",
                    content=(
                        "Use the following retrieved context when it is relevant. "
                        "Memory is dynamic historical experience. "
                        "Knowledge base context is static project documentation.\n\n"
                        f"{budgeted.text}"
                    ),
                )
            )

        messages.append(ModelMessage(role="user", content=normalized_user_input))

        return ContextComposerResult(
            messages=messages,
            memory_context=memory_context,
            rag_context=normalized_rag_context,
            original_context_chars=budgeted.original_chars,
            final_context_chars=budgeted.final_chars,
            truncated=budgeted.truncated,
            memory_count=len(self._dedupe_memory_results(memory_results or [])),
        )

    def _build_context_sections(
        self,
        *,
        memory_context: str,
        rag_context: str | None,
    ) -> list[str]:
        sections: list[str] = []
        if memory_context:
            sections.append(memory_context)
        if rag_context is not None:
            sections.append(f"## Knowledge Base Context\n{rag_context}")
        return sections

    def _format_memory_context(self, results: list[MemorySearchResult]) -> str:
        deduped = self._dedupe_memory_results(results)
        if not deduped:
            return ""

        lines = ["## Long-term Memory"]
        for result in deduped:
            record = result.record
            scope = (
                f"{record.scope.value}:{record.scope_id}"
                if record.scope_id is not None
                else record.scope.value
            )
            lines.append(
                "- "
                f"[rank={result.rank} "
                f"score={result.score:.4f} "
                f"type={record.type.value} "
                f"scope={scope}] "
                f"{record.content}"
            )

        return "\n".join(lines)

    def _dedupe_memory_results(
        self,
        results: list[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        deduped: list[MemorySearchResult] = []
        seen_contents: set[str] = set()

        for result in sorted(results, key=lambda item: item.rank):
            normalized_content = " ".join(result.record.content.casefold().split())
            if normalized_content in seen_contents:
                continue
            seen_contents.add(normalized_content)
            deduped.append(result)

        return deduped

    def _normalize_optional_context(self, context: str | None) -> str | None:
        if context is None:
            return None

        normalized = context.strip()
        if not normalized:
            return None

        return normalized
