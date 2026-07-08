from __future__ import annotations

from dataclasses import dataclass

_TRUNCATION_MARKER = "\n\n[context truncated]"


@dataclass(frozen=True)
class ContextBudgetResult:
    """Result of applying a character budget to context text."""

    text: str
    original_chars: int
    final_chars: int
    truncated: bool


def enforce_context_budget(text: str, *, max_chars: int) -> ContextBudgetResult:
    """Truncate context text to a deterministic character budget."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    original_chars = len(text)
    if original_chars <= max_chars:
        return ContextBudgetResult(
            text=text,
            original_chars=original_chars,
            final_chars=original_chars,
            truncated=False,
        )

    if max_chars <= len(_TRUNCATION_MARKER):
        truncated_text = text[:max_chars]
    else:
        prefix_length = max_chars - len(_TRUNCATION_MARKER)
        truncated_text = f"{text[:prefix_length]}{_TRUNCATION_MARKER}"

    return ContextBudgetResult(
        text=truncated_text,
        original_chars=original_chars,
        final_chars=len(truncated_text),
        truncated=True,
    )
