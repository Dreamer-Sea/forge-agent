from __future__ import annotations


class LangChainIntegrationError(RuntimeError):
    """Raised when LangChain integration cannot be used."""


def missing_langchain_dependency_error() -> LangChainIntegrationError:
    """Return a clear error for missing optional LangChain dependencies."""
    return LangChainIntegrationError(
        "LangChain integration requires optional dependency. "
        "Install it with: uv sync --extra langchain"
    )
