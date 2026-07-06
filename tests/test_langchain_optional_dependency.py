from __future__ import annotations

from forge_agent.integrations.langchain.errors import (
    LangChainIntegrationError,
    missing_langchain_dependency_error,
)


def test_missing_langchain_dependency_error_is_clear() -> None:
    error = missing_langchain_dependency_error()

    assert isinstance(error, LangChainIntegrationError)
    assert "LangChain integration requires optional dependency" in str(error)
    assert "uv sync --extra langchain" in str(error)
