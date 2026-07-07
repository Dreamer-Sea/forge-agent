"""Verifier protocol for runtime reflection."""

from __future__ import annotations

from typing import Protocol

from forge_agent.reflection.models import VerificationResult
from forge_agent.runtime.base import RunResult
from forge_agent.runtime.state import AgentState


class Verifier(Protocol):
    """Protocol for deterministic runtime verification."""

    def verify(
        self,
        result: RunResult,
        state: AgentState | None = None,
    ) -> VerificationResult:
        """Verify an agent run result before it is returned to the caller."""
        ...
