from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from forge_agent.planning.models import Plan, PlanStep

if TYPE_CHECKING:
    from forge_agent.runtime.state import AgentState


class PlanningContext(BaseModel):
    """Optional inputs that can influence planning without coupling to runtime."""

    knowledge_base_enabled: bool = False
    workspace_available: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Planner(Protocol):
    """Planner interface used by planning-capable runtimes."""

    def create_plan(
        self,
        user_input: str,
        context: PlanningContext | None = None,
    ) -> Plan:
        """Create a plan for the user input."""

    def select_next_step(self, plan: Plan, state: AgentState) -> PlanStep | None:
        """Select the next executable step for the current plan."""
