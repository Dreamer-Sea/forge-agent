from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from forge_agent.planning.models import Plan, PlanStep
from forge_agent.planning.planner import PlanningContext

if TYPE_CHECKING:
    from forge_agent.runtime.state import AgentState


class SimplePlanner:
    """A deterministic rule-based planner for tests and local demos."""

    _COMPOUND_SPLIT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*(?:然后|并且|\bthen\b|\band\b)\s*",
        re.IGNORECASE,
    )

    _RAG_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "rag",
        "retrieval",
        "retrieve",
        "knowledge base",
        "knowledge-base",
        "citation",
        "citations",
        "context",
        "知识库",
        "检索",
        "引用",
        "证据",
    )

    _FILE_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "workspace",
        "file",
        "files",
        "directory",
        "folder",
        "read file",
        "list files",
        "write file",
        "工作区",
        "文件",
        "目录",
        "读取",
        "写入",
    )

    def create_plan(
        self,
        user_input: str,
        context: PlanningContext | None = None,
    ) -> Plan:
        """Create a deterministic plan from user input."""

        normalized_input = user_input.strip()

        if self._looks_like_rag_task(normalized_input, context):
            steps = self._create_rag_steps(normalized_input)
        elif self._looks_like_file_task(normalized_input, context):
            steps = self._create_file_steps(normalized_input)
        else:
            steps = self._create_general_steps(normalized_input)

        return Plan(user_input=user_input, steps=steps)

    def select_next_step(self, plan: Plan, state: AgentState) -> PlanStep | None:
        """Return the next pending step whose dependencies have succeeded."""

        _ = state.current_step_id
        return plan.next_executable_step()

    def _create_general_steps(self, user_input: str) -> list[PlanStep]:
        parts = self._split_compound_task(user_input)

        if len(parts) == 1:
            return [
                PlanStep(
                    id="answer",
                    description=user_input,
                    expected_outcome="A direct answer is produced.",
                )
            ]

        return [
            PlanStep(
                id=f"step_{index}",
                description=part,
                expected_outcome=f"Completed: {part}",
            )
            for index, part in enumerate(parts, start=1)
        ]

    def _create_rag_steps(self, user_input: str) -> list[PlanStep]:
        return [
            PlanStep(
                id="retrieve_context",
                description=f"Retrieve context for: {user_input}",
                expected_outcome="Relevant context and citations are available.",
                tool_hint="search_knowledge_base",
                evidence_required=True,
            ),
            PlanStep(
                id="answer_with_context",
                description="Answer the user using the retrieved context.",
                expected_outcome="A grounded answer with citations is produced.",
                depends_on=["retrieve_context"],
            ),
        ]

    def _create_file_steps(self, user_input: str) -> list[PlanStep]:
        return [
            PlanStep(
                id="inspect_workspace",
                description=f"Inspect workspace context for: {user_input}",
                expected_outcome="Relevant workspace files or directories are identified.",
                tool_hint="list_files",
            ),
            PlanStep(
                id="execute_operation",
                description=f"Execute the requested file operation: {user_input}",
                expected_outcome="The requested file operation is completed or safely denied.",
                depends_on=["inspect_workspace"],
                tool_hint="file_tools",
            ),
            PlanStep(
                id="summarize_result",
                description="Summarize the file operation result.",
                expected_outcome="The user receives a clear summary of the operation result.",
                depends_on=["execute_operation"],
            ),
        ]

    def _split_compound_task(self, user_input: str) -> list[str]:
        parts = [
            part.strip(" ,，。.")
            for part in self._COMPOUND_SPLIT_PATTERN.split(user_input)
        ]
        return [part for part in parts if part]

    def _looks_like_rag_task(
        self,
        user_input: str,
        context: PlanningContext | None,
    ) -> bool:
        lower_input = user_input.lower()

        if context is not None and context.knowledge_base_enabled:
            return True

        return any(keyword in lower_input for keyword in self._RAG_KEYWORDS)

    def _looks_like_file_task(
        self,
        user_input: str,
        context: PlanningContext | None,
    ) -> bool:
        lower_input = user_input.lower()

        if context is not None and context.workspace_available:
            return True

        return any(keyword in lower_input for keyword in self._FILE_KEYWORDS)
