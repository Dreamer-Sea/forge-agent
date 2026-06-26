from __future__ import annotations

from forge_agent.rag.knowledge_base import KnowledgeBase
from forge_agent.security import PermissionPolicy, Workspace
from forge_agent.tools.calculator import CalculatorTool
from forge_agent.tools.echo import EchoTextTool
from forge_agent.tools.file_tools import ListFilesTool, ReadFileTool, WriteFileTool
from forge_agent.tools.rag_tool import SearchKnowledgeBaseTool
from forge_agent.tools.registry import ToolRegistry


def create_default_tool_registry(
    knowledge_base: KnowledgeBase | None = None,
    workspace: Workspace | None = None,
    permission_policy: PermissionPolicy | None = None,
    safe_knowledge_base_path: str | None = None,
) -> ToolRegistry:
    """Create the default tool registry."""

    registry = ToolRegistry()
    registry.register(
        ListFilesTool(
            workspace=workspace,
            permission_policy=permission_policy,
        )
    )
    registry.register(
        ReadFileTool(
            workspace=workspace,
            permission_policy=permission_policy,
        )
    )
    registry.register(
        WriteFileTool(
            workspace=workspace,
            permission_policy=permission_policy,
        )
    )
    registry.register(CalculatorTool())
    registry.register(EchoTextTool())

    if knowledge_base is not None:
        registry.register(
            SearchKnowledgeBaseTool(
                knowledge_base,
                permission_policy=permission_policy,
                safe_index_path=safe_knowledge_base_path,
            )
        )

    return registry
