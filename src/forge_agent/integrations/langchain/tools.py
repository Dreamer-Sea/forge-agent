from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, create_model

from forge_agent.integrations.langchain.errors import (
    missing_langchain_dependency_error,
)
from forge_agent.tools.base import Tool, ToolSchema
from forge_agent.tools.registry import ToolRegistry


def forge_tool_to_langchain_tool(tool: Tool) -> Any:
    """Convert a forge-agent tool into a LangChain StructuredTool.

    The adapter keeps forge-agent as the execution authority. Validation,
    permission checks, structured success/error results, and trace-ready result
    semantics stay owned by the original tool implementation.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as error:
        raise missing_langchain_dependency_error() from error

    schema = tool.schema()
    args_schema = _tool_schema_to_args_model(schema)

    def _run(**kwargs: Any) -> dict[str, Any]:
        result = tool.execute(arguments=kwargs)
        return result.model_dump()

    _run.__name__ = schema.name
    _run.__doc__ = schema.description

    return StructuredTool.from_function(
        func=_run,
        name=schema.name,
        description=schema.description,
        args_schema=args_schema,
    )


def forge_registry_to_langchain_tools(registry: ToolRegistry) -> list[Any]:
    """Convert all registered forge-agent tools into LangChain tools."""
    return [
        forge_tool_to_langchain_tool(registry.get(schema.name)) for schema in registry.schemas()
    ]


def _tool_schema_to_args_model(schema: ToolSchema) -> type[BaseModel]:
    """Convert a forge-agent JSON schema into a minimal Pydantic args model.

    The first version supports the JSON Schema subset used by current
    forge-agent tools. More complex JSON Schema features can be added later
    without changing the public adapter contract.
    """
    parameters = schema.parameters
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = _json_schema_type_to_python_type(field_schema)
        description = field_schema.get("description")
        default = ... if field_name in required else field_schema.get("default", None)

        fields[field_name] = (
            field_type,
            Field(default=default, description=description),
        )

    model_name = f"{_to_pascal_case(schema.name)}Args"
    return create_model(model_name, **fields)


def _json_schema_type_to_python_type(field_schema: dict[str, Any]) -> Any:
    schema_type = field_schema.get("type")

    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list[Any]
    if schema_type == "object":
        return dict[str, Any]

    return Any


def _to_pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in value.replace("-", "_").split("_") if part)
