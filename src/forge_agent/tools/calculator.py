from __future__ import annotations

import ast
from typing import Any

from pydantic import BaseModel, ValidationError

from forge_agent.tools.base import ToolResult, ToolSchema


class CalculatorArgs(BaseModel):
    expression: str


class CalculatorTool:
    name = "calculator"
    description = "Evaluate a safe arithmetic expression."

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=CalculatorArgs.model_json_schema(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        try:
            args = CalculatorArgs.model_validate(arguments)
            parsed = ast.parse(args.expression, mode="eval")
            result = _evaluate_ast(parsed)

            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=True,
                payload={"expression": args.expression, "result": result},
            )
        except ValidationError as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="validation_error",
                error_message=str(error),
            )
        except (SyntaxError, ValueError, ZeroDivisionError) as error:
            return ToolResult(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                error_code="invalid_expression",
                error_message=str(error),
            )


def _evaluate_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_ast(node.body)

    if isinstance(node, ast.Constant):
        value = node.value

        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("Only numeric constants are supported")

        return float(value)

    if isinstance(node, ast.UnaryOp):
        value = _evaluate_ast(node.operand)

        if isinstance(node.op, ast.UAdd):
            return value

        if isinstance(node.op, ast.USub):
            return -value

        raise ValueError("Unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left)
        right = _evaluate_ast(node.right)

        if isinstance(node.op, ast.Add):
            return left + right

        if isinstance(node.op, ast.Sub):
            return left - right

        if isinstance(node.op, ast.Mult):
            return left * right

        if isinstance(node.op, ast.Div):
            return left / right

        raise ValueError("Unsupported binary operator")

    raise ValueError("Unsupported expression")