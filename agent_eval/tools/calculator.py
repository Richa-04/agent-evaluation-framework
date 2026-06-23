"""Calculator tool: safely evaluate a basic arithmetic expression.

Uses an AST walk with an allowlist of node/operator types — no ``eval``, no name
lookups, no calls. Supports + - * / // % ** and parentheses over numeric literals.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from .base import Tool, ToolError

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Guardrail so e.g. 9**9**9 can't lock the process.
_MAX_EXPONENT = 1000


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and right > _MAX_EXPONENT:
            raise ToolError(f"exponent too large (max {_MAX_EXPONENT})")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def _safe_eval(expression: str) -> float | int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid arithmetic syntax: {exc.msg}") from exc
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a basic arithmetic expression and return the numeric result. "
        "Supports +, -, *, /, //, %, ** and parentheses over numbers, "
        "e.g. '(12 + 8) * 5'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression to evaluate, e.g. '47 * 19'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    async def _execute(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ToolError("'expression' must be a non-empty string")
        value = _safe_eval(expression)
        return _format_number(value), {"expression": expression, "value": value}
