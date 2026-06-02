from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any


class NumericExpressionError(ValueError):
    """Raised when a numeric protocol expression cannot be resolved."""


def evaluate_numeric_expression(value: Any, symbols: Mapping[str, float] | None = None) -> float:
    """Evaluate a numeric literal or a small arithmetic expression safely."""

    if isinstance(value, bool):
        raise NumericExpressionError(f"Boolean values are not valid numeric expressions: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise NumericExpressionError(f"Unsupported numeric expression type: {type(value).__name__}")

    text = value.strip()
    if not text:
        raise NumericExpressionError("Empty numeric expression")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise NumericExpressionError(f"Invalid numeric expression '{text}': {exc.msg}") from exc

    return _eval_ast(tree.body, symbols or {})


def build_protocol_numeric_symbols(protocol_yaml: Mapping[str, Any]) -> dict[str, float]:
    """Resolve top-level numeric protocol variables, including derived expressions."""

    pending: dict[str, Any] = {
        key: value
        for key, value in protocol_yaml.items()
        if not isinstance(value, (dict, list))
    }
    resolved: dict[str, float] = {}

    progress = True
    while progress and pending:
        progress = False
        for key, value in list(pending.items()):
            try:
                resolved[key] = evaluate_numeric_expression(value, resolved)
            except NumericExpressionError:
                continue
            else:
                pending.pop(key)
                progress = True

    return resolved


def _eval_ast(node: ast.AST, symbols: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise NumericExpressionError(f"Unsupported constant in numeric expression: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise NumericExpressionError(f"Unknown numeric symbol '{node.id}'")
        return float(symbols[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_ast(node.operand, symbols)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    ):
        left = _eval_ast(node.left, symbols)
        right = _eval_ast(node.right, symbols)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        return left % right
    raise NumericExpressionError(
        f"Unsupported syntax in numeric expression: {ast.dump(node, include_attributes=False)}"
    )