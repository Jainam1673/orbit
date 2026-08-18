"""Safe isolated Python mathematical calculator tool for long-horizon reasoning agents."""

import ast
import math
import time
from fractions import Fraction
from typing import Any

from orbit.agents.tools.base import BaseTool, ToolResult

# Safe mathematical execution namespace
_SAFE_MATH_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "comb": math.comb,
    "perm": math.perm,
    "pi": math.pi,
    "e": math.e,
    "Fraction": Fraction,
    "math": math,
}

_FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.AsyncFunctionDef,
)


class SafeMathASTValidator(ast.NodeVisitor):
    """Validates that an AST contains only safe mathematical expressions."""

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, _FORBIDDEN_AST_NODES):
            raise TypeError(f"Forbidden syntax node: {type(node).__name__}")
        # Check attribute access to prevent dunder introspection (e.g. __class__, __subclasses__)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"Forbidden private attribute access: {node.attr}")
        self.generic_visit(node)


class PythonCalculatorTool(BaseTool):
    """Safe mathematical calculator evaluating expressions in an isolated environment."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluates a mathematical Python expression safely (e.g., '12 * 15 + 4' or 'comb(10, 3)').",
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate.",
                }
            },
            "required": ["expression"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        expression = args.get("expression", "")
        if not expression or not expression.strip():
            return ToolResult(
                success=False,
                output="",
                error="Empty expression provided.",
            )

        start_time = time.perf_counter()
        clean_expr = expression.strip()

        try:
            # 1. Parse and validate AST
            parsed_ast = ast.parse(clean_expr, mode="eval")
            validator = SafeMathASTValidator()
            validator.visit(parsed_ast)

            # 2. Evaluate in restricted namespace
            compiled = compile(parsed_ast, filename="<calculator>", mode="eval")
            result = eval(compiled, _SAFE_MATH_GLOBALS, {})

            latency = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=True,
                output=str(result),
                latency_ms=latency,
            )
        except (
            ValueError,
            TypeError,
            SyntaxError,
            ZeroDivisionError,
            OverflowError,
            NameError,
            AttributeError,
        ) as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                output="",
                latency_ms=latency,
                error=f"Evaluation error: {type(e).__name__}: {e}",
            )
