"""Stateful symbolic Python and SymPy REPL tool with persistent workspace across multi-turn reasoning."""

from __future__ import annotations

import ast
import io
import math
import sys
from typing import Any, ClassVar

import numpy as np
import sympy as sp

from orbit.agents.tools.base import BaseTool, ToolResult


class SafeSymbolicASTValidator(ast.NodeVisitor):
    """AST security visitor enforcing safety restrictions on symbolic execution."""

    DISALLOWED_NODES: ClassVar[tuple[type[ast.AST], ...]] = (
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.AsyncFunctionDef,
    )

    DISALLOWED_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "__import__",
            "eval",
            "exec",
            "open",
            "compile",
            "globals",
            "locals",
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
            "input",
            "breakpoint",
            "exit",
            "quit",
            "os",
            "sys",
            "subprocess",
            "shutil",
        }
    )

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, self.DISALLOWED_NODES):
            raise SecurityError(
                f"Disallowed construct '{type(node).__name__}' detected."
            )
        super().visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.DISALLOWED_NAMES:
            raise SecurityError(f"Access to restricted identifier '{node.id}' is blocked.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise SecurityError(f"Access to private/dunder attribute '{node.attr}' is blocked.")
        self.generic_visit(node)


class SecurityError(Exception):
    """Raised when unsafe code execution is attempted."""


class StatefulSymbolicREPLTool(BaseTool):
    """Persistent, stateful Python and SymPy REPL environment for multi-turn symbolic reasoning."""

    def __init__(self):
        super().__init__(
            name="symbolic_repl",
            description=(
                "Stateful Python and SymPy REPL. Variables, symbols, and functions persist across turns. "
                "Useful for algebra, solving polynomial systems, matrix operations, and exact symbolic proofs."
            ),
        )
        self.validator = SafeSymbolicASTValidator()
        self.session_env: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        """Resets the persistent workspace to the initial clean symbolic state."""
        self.session_env = {
            "sp": sp,
            "sympy": sp,
            "Symbol": sp.Symbol,
            "symbols": sp.symbols,
            "solve": sp.solve,
            "solveset": sp.solveset,
            "simplify": sp.simplify,
            "expand": sp.expand,
            "factor": sp.factor,
            "diff": sp.diff,
            "integrate": sp.integrate,
            "Matrix": sp.Matrix,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "oo": sp.oo,
            "log": sp.log,
            "exp": sp.exp,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "math": math,
            "np": np,
            "numpy": np,
        }

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python/SymPy code block to execute in persistent state workspace.",
                }
            },
            "required": ["code"],
        }

    def execute(self, args: dict[str, Any] | str) -> ToolResult:
        """Executes symbolic code within the persistent workspace."""
        if isinstance(args, dict):
            code = args.get("code", "")
        else:
            code = str(args)

        clean_code = code.strip()
        if not clean_code:
            return ToolResult(success=False, output="", error="No code provided for execution.")

        # 1. AST Static Security Inspection
        try:
            parsed_ast = ast.parse(clean_code)
            self.validator.visit(parsed_ast)
        except SecurityError as sec_err:
            return ToolResult(
                success=False,
                output="",
                error=f"Security Violation: {sec_err}",
            )
        except SyntaxError as syn_err:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax Error: {syn_err}",
            )

        # 2. Execution with captured stdout
        stdout_buf = io.StringIO()
        old_stdout = sys.stdout

        try:
            sys.stdout = stdout_buf

            # If the last statement is an expression, evaluate it to return its value
            if parsed_ast.body and isinstance(parsed_ast.body[-1], ast.Expr):
                *exec_stmts, eval_expr = parsed_ast.body
                if exec_stmts:
                    exec_mod = ast.Module(body=exec_stmts, type_ignores=[])
                    exec(compile(exec_mod, "<symbolic_repl>", "exec"), self.session_env)  # noqa: S102

                eval_mod = ast.Expression(body=eval_expr.value)
                result_val = eval(compile(eval_mod, "<symbolic_repl>", "eval"), self.session_env)

                printed_out = stdout_buf.getvalue().strip()
                if result_val is not None:
                    final_out = f"{printed_out}\nResult: {result_val}".strip()
                else:
                    final_out = printed_out
            else:
                exec(compile(parsed_ast, "<symbolic_repl>", "exec"), self.session_env)  # noqa: S102
                final_out = stdout_buf.getvalue().strip()

            return ToolResult(
                success=True,
                output=final_out if final_out else "Execution completed with no output.",
            )

        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=stdout_buf.getvalue().strip(),
                error=f"Runtime Error: {type(exc).__name__}: {exc}",
            )
        finally:
            sys.stdout = old_stdout
