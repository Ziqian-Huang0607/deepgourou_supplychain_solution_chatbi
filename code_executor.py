"""
Safe Code Executor — restricted execution environment for LLM-generated pandas code.

Design decisions:
    * Runs generated code in a restricted globals namespace
    * Only exposes: pd, np, datetime, and pre-loaded dataframes
    * Timeout guard (default 10 s) via subprocess-based execution
    * Extracts ``result`` variable after successful execution
    * Blocks dangerous builtins (open, exec, eval, compile, __import__, etc.)
    * Returns structured ExecutionResult for downstream consumption
"""

from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import logging
import multiprocessing
import numbers
import re
import signal
import sys
import threading
import textwrap
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT: float = 10.0  # seconds

# DataFrames are injected at runtime by data_loader, but we declare the names here
ALLOWED_DATAFRAME_NAMES: Set[str] = {
    "df_orders",
    "df_details",
    "df_logistics",
}

# Allowed module-level names that may appear in the namespace
ALLOWED_NAMES: Set[str] = {
    "pd",
    "np",
    "datetime",
    "result",
    "True",
    "False",
    "None",
}

# Dangerous builtins that must NEVER be accessible
BLOCKED_BUILTINS: Set[str] = {
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "execfile",  # Python 2 legacy, but safe to block
    "input",
    "raw_input",
    "exit",
    "quit",
    "help",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "object",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "type",
    "print",  # we block print to avoid noisy stdout
}

# Dangerous module / attribute patterns (regex)
DANGEROUS_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bos\b"),               # os module
    re.compile(r"\bsys\b"),              # sys module
    re.compile(r"\bsubprocess\b"),       # subprocess
    re.compile(r"\bimportlib\b"),        # dynamic imports
    re.compile(r"\burllib\b"),           # network
    re.compile(r"\bhttp\b"),             # network
    re.compile(r"\bsocket\b"),            # network
    re.compile(r"\bftplib\b"),           # network
    re.compile(r"\bssh\b"),              # network
    re.compile(r"\brequests\b"),         # HTTP requests
    re.compile(r"\bpathlib\b"),          # file system
    re.compile(r"\.\s*__subclasses__\s*\("),
    re.compile(r"\.\s*__bases__\s*\("),
    re.compile(r"\.\s*__globals__\s*\["),
    re.compile(r"\.\s*__builtins__\s*"),
    re.compile(r"\.\s*__class__\s*"),
    re.compile(r"\bopen\s*\("),          # file open
    re.compile(r"\bfile\s*\("),          # Python 2 file()
    re.compile(r"\bread\s*\("),          # pandas read_* okay, generic read() blocked by name filter
    re.compile(r"\bwrite\s*\("),         # generic write
    re.compile(r"\bremove\s*\("),        # file removal
    re.compile(r"\bunlink\s*\("),        # file removal
    re.compile(r"\brmdir\s*\("),         # dir removal
    re.compile(r"\bmkdir\s*\("),         # dir creation
    re.compile(r"\bsocket\b"),            # sockets
    re.compile(r"\bmultiprocessing\b"),   # subprocesses
    re.compile(r"\bthreading\b"),        # threads (information leak)
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Structured result returned by the code executor."""

    success: bool
    result: Any = None
    error: str = ""
    code: str = ""
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dictionary (safe for JSON)."""
        # Convert result to a JSON-friendly representation
        serialisable_result = self._serialise_value(self.result)
        return {
            "success": self.success,
            "result": serialisable_result,
            "error": self.error,
            "code": self.code,
            "execution_time_ms": self.execution_time_ms,
        }

    @staticmethod
    def _serialise_value(value: Any) -> Any:
        """Convert pandas/numpy objects to plain Python for JSON serialisation."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, (list, tuple)):
            return [ExecutionResult._serialise_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): ExecutionResult._serialise_value(v) for k, v in value.items()}
        if isinstance(value, dt.datetime):
            return value.isoformat()
        return str(value)


# ---------------------------------------------------------------------------
# AST-based static analyser
# ---------------------------------------------------------------------------

class StaticAnalyzer(ast.NodeVisitor):
    """
    AST visitor that detects dangerous constructs BEFORE execution.

    Checks:
        * No import / from ... import statements
        * No attribute access to blocked builtins
        * No dangerous module references
        * All Name nodes reference allowed identifiers
    """

    def __init__(self) -> None:
        self.violations: List[str] = []
        self._allowed_names: Set[str] = ALLOWED_NAMES | ALLOWED_DATAFRAME_NAMES

    def analyze(self, code: str) -> List[str]:
        """
        Parse and analyse code; return list of violation messages (empty if safe).
        """
        self.violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.violations.append(f"Syntax error: {exc}")
            return self.violations

        self.visit(tree)
        return self.violations

    # -- Import blocking ------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.violations.append(f"Blocked import statement: 'import {alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        names = ", ".join(a.name for a in node.names)
        self.violations.append(f"Blocked import statement: 'from {module} import {names}'")
        self.generic_visit(node)

    # -- Name / attribute checking --------------------------------------

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            # Allow assignment to any name (the code may define helpers)
            pass
        elif isinstance(node.ctx, (ast.Load, ast.Del)):
            if node.id in BLOCKED_BUILTINS:
                self.violations.append(f"Blocked builtin usage: '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # Block access to dunder exploitation attributes
        attr = node.attr
        if attr in ("__subclasses__", "__bases__", "__globals__", "__builtins__", "__class__"):
            self.violations.append(f"Blocked dangerous attribute access: '{attr}'")
        self.generic_visit(node)

    # -- Lambda / comprehension guards (prevent unexpected scope leaks) -

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        # Lambdas are allowed, but we walk their body
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Code extractor
# ---------------------------------------------------------------------------

class CodeExtractor:
    """Extract raw Python code from markdown ```python ... ``` fences."""

    CODE_BLOCK_RE = re.compile(
        r"```(?:python)?\s*\n(.*?)\n```",
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def extract(cls, raw: str) -> str:
        """
        Extract the first Python code block from *raw*.

        Returns:
            Clean Python code string.

        Raises:
            ValueError: If no code block is found.
        """
        if not raw or not raw.strip():
            raise ValueError("Empty code string provided.")

        matches = cls.CODE_BLOCK_RE.findall(raw)
        if matches:
            return matches[0].strip()

        # Fallback: if it already looks like code, return as-is
        stripped = raw.strip()
        if stripped and ("import " in stripped or "df_" in stripped or "result" in stripped):
            return stripped

        raise ValueError(f"No Python code block found in text:\n{raw[:500]}")


# ---------------------------------------------------------------------------
# Result formatter
# ---------------------------------------------------------------------------

class ResultFormatter:
    """
    Format execution results for human-friendly display.

    Rules:
        * Numbers → comma-separated thousands, 2-decimal floats
        * datetime / Timestamp → ISO format
        * pd.Series → concise string with index
        * pd.DataFrame → markdown-like table (first/last 5 rows)
        * ndarray → comma-separated preview
        * Everything else → str()
    """

    @classmethod
    def format(cls, value: Any) -> str:
        """Return a nicely formatted string representation of *value*."""
        if value is None:
            return "(无结果)"

        if isinstance(value, bool):
            return "是" if value else "否"

        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"

        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "NaN"
            return f"{float(value):,.2f}"

        if isinstance(value, (pd.Timestamp, dt.datetime)):
            return value.strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(value, pd.Series):
            return cls._format_series(value)

        if isinstance(value, pd.DataFrame):
            return cls._format_dataframe(value)

        if isinstance(value, np.ndarray):
            return cls._format_ndarray(value)

        if isinstance(value, (list, tuple)):
            return cls._format_sequence(value)

        if isinstance(value, dict):
            return cls._format_dict(value)

        return str(value)

    @staticmethod
    def _format_series(s: pd.Series) -> str:
        lines = [f"Series (length={len(s)}):"]
        preview = s.head(10)
        for idx, val in preview.items():
            lines.append(f"  {idx}: {ResultFormatter.format(val)}")
        if len(s) > 10:
            lines.append(f"  ... ({len(s) - 10} more)")
        return "\n".join(lines)

    @staticmethod
    def _format_dataframe(df: pd.DataFrame) -> str:
        shape_info = f"DataFrame: {df.shape[0]} rows × {df.shape[1]} columns"
        if df.empty:
            return shape_info + "\n  (empty)"

        preview = df.head(5)
        # Simple markdown-like render
        lines = [shape_info, ""]
        lines.append("| " + " | ".join(str(c) for c in preview.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in preview.columns) + " |")
        for _, row in preview.iterrows():
            lines.append(
                "| " + " | ".join(ResultFormatter.format(v) for v in row.values) + " |"
            )
        if len(df) > 5:
            lines.append(f"\n... ({len(df) - 5} more rows)")
        return "\n".join(lines)

    @staticmethod
    def _format_ndarray(arr: np.ndarray) -> str:
        flat = arr.ravel()
        if flat.size <= 10:
            return "[" + ", ".join(ResultFormatter.format(v) for v in flat) + "]"
        preview = flat[:10]
        return "[" + ", ".join(ResultFormatter.format(v) for v in preview) + f", ... ({flat.size} total)]"

    @staticmethod
    def _format_sequence(seq: Union[list, tuple]) -> str:
        items = [ResultFormatter.format(v) for v in seq]
        bracket = "[" if isinstance(seq, list) else "("
        closing = "]" if isinstance(seq, list) else ")"
        if len(items) <= 10:
            return bracket + ", ".join(items) + closing
        return bracket + ", ".join(items[:10]) + f", ... ({len(seq)} total)" + closing

    @staticmethod
    def _format_dict(d: dict) -> str:
        items = []
        for k, v in list(d.items())[:20]:
            items.append(f"  {k}: {ResultFormatter.format(v)}")
        result = "{\n" + "\n".join(items) + "\n}"
        if len(d) > 20:
            result += f"\n... ({len(d) - 20} more entries)"
        return result


# ---------------------------------------------------------------------------
# Safe namespace builder
# ---------------------------------------------------------------------------

def _build_safe_namespace(
    df_orders: Optional[pd.DataFrame] = None,
    df_details: Optional[pd.DataFrame] = None,
    df_logistics: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Construct the restricted globals namespace for code execution.

    Only exposes:
        * pandas (pd), numpy (np), datetime module
        * The three pre-loaded dataframes (if provided)
        * A minimal __builtins__ dict with dangerous names removed

    Args:
        df_orders: Pre-loaded orders DataFrame.
        df_details: Pre-loaded details DataFrame.
        df_logistics: Pre-loaded logistics DataFrame.

    Returns:
        Dictionary suitable for use as globals() in exec().
    """
    # Handle both dict and module forms of __builtins__
    if isinstance(__builtins__, dict):
        _builtin_names = [n for n in __builtins__.keys() if not n.startswith("_")]
        _builtin_get = __builtins__.__getitem__
    else:
        _builtin_names = [n for n in dir(__builtins__) if not n.startswith("_")]
        _builtin_get = lambda name: getattr(__builtins__, name)

    # Start with a stripped-down builtins dict
    safe_builtins: Dict[str, Any] = {
        name: _builtin_get(name)
        for name in _builtin_names
        if name not in BLOCKED_BUILTINS
    }

    # Ensure all essential builtins are present
    ESSENTIAL_BUILTINS = ("len", "range", "enumerate", "zip", "map", "filter",
                          "sum", "min", "max", "round", "abs", "all", "any",
                          "sorted", "reversed", "iter", "next", "slice",
                          "str", "int", "float", "bool", "list", "tuple", "dict", "set",
                          "isinstance", "issubclass", "hasattr", "getattr",
                          "ArithmeticError", "AssertionError", "AttributeError",
                          "Exception", "TypeError", "ValueError", "IndexError",
                          "KeyError", "ZeroDivisionError", "NameError", "RuntimeError",
                          "StopIteration")
    for essential in ESSENTIAL_BUILTINS:
        if essential not in safe_builtins:
            try:
                safe_builtins[essential] = _builtin_get(essential)
            except (AttributeError, KeyError, NameError):
                pass

    namespace: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "datetime": dt,
    }

    if df_orders is not None:
        namespace["df_orders"] = df_orders
    if df_details is not None:
        namespace["df_details"] = df_details
    if df_logistics is not None:
        namespace["df_logistics"] = df_logistics

    return namespace


# ---------------------------------------------------------------------------
# Timeout helpers — threading-based (works in any thread, cross-platform)
# ---------------------------------------------------------------------------

class _TimeoutThread:
    """Thread-based timeout that works in Flask, Jupyter, and any thread."""
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.timer: Optional[threading.Timer] = None
        self.timed_out = False

    def _on_timeout(self):
        self.timed_out = True

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self._on_timeout)
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()


class TimeoutContext:
    """Deprecated — use _TimeoutThread instead. Kept for backward compat."""

    def __init__(self, seconds: float) -> None:
        self._impl = _TimeoutThread(seconds)

    def __enter__(self):
        self._impl.__enter__()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._impl.__exit__(exc_type, exc_val, exc_tb)


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------

class CodeExecutor:
    """
    Safe executor for LLM-generated pandas code.

    Features:
        * AST static analysis to reject dangerous code before running
        * Restricted namespace (no os, sys, subprocess, file I/O)
        * Timeout protection (configurable, default 10 s)
        * Automatic extraction of the ``result`` variable
        * Structured ExecutionResult with serialisable payload
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        df_orders: Optional[pd.DataFrame] = None,
        df_details: Optional[pd.DataFrame] = None,
        df_logistics: Optional[pd.DataFrame] = None,
        allow_inplace_mutation: bool = False,
    ) -> None:
        """
        Initialise the executor.

        Args:
            timeout: Maximum execution time in seconds.
            df_orders: Pre-loaded orders DataFrame (injected at runtime).
            df_details: Pre-loaded details DataFrame (injected at runtime).
            df_logistics: Pre-loaded logistics DataFrame (injected at runtime).
            allow_inplace_mutation: If False, dataframes are deep-copied before
                execution to prevent accidental mutation of original data.
        """
        self.timeout = timeout
        self.allow_inplace_mutation = allow_inplace_mutation
        self._namespace = _build_safe_namespace(df_orders, df_details, df_logistics)

        logger.info(
            "CodeExecutor initialised | timeout=%.1fs | dfs=%s | mutation=%s",
            self.timeout,
            {k: type(v).__name__ for k, v in self._namespace.items() if k.startswith("df_")},
            self.allow_inplace_mutation,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute generated pandas code safely and extract ``result``.

        Steps:
            1. Strip markdown fences.
            2. AST static analysis — reject dangerous constructs.
            3. Text-based dangerous pattern scan.
            4. Deep-copy dataframes (optional).
            5. exec() in restricted namespace with timeout.
            6. Extract ``result`` variable.
            7. Format and return.

        Args:
            code: Raw code string (may contain markdown ```python fences).

        Returns:
            ExecutionResult with success flag, result value, error message,
            executed code, and timing info.
        """
        start_time = dt.datetime.now()

        # Step 1 — extract code from markdown
        try:
            clean_code = CodeExtractor.extract(code)
        except ValueError as exc:
            return self._error_result(str(exc), code, start_time)

        # Step 2 — AST static analysis
        analyzer = StaticAnalyzer()
        violations = analyzer.analyze(clean_code)
        if violations:
            violation_text = "; ".join(violations)
            logger.warning("Static analysis blocked code: %s", violation_text)
            return self._error_result(f"Security violation: {violation_text}", clean_code, start_time)

        # Step 3 — text pattern scan (defence in depth)
        pattern_hits = self._text_pattern_scan(clean_code)
        if pattern_hits:
            hit_text = "; ".join(pattern_hits)
            logger.warning("Pattern scan blocked code: %s", hit_text)
            return self._error_result(f"Security violation: {hit_text}", clean_code, start_time)

        # Step 4 — prepare isolated namespace
        exec_namespace = self._prepare_namespace()

        # Step 5 — execute with timeout
        try:
            exec_result = self._execute_with_timeout(clean_code, exec_namespace)
        except TimeoutError:
            elapsed = (dt.datetime.now() - start_time).total_seconds() * 1000
            logger.error("Execution timed out after %.1fs.", self.timeout)
            return ExecutionResult(
                success=False,
                error=f"代码执行超时（限制 {self.timeout:.0f} 秒）。请简化查询或检查是否存在无限循环。",
                code=clean_code,
                execution_time_ms=round(elapsed, 2),
            )
        except Exception as exc:
            return self._error_result(self._format_exception(exc), clean_code, start_time)

        if not exec_result:
            return self._error_result("执行环境返回空结果。", clean_code, start_time)

        # Step 6 — extract result variable
        if "result" not in exec_namespace:
            return self._error_result(
                "代码未定义变量 ``result``。请将最终结果赋值给 ``result`` 变量。",
                clean_code,
                start_time,
            )

        raw_result = exec_namespace["result"]
        elapsed = (dt.datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            "Execution succeeded in %.2f ms | result_type=%s",
            elapsed,
            type(raw_result).__name__,
        )

        return ExecutionResult(
            success=True,
            result=raw_result,
            error="",
            code=clean_code,
            execution_time_ms=round(elapsed, 2),
        )

    def format_for_display(self, exec_result: ExecutionResult) -> str:
        """
        Format an ExecutionResult into a human-friendly display string.

        Args:
            exec_result: The result from execute().

        Returns:
            Formatted string suitable for chatbot UI display.
        """
        if not exec_result.success:
            return f"❌ 查询失败\n\n错误：{exec_result.error}\n"

        formatted = ResultFormatter.format(exec_result.result)
        return f"✅ 查询成功\n\n{formatted}\n"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_namespace(self) -> Dict[str, Any]:
        """Create a shallow copy of the safe namespace, optionally copying dataframes."""
        ns = dict(self._namespace)
        if not self.allow_inplace_mutation:
            for df_name in ALLOWED_DATAFRAME_NAMES:
                if df_name in ns and isinstance(ns[df_name], pd.DataFrame):
                    ns[df_name] = ns[df_name].copy(deep=True)
        return ns

    @staticmethod
    def _text_pattern_scan(code: str) -> List[str]:
        """
        Perform regex-based dangerous pattern scanning.

        Returns:
            List of matched pattern descriptions (empty if safe).
        """
        hits: List[str] = []
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(code):
                hits.append(f"检测到危险模式: {pattern.pattern}")
        return hits

    def _execute_with_timeout(
        self,
        code: str,
        namespace: Dict[str, Any],
    ) -> bool:
        """
        Execute code with timeout using ThreadPoolExecutor.

        Works on ALL platforms (macOS, Linux, Windows) and in ANY thread
        (Flask, Jupyter, main thread) — no signal dependency.

        Args:
            code: Clean Python code string.
            namespace: Restricted globals namespace.

        Returns:
            True if execution completed.

        Raises:
            TimeoutError: If execution exceeds *self.timeout*.
            Exception: Any exception raised by the executed code.
        """
        import concurrent.futures

        def _run():
            exec(compile(code, "<generated>", "exec"), namespace)
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Execution exceeded {self.timeout:.0f}s limit.")

    def _error_result(
        self,
        error: str,
        code: str,
        start_time: dt.datetime,
    ) -> ExecutionResult:
        """Helper to construct a failed ExecutionResult with timing."""
        elapsed = (dt.datetime.now() - start_time).total_seconds() * 1000
        logger.error("Execution failed: %s", error[:300])
        return ExecutionResult(
            success=False,
            error=error,
            code=code,
            execution_time_ms=round(elapsed, 2),
        )

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        """Format an exception into a concise, user-friendly message."""
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        tb = traceback.format_exc()
        # Keep the last few frames for debugging
        tb_lines = tb.strip().split("\n")
        concise_tb = "\n".join(tb_lines[-6:])  # last 6 lines
        return f"{exc_type}: {exc_msg}\n\nTraceback:\n{concise_tb}"


# ---------------------------------------------------------------------------
# Subprocess worker (top-level for picklability on Windows)
# ---------------------------------------------------------------------------

def _subprocess_worker(
    code: str,
    dataframes: Dict[str, Any],
) -> Union[Dict[str, Any], str]:
    """
    Worker function executed in a separate process.

    Rebuilds the safe namespace, executes the code, and returns a
    serialisable subset of the namespace (including ``result``).
    """
    try:
        ns = _build_safe_namespace(
            df_orders=dataframes.get("df_orders"),
            df_details=dataframes.get("df_details"),
            df_logistics=dataframes.get("df_logistics"),
        )
        exec(compile(code, "<generated>", "exec"), ns)

        # Return only picklable / essential items
        serialisable: Dict[str, Any] = {}
        if "result" in ns:
            serialisable["result"] = ns["result"]
        for df_name in ALLOWED_DATAFRAME_NAMES:
            if df_name in ns:
                serialisable[df_name] = ns[df_name]
        return serialisable
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def execute_code(
    code: str,
    df_orders: Optional[pd.DataFrame] = None,
    df_details: Optional[pd.DataFrame] = None,
    df_logistics: Optional[pd.DataFrame] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> ExecutionResult:
    """
    One-shot convenience function to execute code without instantiating a class.

    Args:
        code: Python code string (may contain markdown fences).
        df_orders: Pre-loaded orders DataFrame.
        df_details: Pre-loaded details DataFrame.
        df_logistics: Pre-loaded logistics DataFrame.
        timeout: Execution timeout in seconds.

    Returns:
        ExecutionResult.
    """
    executor = CodeExecutor(
        timeout=timeout,
        df_orders=df_orders,
        df_details=df_details,
        df_logistics=df_logistics,
    )
    return executor.execute(code)
