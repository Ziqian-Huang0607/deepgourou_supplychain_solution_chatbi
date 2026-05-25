"""
Self-correction and retry module for the ChatBI agent.

This module implements an intelligent error correction loop that:
1. Classifies execution errors into specific types
2. Builds targeted correction prompts with contextual guidance
3. Sends correction requests to an LLM
4. Re-executes corrected code with safeguards
5. Retries up to a configurable maximum (default 2)

Usage:
    engine = SelfCorrectionEngine(llm_client=my_llm_client)
    result = engine.execute_with_retry(
        question="1月20日有多少配送订单？",
        generated_code="df[df['date'] == '2024-01-20'].shape[0]",
        exec_func=lambda code: exec(code)
    )
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorCategory(Enum):
    """Taxonomy of runtime / syntax errors for targeted correction guidance."""

    SYNTAX = auto()           # SyntaxError, IndentationError
    NAME_LOOKUP = auto()      # KeyError, NameError, AttributeError on columns/vars
    TYPE_MISMATCH = auto()    # TypeError from incompatible operations
    VALUE_INVALID = auto()    # ValueError from bad arguments / filter conditions
    TIMEOUT = auto()          # Execution exceeded time limit
    IMPORT = auto()           # ImportError, ModuleNotFoundError
    INDEX = auto()            # IndexError
    MEMORY = auto()           # MemoryError
    GENERIC = auto()          # Everything else


class ErrorClassifier:
    """Classifies Python exceptions into ``ErrorCategory`` buckets."""

    # Maps exception type names → categories
    _CATEGORY_MAP: Dict[str, ErrorCategory] = {
        "SyntaxError": ErrorCategory.SYNTAX,
        "IndentationError": ErrorCategory.SYNTAX,
        "TabError": ErrorCategory.SYNTAX,
        "NameError": ErrorCategory.NAME_LOOKUP,
        "KeyError": ErrorCategory.NAME_LOOKUP,
        "AttributeError": ErrorCategory.NAME_LOOKUP,
        "TypeError": ErrorCategory.TYPE_MISMATCH,
        "ValueError": ErrorCategory.VALUE_INVALID,
        "IndexError": ErrorCategory.INDEX,
        "ImportError": ErrorCategory.IMPORT,
        "ModuleNotFoundError": ErrorCategory.IMPORT,
        "MemoryError": ErrorCategory.MEMORY,
    }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @classmethod
    def classify(cls, exception: BaseException) -> ErrorCategory:
        """Return the category for *exception*."""
        type_name = type(exception).__name__
        return cls._CATEGORY_MAP.get(type_name, ErrorCategory.GENERIC)

    @classmethod
    def classify_from_tb(cls, tb_str: str) -> ErrorCategory:
        """Infer category from a traceback string (best-effort)."""
        # Look for the final exception line
        lines = tb_str.strip().splitlines()
        if not lines:
            return ErrorCategory.GENERIC
        last_line = lines[-1]
        for type_name, cat in cls._CATEGORY_MAP.items():
            if type_name in last_line:
                return cat
        # Heuristic: timeout-related phrases
        if any(k in tb_str.lower() for k in ("timeout", "timed out", "deadline")):
            return ErrorCategory.TIMEOUT
        return ErrorCategory.GENERIC


# ---------------------------------------------------------------------------
# Correction guidance messages
# ---------------------------------------------------------------------------

_CORRECTION_GUIDANCE: Dict[ErrorCategory, str] = {
    ErrorCategory.SYNTAX: (
        "修复 Python 语法错误。注意括号匹配、引号闭合、缩进正确，"
        "以及不要使用中文标点符号作为代码符号。"
    ),
    ErrorCategory.NAME_LOOKUP: (
        "检查 DataFrame 列名和变量名是否正确。"
        "使用 df.columns 确认实际列名，注意大小写敏感和空格。"
        "如果是字典 key 错误，确认 key 是否存在。"
    ),
    ErrorCategory.TYPE_MISMATCH: (
        "检查操作前的数据类型。使用 .dtype 或 type() 确认类型，"
        "必要时进行类型转换（如 int(), str(), pd.to_numeric()）。"
        "字符串和数字不能直接运算，日期类型要先转换。"
    ),
    ErrorCategory.VALUE_INVALID: (
        "检查筛选条件和数值是否合法。确认日期格式、数值范围，"
        "以及过滤条件中的值是否在数据中存在。"
    ),
    ErrorCategory.TIMEOUT: (
        "简化代码，使用更高效的操作。避免全表扫描，"
        "优先使用向量化操作而非循环，必要时采样处理。"
    ),
    ErrorCategory.IMPORT: (
        "使用已导入的库和模块，不要引入未安装的包。"
        "可用库包括 pandas, numpy, datetime 等标准数据分析库。"
    ),
    ErrorCategory.INDEX: (
        "检查索引是否越界。使用 .iloc[] 或 .loc[] 前先确认长度，"
        "或用 .head()/.tail() 安全访问。"
    ),
    ErrorCategory.MEMORY: (
        "减少内存使用。只选择需要的列，分块处理，"
        "或用 .dropna() 先清理无用数据。"
    ),
    ErrorCategory.GENERIC: (
        "仔细检查代码逻辑。确保所有变量已定义，"
        "操作顺序正确，并考虑边界情况。"
    ),
}


# ---------------------------------------------------------------------------
# LLM client protocol
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Protocol describing the minimal LLM client interface."""

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Return the raw text output from the LLM."""
        ...


# ---------------------------------------------------------------------------
# Correction prompt builder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrectionContext:
    """Immutable context for building a correction prompt."""

    question: str
    generated_code: str
    error_message: str
    error_category: ErrorCategory
    retry_count: int
    previous_attempts: Tuple[str, ...] = ()
    data_schema_hint: str = ""


class CorrectionPromptBuilder:
    """Builds rich correction prompts tailored to the error category."""

    # System prompt template (shared across all retries)
    _SYSTEM_TEMPLATE: str = (
        "你是一位精通 pandas 和供应链数据分析的 Python 专家。"
        "用户用中文提问，你之前生成的代码在执行时出错了。"
        "请根据错误信息和修正建议，重新生成一段正确的 Python 代码。"
        "只输出纯代码，不要解释，不要 markdown 代码块标记（不要 ```），"
        "不要输出任何中文说明，只输出可执行的 Python 表达式或语句。"
    )

    # Correction prompt template
    _CORRECTION_TEMPLATE: str = textwrap.dedent(
        """\
        【用户问题】
        {question}

        {schema_hint}
        【之前生成的代码】
        {generated_code}

        【执行错误】
        {error_message}

        【错误类型】
        {error_category}

        【修正建议】
        {guidance}

        {previous_attempts_section}
        【要求】
        1. 修正上述错误，生成新的正确代码
        2. 代码必须能直接执行，返回结果
        3. 结果赋值给变量 `_chatbi_result`，或最后一个表达式就是结果
        4. 不要输出任何中文说明，不要 markdown 代码块
        5. 使用 pandas 进行数据分析
        6. 如果涉及日期筛选，注意日期格式转换
        7. 列名必须与数据集中的一致

        【修正后的代码】
        """
    )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, ctx: CorrectionContext) -> str:
        """Return a full correction prompt string."""
        guidance = _CORRECTION_GUIDANCE.get(
            ctx.error_category, _CORRECTION_GUIDANCE[ErrorCategory.GENERIC]
        )

        schema_section = (
            f"【数据表结构提示】\n{ctx.data_schema_hint}\n\n"
            if ctx.data_schema_hint
            else ""
        )

        if ctx.previous_attempts:
            prev_lines = "\n".join(
                f"尝试 {i+1}:\n{code}"
                for i, code in enumerate(ctx.previous_attempts)
            )
            prev_section = f"【之前的尝试（均失败）】\n{prev_lines}\n\n"
        else:
            prev_section = ""

        prompt = cls._CORRECTION_TEMPLATE.format(
            question=ctx.question,
            schema_hint=schema_section,
            generated_code=ctx.generated_code.strip(),
            error_message=ctx.error_message.strip(),
            error_category=ctx.error_category.name,
            guidance=guidance,
            previous_attempts_section=prev_section,
        )
        return prompt

    @classmethod
    def build_system_message(cls) -> str:
        """Return the system-level instruction string."""
        return cls._SYSTEM_TEMPLATE


# ---------------------------------------------------------------------------
# Code extraction / sanitisation
# ---------------------------------------------------------------------------

class CodeExtractor:
    """Extract executable Python code from raw LLM output."""

    # Patterns that commonly wrap code in LLM responses
    _MARKDOWN_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
    _RESULT_PREFIX_RE = re.compile(r"^_chatbi_result\s*=\s*", re.MULTILINE)

    @classmethod
    def extract(cls, raw: str) -> str:
        """
        Return the best-effort executable code fragment from *raw*.

        Steps:
        1. Look for markdown code fences.
        2. Fall back to the whole string.
        3. Strip leading/trailing whitespace and language tags.
        """
        raw = raw.strip()
        if not raw:
            raise ValueError("LLM returned empty code string.")

        # Try markdown fences first
        matches = cls._MARKDOWN_RE.findall(raw)
        if matches:
            # Use the longest match (heuristic: most likely the real code)
            code = max(matches, key=len).strip()
        else:
            code = raw

        # Remove common LLM artifacts
        code = code.replace("```python", "").replace("```", "").strip()

        # If the LLM wrapped the result in a _chatbi_result assignment but
        # also prints other things, try to keep only the assignment + any
        # helper lines.
        lines = code.splitlines()
        cleaned_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            # Drop lines that are obviously explanatory (Chinese chars outside strings)
            if cls._is_explanatory_line(stripped):
                continue
            cleaned_lines.append(line)

        if not cleaned_lines:
            raise ValueError("No executable code found after cleaning.")

        return "\n".join(cleaned_lines)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_explanatory_line(line: str) -> bool:
        """Heuristic: return True if *line* looks like Chinese explanatory text."""
        # Allow empty lines
        if not line:
            return False
        # Keep lines that are obviously Python (heuristic: common keywords/symbols)
        python_markers = (
            "import ", "from ", "def ", "class ", "return ", "print(",
            "=", "(", ")", "[", "]", "{", "}", ".", ",", ":", "#",
        )
        if any(line.startswith(m) or m in line for m in python_markers):
            return False
        # If the line is mostly Chinese characters, it's likely explanatory
        chinese_chars = sum(1 for ch in line if "\u4e00" <= ch <= "\u9fff")
        return chinese_chars > len(line) * 0.3


# ---------------------------------------------------------------------------
# Safe execution sandbox
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing corrected code."""

    success: bool
    result: Any = None
    error_message: str = ""
    error_category: ErrorCategory = ErrorCategory.GENERIC
    execution_time_ms: float = 0.0
    stdout: str = ""


class ExecutionSandbox:
    """
    Lightweight sandbox for executing generated pandas code.

    Runs code in a restricted globals dict so that:
    - No builtins are exposed unless explicitly allow-listed.
    - Only pandas/numpy/datetime and basic utilities are available.
    """

    # Allowed builtin names
    _ALLOWED_BUILTINS: Tuple[str, ...] = (
        "abs", "all", "any", "bin", "bool", "dict", "dir", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hasattr",
        "hash", "hex", "id", "int", "isinstance", "issubclass", "iter",
        "len", "list", "map", "max", "min", "next", "oct", "ord",
        "pow", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "type", "vars", "zip",
    )

    def __init__(self, data_globals: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        data_globals :
            Pre-populated namespace that contains the DataFrame(s) etc.
        """
        self._globals = self._build_safe_globals(data_globals or {})

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def execute(
        self,
        code: str,
        timeout_sec: float = 30.0,
    ) -> ExecutionResult:
        """
        Execute *code* safely and return an ``ExecutionResult``.

        The last expression's value is captured if it is an expression
        statement; otherwise the code is expected to assign ``_chatbi_result``.
        """
        start = time.perf_counter()

        # Quick syntax check before running
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return ExecutionResult(
                success=False,
                error_message=f"SyntaxError: {exc.msg} (line {exc.lineno})",
                error_category=ErrorCategory.SYNTAX,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Wrap last expression to capture its value automatically
        wrapped_code = self._wrap_last_expression(code)

        try:
            result = self._run_with_timeout(wrapped_code, timeout_sec)
            exec_time = (time.perf_counter() - start) * 1000

            if "_chatbi_result" in self._globals:
                return ExecutionResult(
                    success=True,
                    result=self._globals["_chatbi_result"],
                    execution_time_ms=exec_time,
                )
            return ExecutionResult(
                success=True,
                result=result,
                execution_time_ms=exec_time,
            )

        except TimeoutError:
            return ExecutionResult(
                success=False,
                error_message=f"Execution timed out after {timeout_sec}s.",
                error_category=ErrorCategory.TIMEOUT,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:
            exec_time = (time.perf_counter() - start) * 1000
            tb_str = traceback.format_exc()
            return ExecutionResult(
                success=False,
                error_message=f"{type(exc).__name__}: {exc}\n{tb_str}",
                error_category=ErrorClassifier.classify(exc),
                execution_time_ms=exec_time,
            )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_safe_globals(cls, data_globals: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a restricted globals dict."""
        # __builtins__ can be a dict or a module depending on context
        if isinstance(__builtins__, dict):
            safe_builtins = {name: __builtins__[name] for name in cls._ALLOWED_BUILTINS if name in __builtins__}
        else:
            safe_builtins = {name: getattr(__builtins__, name) for name in cls._ALLOWED_BUILTINS}
        # Allow open/read for data files if needed, but restrict write
        namespace: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "_chatbi_result": None,
        }
        # Inject standard data science libraries
        try:
            import pandas as pd
            namespace["pd"] = pd
        except ImportError:  # pragma: no cover
            pass
        try:
            import numpy as np
            namespace["np"] = np
        except ImportError:  # pragma: no cover
            pass
        try:
            from datetime import datetime, date, timedelta
            namespace["datetime"] = datetime
            namespace["date"] = date
            namespace["timedelta"] = timedelta
        except ImportError:  # pragma: no cover
            pass

        # User-provided data objects (DataFrames etc.)
        namespace.update(data_globals)
        return namespace

    @staticmethod
    def _wrap_last_expression(code: str) -> str:
        """
        If the last non-empty line is a simple expression, wrap it so its
        value is stored in ``_chatbi_result``.
        """
        lines = code.strip().splitlines()
        if not lines:
            return code

        # Find last non-empty, non-comment line
        last_idx = len(lines) - 1
        while last_idx >= 0 and not lines[last_idx].strip():
            last_idx -= 1
        if last_idx < 0:
            return code

        last_line = lines[last_idx].strip()

        # Skip if already an assignment or import/return/def/class
        skip_prefixes = (
            "import ", "from ", "def ", "class ", "return ",
            "if ", "for ", "while ", "with ", "try:", "except",
            "elif ", "else:", "finally:", "@",
        )
        if last_line.startswith(skip_prefixes):
            return code
        if "=" in last_line and not any(
            op in last_line for op in ("==", "!=", "<=", ">=", "<", ">", "+=", "-=")
        ):
            # Simple assignment already
            if "_chatbi_result" in last_line:
                return code

        # Heuristic: if last line looks like an expression, capture it
        try:
            ast.parse(last_line, mode="eval")
            # It's a valid expression — wrap it
            lines[last_idx] = f"_chatbi_result = ({last_line})"
            return "\n".join(lines)
        except SyntaxError:
            pass

        return code

    def _run_with_timeout(self, code: str, timeout_sec: float) -> Any:
        """Execute *code*; raise ``TimeoutError`` if it takes too long."""
        # We use a simple wall-clock guard — full sandboxing would need
        # multiprocessing / resource limits, but this is enough for a
        # competition environment where the runner itself enforces limits.
        import signal

        def _alarm_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Code execution exceeded {timeout_sec} seconds")

        # Save previous handler
        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(timeout_sec))
        try:
            exec(code, self._globals)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

        return self._globals.get("_chatbi_result")


# ---------------------------------------------------------------------------
# Self-correction engine
# ---------------------------------------------------------------------------

@dataclass
class CorrectionConfig:
    """Configuration for the self-correction engine."""

    max_retries: int = 2
    timeout_per_execution: float = 30.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    data_schema_hint: str = ""


@dataclass
class CorrectionResult:
    """Final outcome of the correction loop."""

    success: bool
    result: Any = None
    final_code: str = ""
    attempts: int = 0
    error_log: List[str] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    used_fallback: bool = False


class SelfCorrectionEngine:
    """
    Orchestrates the self-correction retry loop.

    Usage::

        engine = SelfCorrectionEngine(llm_client=my_llm)
        outcome = engine.execute_with_retry(
            question="...",
            generated_code="...",
            exec_namespace={"df": my_dataframe},
        )
        if outcome.success:
            print(outcome.result)
        else:
            print("Failed after", outcome.attempts, "attempts")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[CorrectionConfig] = None,
    ):
        self._llm = llm_client
        self._cfg = config or CorrectionConfig()
        self._classifier = ErrorClassifier()
        self._prompt_builder = CorrectionPromptBuilder()
        self._extractor = CodeExtractor()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def execute_with_retry(
        self,
        question: str,
        generated_code: str,
        exec_namespace: Optional[Dict[str, Any]] = None,
        original_error: str = "",
    ) -> CorrectionResult:
        """
        Run the correction loop.

        Parameters
        ----------
        question :
            Original Chinese natural-language question.
        generated_code :
            The (failing) code produced by the generation step.
        exec_namespace :
            Globals dict that holds the DataFrame(s).
        original_error :
            Error message from the first failed execution.

        Returns
        -------
        CorrectionResult
        """
        error_msg = original_error
        current_code = generated_code
        previous_attempts: List[str] = []
        error_log: List[str] = []
        total_time_ms = 0.0

        for attempt in range(1, self._cfg.max_retries + 1):
            logger.info("Correction attempt %d/%d", attempt, self._cfg.max_retries)

            # Build correction context
            err_cat = ErrorClassifier.classify_from_tb(error_msg)
            ctx = CorrectionContext(
                question=question,
                generated_code=current_code,
                error_message=error_msg,
                error_category=err_cat,
                retry_count=attempt,
                previous_attempts=tuple(previous_attempts),
                data_schema_hint=self._cfg.data_schema_hint,
            )

            # Generate corrected code via LLM
            try:
                corrected_code = self._request_correction(ctx)
            except Exception as exc:
                logger.exception("LLM correction request failed")
                error_log.append(f"Attempt {attempt}: LLM request failed: {exc}")
                break

            # Execute the corrected code
            sandbox = ExecutionSandbox(data_globals=exec_namespace)
            exec_result = sandbox.execute(
                corrected_code,
                timeout_sec=self._cfg.timeout_per_execution,
            )
            total_time_ms += exec_result.execution_time_ms

            if exec_result.success:
                logger.info("Correction succeeded on attempt %d", attempt)
                return CorrectionResult(
                    success=True,
                    result=exec_result.result,
                    final_code=corrected_code,
                    attempts=attempt,
                    error_log=error_log,
                    total_execution_time_ms=total_time_ms,
                )

            # Failed again — prepare for next iteration
            error_msg = exec_result.error_message
            error_log.append(
                f"Attempt {attempt}: {err_cat.name} → {error_msg[:500]}"
            )
            previous_attempts.append(current_code)
            current_code = corrected_code

        # Exhausted all retries
        logger.warning("Self-correction failed after %d attempts", self._cfg.max_retries)
        return CorrectionResult(
            success=False,
            final_code=current_code,
            attempts=len(previous_attempts) + 1,
            error_log=error_log,
            total_execution_time_ms=total_time_ms,
            used_fallback=True,
        )

    def correct_once(
        self,
        question: str,
        generated_code: str,
        error_message: str,
        previous_attempts: Optional[List[str]] = None,
        exec_namespace: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ExecutionResult]:
        """
        Single-shot correction: build prompt, call LLM, execute.

        Returns
        -------
        (corrected_code, execution_result)
        """
        err_cat = ErrorClassifier.classify_from_tb(error_message)
        ctx = CorrectionContext(
            question=question,
            generated_code=generated_code,
            error_message=error_message,
            error_category=err_cat,
            retry_count=1,
            previous_attempts=tuple(previous_attempts or []),
            data_schema_hint=self._cfg.data_schema_hint,
        )
        corrected_code = self._request_correction(ctx)
        sandbox = ExecutionSandbox(data_globals=exec_namespace)
        exec_result = sandbox.execute(
            corrected_code,
            timeout_sec=self._cfg.timeout_per_execution,
        )
        return corrected_code, exec_result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _request_correction(self, ctx: CorrectionContext) -> str:
        """Send correction prompt to LLM and extract executable code."""
        system_msg = self._prompt_builder.build_system_message()
        prompt = self._prompt_builder.build(ctx)

        full_prompt = f"{system_msg}\n\n{prompt}"

        raw_response = self._llm.generate(
            prompt=full_prompt,
            temperature=self._cfg.llm_temperature,
            max_tokens=self._cfg.llm_max_tokens,
        )

        return self._extractor.extract(raw_response)
