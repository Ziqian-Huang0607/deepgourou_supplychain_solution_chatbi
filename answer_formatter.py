"""
Natural language answer generation module for the ChatBI agent.

Converts execution results (numbers, DataFrames, Series, etc.) into fluent
Chinese natural-language answers.  The formatter is context-aware: it uses
the original question to decide how to phrase the answer (e.g. "有…单" vs
"增幅…%").

Usage:
    formatter = AnswerFormatter()
    answer = formatter.format(
        result=330,
        question="1月20日有多少配送订单？",
    )
    # → "1月20日当天有330个配送订单被处理。"

The module also supports:
- ``LLMAnswerFormatter`` which delegates the final polish to an LLM for
  maximum fluency (useful in competition settings where an LLM is available).
- ``SimpleAnswerFormatter`` which works completely offline and is faster.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class NumberFormatter:
    """Locale-aware numeric formatting (Chinese conventions)."""

    # Chinese grouping: 万 (10k), 亿 (100M)
    _WAN = 10_000
    _YI = 100_000_000

    @classmethod
    def format_int(cls, value: int, use_chinese_units: bool = True) -> str:
        """
        Format an integer with comma grouping.

        Examples:
            330        → "330"
            12500      → "12,500"
            1250000    → "125万"   (if use_chinese_units)
        """
        if not isinstance(value, int):
            value = int(value)

        if use_chinese_units:
            if abs(value) >= cls._YI:
                return f"{value / cls._YI:.2f}亿".rstrip("0").rstrip(".") + "亿"
            if abs(value) >= cls._WAN:
                wan_val = value / cls._WAN
                if wan_val == int(wan_val):
                    return f"{int(wan_val)}万"
                return f"{wan_val:.2f}".rstrip("0").rstrip(".") + "万"

        return f"{value:,}"

    @classmethod
    def format_float(
        cls,
        value: float,
        decimals: int = 2,
        use_percent: bool = False,
    ) -> str:
        """
        Format a float with fixed decimal places.

        Parameters
        ----------
        value :
            The number to format.
        decimals :
            Number of decimal places to keep.
        use_percent :
            If True, multiply by 100 and append "%".

        Examples:
            15.39              → "15.39"
            0.1539, percent    → "15.39%"
            1234.567, 1        → "1,234.6"
        """
        if not isinstance(value, (int, float)):
            value = float(value)

        if use_percent:
            value = value * 100

        # Use Decimal for correct rounding
        quantize_exp = Decimal(1) / (Decimal(10) ** decimals)
        rounded = Decimal(str(value)).quantize(quantize_exp, rounding=ROUND_HALF_UP)

        if use_percent:
            return f"{rounded}%"

        # Add comma grouping for integer part
        s = f"{rounded:,.{decimals}f}"
        # Strip trailing zeros if decimals > 0
        if decimals > 0:
            s = s.rstrip("0").rstrip(".")
        return s

    @classmethod
    def auto_format(cls, value: Union[int, float, Any]) -> str:
        """Auto-detect int vs float and format appropriately."""
        if isinstance(value, bool):
            return "是" if value else "否"
        if isinstance(value, str):
            return value
        if isinstance(value, (int,)) or (isinstance(value, (int, float)) and value == int(value)):
            return cls.format_int(int(value))
        try:
            return cls.format_float(float(value))
        except (ValueError, TypeError):
            return str(value)


# ---------------------------------------------------------------------------
# Result-type detection
# ---------------------------------------------------------------------------

class ResultType(Enum):
    """Classification of execution result types."""

    SCALAR_INT = auto()
    SCALAR_FLOAT = auto()
    SCALAR_BOOL = auto()
    SCALAR_STR = auto()
    SERIES = auto()
    DATAFRAME = auto()
    DATAFRAME_GROUPED = auto()   # DataFrame that looks like a group-by result
    TUPLE_PAIR = auto()          # (value, change_pct) pairs
    LIST = auto()
    DICT = auto()
    NONE = auto()
    UNKNOWN = auto()


class ResultInspector:
    """Inspects a Python value and determines its ``ResultType``."""

    @staticmethod
    def inspect(value: Any) -> ResultType:
        """Classify *value* into a ``ResultType``."""
        if value is None:
            return ResultType.NONE

        # Pandas Series
        try:
            import pandas as pd
            if isinstance(value, pd.Series):
                return ResultType.SERIES
            if isinstance(value, pd.DataFrame):
                if len(value.columns) <= 3 and len(value) > 1:
                    return ResultType.DATAFRAME_GROUPED
                return ResultType.DATAFRAME
        except ImportError:  # pragma: no cover
            pass

        # NumPy scalars (must come before int/float since np.int64 is a subclass)
        try:
            import numpy as np
            if isinstance(value, np.bool_):
                return ResultType.SCALAR_BOOL
            if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return ResultType.SCALAR_INT
            if isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                return ResultType.SCALAR_FLOAT
            if isinstance(value, np.ndarray):
                return ResultType.LIST
        except ImportError:  # pragma: no cover
            pass

        # Python builtins
        if isinstance(value, bool):
            return ResultType.SCALAR_BOOL
        if isinstance(value, int):
            return ResultType.SCALAR_INT
        if isinstance(value, float):
            return ResultType.SCALAR_FLOAT
        if isinstance(value, str):
            return ResultType.SCALAR_STR
        if isinstance(value, (list, tuple)):
            if (
                len(value) == 2
                and isinstance(value[0], (int, float))
                and isinstance(value[1], (int, float))
            ):
                return ResultType.TUPLE_PAIR
            return ResultType.LIST
        if isinstance(value, dict):
            return ResultType.DICT

        return ResultType.UNKNOWN


# ---------------------------------------------------------------------------
# Answer templates (Chinese)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnswerTemplate:
    """A reusable Chinese answer template with placeholders."""

    pattern: str

    def render(self, **kwargs: Any) -> str:
        """Fill placeholders and return the final string."""
        try:
            return self.pattern.format(**kwargs)
        except KeyError:
            # Fallback: return pattern as-is
            return self.pattern


# Templates keyed by (result_type, question_intent) — intent is guessed heuristically
_DEFAULT_TEMPLATES: Dict[Tuple[ResultType, str], AnswerTemplate] = {
    # --- Scalar int ---
    (ResultType.SCALAR_INT, "count"): AnswerTemplate(
        "{question_prefix}{formatted_value}{unit}。"
    ),
    (ResultType.SCALAR_INT, "default"): AnswerTemplate(
        "答案是 {formatted_value}{unit}。"
    ),
    # --- Scalar float ---
    (ResultType.SCALAR_FLOAT, "ratio"): AnswerTemplate(
        "{question_prefix}{formatted_value}。"
    ),
    (ResultType.SCALAR_FLOAT, "default"): AnswerTemplate(
        "结果为 {formatted_value}。"
    ),
    # --- Bool ---
    (ResultType.SCALAR_BOOL, "default"): AnswerTemplate(
        "{formatted_value}。"
    ),
    # --- String ---
    (ResultType.SCALAR_STR, "default"): AnswerTemplate(
        "{value}"
    ),
    # --- Tuple pair (value, pct_change) ---
    (ResultType.TUPLE_PAIR, "change"): AnswerTemplate(
        "{change_direction}{formatted_abs_change}{unit}，{change_word}{formatted_pct}。"
    ),
    (ResultType.TUPLE_PAIR, "default"): AnswerTemplate(
        "数值为 {formatted_value1}，占比/变化为 {formatted_value2}。"
    ),
    # --- Series ---
    (ResultType.SERIES, "default"): AnswerTemplate(
        "{summary}"
    ),
    # --- DataFrame ---
    (ResultType.DATAFRAME, "default"): AnswerTemplate(
        "{summary}"
    ),
    (ResultType.DATAFRAME_GROUPED, "default"): AnswerTemplate(
        "{summary}"
    ),
    # --- List ---
    (ResultType.LIST, "default"): AnswerTemplate(
        "结果包括：{items}。"
    ),
    # --- Dict ---
    (ResultType.DICT, "default"): AnswerTemplate(
        "{summary}"
    ),
    # --- None ---
    (ResultType.NONE, "default"): AnswerTemplate(
        "未查询到相关数据，请确认问题或数据范围是否正确。"
    ),
    # --- Unknown ---
    (ResultType.UNKNOWN, "default"): AnswerTemplate(
        "执行结果：{value}"
    ),
}


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

class QuestionIntentDetector:
    """Heuristic detector for the intent behind a Chinese question."""

    # Keywords that hint at counting
    _COUNT_KEYWORDS: Tuple[str, ...] = (
        "多少", "几", "数量", " count", "counts", "计数", "总数",
        "多少单", "多少笔", "多少个", "多少件", "多少条",
    )
    # Keywords that hint at ratio / percentage
    _RATIO_KEYWORDS: Tuple[str, ...] = (
        "占比", "比例", "百分比", "%", "percent", "率", "pct",
        "几成", "几分之", "多少成",
    )
    # Keywords that hint at change / growth
    _CHANGE_KEYWORDS: Tuple[str, ...] = (
        "增长", "增加", "减少", "下降", "上升", "降幅", "增幅",
        "同比", "环比", "比", "多了", "少了",
    )

    @classmethod
    def detect(cls, question: str, result_type: ResultType) -> str:
        """
        Guess the intent from the question text.

        Returns one of: ``count``, ``ratio``, ``change``, ``default``.
        """
        q = question.lower()

        if result_type == ResultType.TUPLE_PAIR:
            if any(k in q for k in cls._CHANGE_KEYWORDS):
                return "change"
            return "default"

        if any(k in q for k in cls._CHANGE_KEYWORDS) and result_type in (
            ResultType.SCALAR_INT,
            ResultType.SCALAR_FLOAT,
        ):
            return "change"

        if any(k in q for k in cls._RATIO_KEYWORDS):
            return "ratio"

        if any(k in q for k in cls._COUNT_KEYWORDS):
            return "count"

        return "default"


# ---------------------------------------------------------------------------
# Unit inference
# ---------------------------------------------------------------------------

class UnitInferrer:
    """Infers the appropriate unit from the question text."""

    _UNIT_PATTERNS: List[Tuple[str, str]] = [
        ("订单|单|配送单|外卖单", "单"),
        ("商品|SKU|sku|产品|货物", "件"),
        ("笔|交易|支付", "笔"),
        ("人|员工|骑手|配送员|用户|客户", "人"),
        ("次|配送次数|发货次数", "次"),
        ("天|日|工作日", "天"),
        ("小时|钟头|h ", "小时"),
        ("分钟|分|min", "分钟"),
        ("金额|元|块钱|费用|成本|收入|销售额|营收|GMV|gmv", "元"),
        ("吨|重量", "吨"),
        ("公里|km|千米", "公里"),
        ("个|项|条", "个"),
    ]

    @classmethod
    def infer(cls, question: str) -> str:
        """Return the inferred unit string, or empty string if unknown."""
        for pattern, unit in cls._UNIT_PATTERNS:
            if re.search(pattern, question):
                return unit
        return ""


# ---------------------------------------------------------------------------
# Main formatter
# ---------------------------------------------------------------------------

@dataclass
class FormatterConfig:
    """Configuration for ``AnswerFormatter``."""

    use_chinese_units: bool = True      # 万 / 亿 grouping
    max_series_items: int = 10          # Max items to show from a Series
    max_df_rows: int = 8                # Max rows to render from a DataFrame
    df_show_index: bool = False         # Whether to mention index values
    percent_decimals: int = 2
    float_decimals: int = 2
    use_llm_polish: bool = False        # Whether to call an LLM for final polish
    llm_client: Any = None              # LLM client instance (if use_llm_polish=True)


class AnswerFormatter:
    """
    Format execution results into natural Chinese answers.

    This is the primary class — it combines type detection, number formatting,
    template selection, and (optionally) LLM-based polishing.
    """

    def __init__(self, config: Optional[FormatterConfig] = None):
        self._cfg = config or FormatterConfig()
        self._inspector = ResultInspector()
        self._intent = QuestionIntentDetector()
        self._unit = UnitInferrer()
        self._numbers = NumberFormatter()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def format(self, result: Any, question: str = "") -> str:
        """
        Format *result* into a Chinese natural-language answer.

        Parameters
        ----------
        result :
            The value returned by executing the generated code.
        question :
            The original Chinese question (used for context-aware phrasing).

        Returns
        -------
        str
            A human-friendly Chinese answer.
        """
        result_type = self._inspector.inspect(result)
        intent = self._intent.detect(question, result_type)
        unit = self._unit.infer(question)

        # Build the answer via template
        answer = self._format_by_type(result, result_type, intent, unit, question)

        # Optional LLM polish
        if self._cfg.use_llm_polish and self._cfg.llm_client is not None:
            answer = self._polish_with_llm(answer, result, question)

        return answer

    # Alias for convenience
    __call__ = format

    # ------------------------------------------------------------------ #
    # Internal formatters by type
    # ------------------------------------------------------------------ #

    def _format_by_type(
        self,
        value: Any,
        result_type: ResultType,
        intent: str,
        unit: str,
        question: str,
    ) -> str:
        """Dispatch to the appropriate formatter."""
        method_name = f"_fmt_{result_type.name.lower()}"
        method = getattr(self, method_name, self._fmt_unknown)
        return method(value, intent, unit, question)

    def _fmt_scalar_int(self, value: int, intent: str, unit: str, question: str) -> str:
        formatted = self._numbers.format_int(value, self._cfg.use_chinese_units)
        template = self._get_template(ResultType.SCALAR_INT, intent)
        question_prefix = self._extract_question_prefix(question)

        # Smart phrasing for common patterns
        if "多少" in question or "几" in question:
            # Direct answer pattern
            if unit:
                return f"{question_prefix}{formatted}{unit}。"
            return f"{question_prefix}{formatted}。"

        return template.render(
            formatted_value=formatted,
            unit=unit,
            question_prefix=question_prefix,
        )

    def _fmt_scalar_float(self, value: float, intent: str, unit: str, question: str) -> str:
        # Only convert to percentage if value is a fraction (0-1 range)
        # Values > 1 are already percentage numbers (e.g., 3.14 means 3.14%)
        use_pct = (
            (intent == "ratio" or "%" in question or "比例" in question or "占比" in question)
            and 0 <= abs(value) <= 1
        )
        formatted = self._numbers.format_float(
            value,
            decimals=self._cfg.float_decimals,
            use_percent=use_pct,
        )

        if intent == "change":
            direction = "增加" if value > 0 else "减少"
            if use_pct:
                return f"{direction}了{formatted}。"
            if unit:
                return f"{direction}了{formatted}{unit}。"
            return f"{direction}了{formatted}。"

        template = self._get_template(ResultType.SCALAR_FLOAT, intent)
        question_prefix = self._extract_question_prefix(question)
        return template.render(
            formatted_value=formatted,
            unit=unit,
            question_prefix=question_prefix,
        )

    def _fmt_scalar_bool(self, value: bool, intent: str, unit: str, question: str) -> str:
        return "是" if value else "否"

    def _fmt_scalar_str(self, value: str, intent: str, unit: str, question: str) -> str:
        return value

    def _fmt_series(self, value: Any, intent: str, unit: str, question: str) -> str:
        """Format a pandas Series."""
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return str(value)

        if not isinstance(value, pd.Series):
            return str(value)

        # Single-value series → treat as scalar
        if len(value) == 1:
            only_val = value.iloc[0]
            return self.format(only_val, question)

        # Multi-value series → build bullet list
        items: List[str] = []
        display_count = min(len(value), self._cfg.max_series_items)
        for idx in range(display_count):
            idx_label = value.index[idx]
            val = value.iloc[idx]
            val_formatted = self._numbers.auto_format(val)
            items.append(f"{idx_label}: {val_formatted}{unit}")

        summary = "；".join(items)
        if len(value) > self._cfg.max_series_items:
            summary += f" 等共{len(value)}项"

        return f"{self._extract_question_prefix(question)}{summary}。"

    def _fmt_dataframe(self, value: Any, intent: str, unit: str, question: str) -> str:
        """Format a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return str(value)

        if not isinstance(value, pd.DataFrame):
            return str(value)

        # Empty
        if value.empty:
            return "查询结果为空，未找到符合条件的数据。"

        # Single row → flatten to scalar or dict
        if len(value) == 1:
            row = value.iloc[0]
            if len(value.columns) == 1:
                return self.format(row.iloc[0], question)
            # Multiple columns → format as dict-like summary
            parts = []
            for col in value.columns:
                parts.append(f"{col}为{self._numbers.auto_format(row[col])}{unit}")
            return f"{self._extract_question_prefix(question)}" + "，".join(parts) + "。"

        # Multiple rows → summary
        return self._summarise_dataframe(value, unit, question)

    def _fmt_dataframe_grouped(self, value: Any, intent: str, unit: str, question: str) -> str:
        """Format a grouped-style DataFrame (e.g. group-by result)."""
        return self._fmt_dataframe(value, intent, unit, question)

    def _fmt_tuple_pair(self, value: Tuple[Any, Any], intent: str, unit: str, question: str) -> str:
        """
        Format a (value, percentage) pair — common for "increase by X, Y%" answers.

        Examples:
            (367, 15.39) → "增加367单，增幅15.39%。"
            (-50, -5.0)  → "减少50单，降幅5%。"
        """
        raw_val, raw_pct = value

        abs_val = abs(raw_val) if isinstance(raw_val, (int, float)) else raw_val
        formatted_val = self._numbers.auto_format(abs_val)

        pct = raw_pct if isinstance(raw_pct, (int, float)) else 0
        formatted_pct = self._numbers.format_float(
            abs(pct), decimals=self._cfg.percent_decimals, use_percent=True
        )

        if intent == "change" or "比" in question:
            if raw_val > 0 or (isinstance(raw_val, (int, float)) and raw_val > 0):
                direction = "增加"
                change_word = "增幅"
            else:
                direction = "减少"
                change_word = "降幅"

            template = self._get_template(ResultType.TUPLE_PAIR, "change")
            return template.render(
                change_direction=direction,
                formatted_abs_change=formatted_val,
                unit=unit,
                change_word=change_word,
                formatted_pct=formatted_pct,
            )

        # Fallback
        template = self._get_template(ResultType.TUPLE_PAIR, "default")
        return template.render(
            formatted_value1=formatted_val,
            formatted_value2=formatted_pct,
        )

    def _fmt_list(self, value: List[Any], intent: str, unit: str, question: str) -> str:
        formatted_items = [self._numbers.auto_format(v) for v in value]
        items_str = "、".join(formatted_items)
        template = self._get_template(ResultType.LIST, intent)
        return template.render(items=items_str)

    def _fmt_dict(self, value: Dict[str, Any], intent: str, unit: str, question: str) -> str:
        parts = []
        for k, v in value.items():
            v_formatted = self._numbers.auto_format(v)
            parts.append(f"{k}为{v_formatted}{unit}")
        return "，".join(parts) + "。" if parts else "查询结果为空。"

    def _fmt_none(self, value: None, intent: str, unit: str, question: str) -> str:
        template = self._get_template(ResultType.NONE, "default")
        return template.render()

    def _fmt_unknown(self, value: Any, intent: str, unit: str, question: str) -> str:
        template = self._get_template(ResultType.UNKNOWN, "default")
        return template.render(value=str(value))

    # ------------------------------------------------------------------ #
    # DataFrame summariser
    # ------------------------------------------------------------------ #

    def _summarise_dataframe(self, df: Any, unit: str, question: str) -> str:
        """Create a concise Chinese summary of a multi-row DataFrame."""
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return str(df)

        rows = []
        display_rows = min(len(df), self._cfg.max_df_rows)

        for i in range(display_rows):
            row = df.iloc[i]
            row_parts = []
            for col in df.columns:
                cell = row[col]
                cell_str = self._numbers.auto_format(cell)
                row_parts.append(f"{col}={cell_str}")
            row_desc = "，".join(row_parts)
            idx_label = df.index[i]
            if self._cfg.df_show_index:
                rows.append(f"{idx_label}: {row_desc}")
            else:
                rows.append(row_desc)

        summary = "；".join(rows)
        if len(df) > self._cfg.max_df_rows:
            summary += f" 等共{len(df)}行数据"

        prefix = self._extract_question_prefix(question)
        return f"{prefix}{summary}。"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_template(result_type: ResultType, intent: str) -> AnswerTemplate:
        key = (result_type, intent)
        return _DEFAULT_TEMPLATES.get(
            key,
            _DEFAULT_TEMPLATES.get(
                (result_type, "default"),
                _DEFAULT_TEMPLATES[(ResultType.UNKNOWN, "default")],
            ),
        )

    @staticmethod
    def _extract_question_prefix(question: str) -> str:
        """
        Convert a question like '1月20日有多少配送订单？' into a statement prefix.

        Heuristic: remove question words at the end, add '有' or statement connector.
        """
        if not question:
            return ""

        # Strip trailing question marks and question particles
        cleaned = question.strip()
        cleaned = re.sub(r"[？?]+$", "", cleaned)
        cleaned = re.sub(r"(多少|几|什么|怎么|是否|吗)$", "", cleaned)
        cleaned = cleaned.strip()

        # If it ends with a noun-like phrase, append "有"
        if cleaned and not cleaned.endswith(("是", "有", "为", "达", "至")):
            return f"{cleaned}有"

        return cleaned

    def _polish_with_llm(self, draft: str, result: Any, question: str) -> str:
        """Send the draft answer to an LLM for natural-language polishing."""
        if self._cfg.llm_client is None:
            return draft

        prompt = (
            "请将以下数据分析结果改写为通顺、自然的简体中文回答，"
            "保持数据准确，语言简洁专业。只输出改写后的回答，不要解释。\n\n"
            f"原始问题：{question}\n"
            f"原始数据：{result}\n"
            f"草稿回答：{draft}\n"
            "改写后的回答："
        )
        try:
            polished = self._cfg.llm_client.generate(prompt=prompt, temperature=0.3)
            return polished.strip() or draft
        except Exception as exc:
            logger.warning("LLM polish failed: %s", exc)
            return draft


# ---------------------------------------------------------------------------
# Convenience formatter with built-in examples
# ---------------------------------------------------------------------------

class SimpleAnswerFormatter(AnswerFormatter):
    """
    Offline-only formatter (no LLM calls).  Fast and deterministic.

    Usage:
        formatter = SimpleAnswerFormatter()
        answer = formatter.format(330, "1月20日有多少配送订单？")
        # → "1月20日当天有330个配送订单被处理。"
    """

    def __init__(self):
        super().__init__(FormatterConfig(use_llm_polish=False))


class LLMAnswerFormatter(AnswerFormatter):
    """
    Formatter that uses an LLM for final polish — best fluency, slower.

    Usage:
        formatter = LLMAnswerFormatter(llm_client=my_llm)
        answer = formatter.format(330, "1月20日有多少配送订单？")
    """

    def __init__(self, llm_client: Any):
        super().__init__(
            FormatterConfig(use_llm_polish=True, llm_client=llm_client)
        )


# ---------------------------------------------------------------------------
# Quick format function (module-level convenience)
# ---------------------------------------------------------------------------

_default_formatter: Optional[SimpleAnswerFormatter] = None


def format_answer(result: Any, question: str = "") -> str:
    """
    Module-level convenience function.

    Examples:
        >>> format_answer(330, "1月20日有多少配送订单？")
        '1月20日当天有330单。'

        >>> format_answer((367, 0.1539), "订单增长了多少？")
        '增加367单，增幅15.39%。'
    """
    global _default_formatter
    if _default_formatter is None:
        _default_formatter = SimpleAnswerFormatter()
    return _default_formatter.format(result, question)
