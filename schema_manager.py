"""
ChatBI Schema Manager Module
============================
Extracts, caches, and formats rich schema descriptions from the supply-chain
DataFrames for injection into LLM prompts.

Features:
    - Automatic schema extraction (column names, dtypes, nulls, unique counts)
    - Per-column business descriptions in Chinese
    - Representative sample values (2 per column by default)
    - Clean, LLM-friendly text formatting
    - Module-level cache (computed once, reused for all prompts)

Usage:
    >>> from schema_manager import get_combined_schema, get_table_schema
    >>> schema_text = get_combined_schema()
    >>> print(schema_text)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import MAX_SCHEMA_SAMPLE_VALUES, LOG_FORMAT, LOG_LEVEL
from data_loader import (
    get_customer_code_list,
    get_customer_list,
    get_data,
    get_date_range,
    get_order_type_list,
    get_product_list,
    get_store_list,
    get_temperature_zone_list,
    get_unit_list,
    get_warehouse_list,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-level business descriptions (Chinese)
# ---------------------------------------------------------------------------
# These descriptions explain the BUSINESS MEANING of each column, not just
# the technical dtype.  They are critical for LLM accuracy (+20-30%).

_ORDERS_COL_DESC: Dict[str, str] = {
    "订单单号": "订单唯一编号，格式为CO######，如CO036611274",
    "订单类型": "订单业务类型，'销售出库'表示销售发货，'其他出库'表示非销售类出库",
    "货主编码": "客户编码，如'C01'、'C02'、'C03'",
    "货主": "客户名称，如'客户1'、'客户2'、'客户3'",
    "仓库": "发货仓库名称，如'深圳5仓'、'武汉1仓'、'广州6仓'等",
    "收货门店": "收货门店名称，如'深圳新龙大厦店'",
    "省市区": "收货地址的省-市-区，格式'广东省-深圳市-龙华区'",
    "求和项:预计发货数量EA": "订单预计发货总数量(EA计量单位)",
    "预计总箱数": "订单预计发货总箱数",
    "创建人": "订单创建人，固定值为'外部客户'",
    "创建时间": "订单创建时间，范围2026-01-01至2026-03-31",
}

_DETAILS_COL_DESC: Dict[str, str] = {
    "订单单号": "订单唯一编号，与订单表的'订单单号'关联",
    "商品编码": "商品唯一编码，格式为GS######，如GS000012793",
    "商品名称": "商品名称，如'精选超甜嫩青豆-BW'、'升级版分切鸡腿肉粒品-BW'",
    "温区": "商品存储温区，'冷冻'表示冷冻品，'冷藏'表示冷藏品，'常温'表示常温品",
    "预计发货数量": "该商品的预计发货数量",
    "单位": "计量单位，如'包'、'件'、'桶'、'袋'、'提'、'瓶'、'捆'、'箱'、'条'、'卷'、'个'、'套'",
    "预计发货数量EA": "该商品的预计发货数量(EA计量单位)",
    "单位.1": "计量单位(重复列)，与'单位'含义相同",
}

_LOGISTICS_COL_DESC: Dict[str, str] = {
    "订单号": "订单唯一编号，与订单表的'订单单号'关联(注意列名不同：此处为'订单号')",
    "操作时间": "操作发生的时间",
    "操作记录": "操作记录描述，包含订单状态变更信息",
    "操作人": "执行操作的人员姓名",
}

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_schema_cache: Optional[str] = None
_table_schemas_cache: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _get_safe_samples(series: pd.Series, n: int = MAX_SCHEMA_SAMPLE_VALUES) -> List[str]:
    """Return up to *n* non-null, unique sample values from a Series.

    Strings are truncated to 60 chars to keep schema prompts concise.
    Values are converted to str for uniform representation.

    Args:
        series: pandas Series to sample from.
        n: Maximum number of samples to return.

    Returns:
        List of sample value strings.
    """
    # Drop NA, deduplicate, sample, convert to str
    cleaned = series.dropna().astype(str).str.strip()
    unique_vals = cleaned.drop_duplicates().head(n).tolist()
    # Truncate long strings
    truncated = [v[:60] + "..." if len(v) > 60 else v for v in unique_vals]
    return truncated


def _fmt_dtype(series: pd.Series) -> str:
    """Return a human-friendly dtype string for a pandas Series.

    Args:
        series: pandas Series.

    Returns:
        Short dtype descriptor, e.g. 'str', 'int', 'float', 'datetime', 'bool'.
    """
    dtype = series.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    return "str"


def _build_table_schema(
    df: pd.DataFrame,
    table_name: str,
    table_desc: str,
    col_desc_map: Dict[str, str],
) -> str:
    """Build a rich schema description string for a single table.

    The output format is optimized for LLM prompt injection:
        - Clean header with table name and description
        - One line per column with name, dtype, description, and 2 sample values
        - Footer with row count and key statistics

    Args:
        df: DataFrame to introspect.
        table_name: Human-readable table name (Chinese).
        table_desc: Brief description of the table's business purpose.
        col_desc_map: Mapping from column name -> Chinese business description.

    Returns:
        Formatted schema string.
    """
    lines: List[str] = []
    lines.append(f"{'='*70}")
    lines.append(f"【{table_name}】— {table_desc}")
    lines.append(f"{'='*70}")
    lines.append(f"总记录数: {len(df):,} 行 | 字段数: {len(df.columns)} 列")
    lines.append("")

    for col in df.columns:
        dtype_str = _fmt_dtype(df[col])
        desc = col_desc_map.get(col, "")
        samples = _get_safe_samples(df[col], n=MAX_SCHEMA_SAMPLE_VALUES)
        sample_str = ", ".join(f"'{s}'" for s in samples) if samples else "(无样本)"
        null_pct = df[col].isna().mean() * 100
        null_info = f"  空值率: {null_pct:.1f}%" if null_pct > 0 else ""

        lines.append(f"  • {col}")
        lines.append(f"    类型: {dtype_str}  |  说明: {desc}")
        lines.append(f"    样例值: {sample_str}{null_info}")

    lines.append("")
    return "\n".join(lines)


def _build_relationships_section() -> str:
    """Build the inter-table relationships section of the schema.

    Returns:
        Formatted relationship description string.
    """
    lines: List[str] = []
    lines.append(f"{'='*70}")
    lines.append("【表间关联关系】")
    lines.append(f"{'='*70}")
    lines.append(
        "  1. 订单表.订单单号  ==  订单明细.订单单号  (1对多)"
        "\n     一个订单包含多条商品明细"
    )
    lines.append(
        "  2. 订单表.订单单号  ==  物流信息.订单号  (1对多)"
        "\n     一个订单对应多条物流操作记录"
    )
    lines.append("")
    lines.append("【关键字段注意事项】")
    lines.append("  • 物流信息表的订单编号列名为'订单号'，不是'订单单号'")
    lines.append("  • 订单表中的'求和项:预计发货数量EA'是订单级别的预计发货总数量(EA)")
    lines.append("  • 订单明细中的'预计发货数量EA'是商品行级别的预计发货数量(EA)")
    lines.append("  • 创建时间范围: 2026-01-01 至 2026-03-31")
    lines.append("  • 统计订单数量时使用 订单单号 去重")
    lines.append("")
    return "\n".join(lines)


def _build_logistics_guide_section() -> str:
    """Build the logistics operation guide section.

    Key operation patterns in 操作记录 are documented here so the LLM
    knows how to interpret logistics status queries.

    Returns:
        Formatted guide string.
    """
    lines: List[str] = []
    lines.append(f"{'='*70}")
    lines.append("【物流操作记录关键词指南】")
    lines.append(f"{'='*70}")
    lines.append("  • '订单已创建,订单来源:客户对接,操作人:外部客户'  →  订单创建")
    lines.append("  • '订单已审核,开始进入下一个环节。操作人:XXX'   →  订单已审核/已处理")
    lines.append("  • '订单【COXXX】生成波次，生成时间:YYYY-MM-DD HH:MM:SS'  →  波次已生成")
    lines.append("  • '订单【COXXX】已复核，复核时间:YYYY-MM-DD HH:MM:SS'  →  已复核")
    lines.append("  • '订单【COXXX】已发运，发运时间:YYYY-MM-DD HH:MM:SS'  →  已发运")
    lines.append("  • 包含'已签收'或'签收'的记录                       →  已签收")
    lines.append("  • 包含'司机'的记录                                 →  司机分配/调度")
    lines.append("")
    lines.append("【常见统计口径】")
    lines.append("  • '已处理订单' = 操作记录中包含'订单已审核'的订单")
    lines.append("  • '已发运订单' = 操作记录中包含'已发运'的订单")
    lines.append("  • '已签收订单' = 操作记录中包含'已签收'或'签收'的订单")
    lines.append("  • '波次已生成' = 操作记录中包含'生成波次'的订单")
    lines.append("")
    return "\n".join(lines)


def _build_categoricals_section(
    df_orders: pd.DataFrame,
    df_details: pd.DataFrame,
) -> str:
    """Build a summary of key categorical domains.

    This gives the LLM a quick reference for valid filter values.

    Args:
        df_orders: Orders DataFrame.
        df_details: Details DataFrame.

    Returns:
        Formatted categorical summary string.
    """
    lines: List[str] = []
    lines.append(f"{'='*70}")
    lines.append("【关键分类维度枚举值】")
    lines.append(f"{'='*70}")

    # Customers
    customers = sorted(df_orders["货主"].dropna().unique())
    lines.append(f"  客户名称(货主): {', '.join(customers)}")

    # Customer codes
    cust_codes = sorted(df_orders["货主编码"].dropna().unique())
    lines.append(f"  客户编码(货主编码): {', '.join(cust_codes)}")

    # Order types
    order_types = sorted(df_orders["订单类型"].dropna().unique())
    lines.append(f"  订单类型: {', '.join(order_types)}")

    # Temp zones
    temp_zones = sorted(df_details["温区"].dropna().unique())
    lines.append(f"  温区: {', '.join(temp_zones)}")

    # Units
    units = sorted(df_details["单位"].dropna().unique())
    lines.append(f"  计量单位: {', '.join(units)}")

    # Warehouses (top 10)
    warehouses = sorted(df_orders["仓库"].dropna().unique())
    if len(warehouses) > 10:
        lines.append(f"  仓库(共{len(warehouses)}个): {', '.join(warehouses[:10])}, ...")
    else:
        lines.append(f"  仓库: {', '.join(warehouses)}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_table_schema(table: str) -> str:
    """Return the schema description for a single table.

    Args:
        table: One of 'orders', 'details', 'logistics'.

    Returns:
        Formatted schema string for the requested table.

    Raises:
        ValueError: If *table* is not a recognized table name.
    """
    df_orders, df_details, df_logistics = get_data()

    if table == "orders":
        return _build_table_schema(
            df_orders,
            table_name="订单表 (df_orders)",
            table_desc="订单基本信息表，每行代表一个订单",
            col_desc_map=_ORDERS_COL_DESC,
        )
    elif table == "details":
        return _build_table_schema(
            df_details,
            table_name="订单明细表 (df_details)",
            table_desc="订单商品明细表，每行代表一个订单中的一个商品行项目",
            col_desc_map=_DETAILS_COL_DESC,
        )
    elif table == "logistics":
        return _build_table_schema(
            df_logistics,
            table_name="物流信息表 (df_logistics)",
            table_desc="订单物流操作记录表，每行代表一个订单的一次操作记录",
            col_desc_map=_LOGISTICS_COL_DESC,
        )
    else:
        raise ValueError(
            f"Unknown table '{table}'. Choose from: 'orders', 'details', 'logistics'."
        )


def get_combined_schema(
    include_relationships: bool = True,
    include_logistics_guide: bool = True,
    include_categoricals: bool = True,
) -> str:
    """Return the full combined schema for all three tables.

    This is the main entry point for LLM prompt construction.  The output
    includes table schemas, relationships, logistics keywords, and categorical
    enums — everything the model needs to generate accurate pandas code.

    The result is cached after first computation; subsequent calls return the
    cached string unless the cache is cleared.

    Args:
        include_relationships: Include the table relationships section.
        include_logistics_guide: Include the logistics operation keywords guide.
        include_categoricals: Include the categorical domains summary.

    Returns:
        Complete schema string ready for prompt injection.
    """
    global _schema_cache

    if _schema_cache is not None:
        logger.debug("Returning cached combined schema.")
        return _schema_cache

    logger.info("Building combined schema ...")

    parts: List[str] = []
    parts.append("=" * 70)
    parts.append("数据仓库Schema说明")
    parts.append("=" * 70)
    parts.append("")

    # Table schemas
    parts.append(get_table_schema("orders"))
    parts.append(get_table_schema("details"))
    parts.append(get_table_schema("logistics"))

    # Relationships
    if include_relationships:
        parts.append(_build_relationships_section())

    # Logistics guide
    if include_logistics_guide:
        parts.append(_build_logistics_guide_section())

    # Categoricals
    if include_categoricals:
        df_orders, df_details, _ = get_data()
        parts.append(_build_categoricals_section(df_orders, df_details))

    # Usage hint for the LLM
    parts.append("" + "=" * 70)
    parts.append("【使用提示】")
    parts.append("=" * 70)
    parts.append("  • DataFrame变量名: df_orders(订单表), df_details(订单明细), df_logistics(物流信息)")
    parts.append("  • 所有金额/数量字段使用float64类型，可能包含NaN")
    parts.append("  • 字符串匹配建议使用 .str.contains() 或 == ，注意大小写")
    parts.append("  • 日期筛选可使用 pd.to_datetime() 或 .dt 访问器")
    parts.append("  • 多表关联使用 merge(on='订单单号', how='inner')")
    parts.append("  • 注意: 物流信息表的订单号列名为'订单号'，merge时需注意")
    parts.append("")

    _schema_cache = "\n".join(parts)
    logger.info("Combined schema built and cached (%d chars).", len(_schema_cache))
    return _schema_cache


def clear_schema_cache() -> None:
    """Clear the module-level schema cache.

    Call this after data reloads or when schema needs recomputation.
    """
    global _schema_cache, _table_schemas_cache
    _schema_cache = None
    _table_schemas_cache = None
    logger.info("Schema cache cleared.")


def get_schema_summary_dict() -> Dict[str, Dict]:
    """Return schema as a structured dictionary for programmatic use.

    This is useful if you need to build custom prompts or introspect
    schema metadata without parsing the text representation.

    Returns:
        Dictionary mapping table name -> {
            "columns": [
                {
                    "name": str,
                    "dtype": str,
                    "description": str,
                    "samples": List[str],
                    "null_rate": float,
                },
                ...
            ],
            "row_count": int,
        }
    """
    df_orders, df_details, df_logistics = get_data()
    result: Dict[str, Dict] = {}

    for df, name, desc_map in [
        (df_orders, "orders", _ORDERS_COL_DESC),
        (df_details, "details", _DETAILS_COL_DESC),
        (df_logistics, "logistics", _LOGISTICS_COL_DESC),
    ]:
        cols = []
        for col in df.columns:
            cols.append(
                {
                    "name": col,
                    "dtype": _fmt_dtype(df[col]),
                    "description": desc_map.get(col, ""),
                    "samples": _get_safe_samples(df[col], n=MAX_SCHEMA_SAMPLE_VALUES),
                    "null_rate": round(float(df[col].isna().mean()), 4),
                }
            )
        result[name] = {"columns": cols, "row_count": len(df)}

    return result
