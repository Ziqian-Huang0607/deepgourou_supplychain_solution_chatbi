"""
ChatBI Data Loader Module
=========================
Handles loading, normalization, and caching of supply chain order data
from two Excel files (January and February-March 2026).

Features:
    - Loads both Excel files and all 3 sheets
    - Normalizes column-name differences between Jan and Feb-Mar files
    - Concatenates data into unified DataFrames
    - Implements singleton caching (data loaded once, reused thereafter)
    - Provides utility queries for schema discovery

Usage:
    >>> from data_loader import get_data, get_date_range, get_customer_list
    >>> df_orders, df_details, df_logistics = get_data()
    >>> print(df_orders.shape, df_details.shape, df_logistics.shape)
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import (
    DATETIME_COLUMNS,
    FEB_MAR_DETAILS_COL_MAP,
    FILE_FEB_MAR,
    FILE_JAN,
    JAN_DETAILS_COL_MAP,
    LOG_FORMAT,
    LOG_LEVEL,
    READ_EXCEL_KWARGS,
    SHEET_DETAILS,
    SHEET_LOGISTICS,
    SHEET_ORDERS,
    get_file_paths,
    validate_paths,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cache containers (module-level singletons)
# ---------------------------------------------------------------------------
_data_cache: Optional[Dict[str, pd.DataFrame]] = None
_load_error: Optional[Exception] = None


# ---------------------------------------------------------------------------
# Core loading helpers
# ---------------------------------------------------------------------------

def _read_sheet(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Read a single sheet from an Excel file with standardized options.

    Args:
        file_path: Absolute path to the Excel file.
        sheet_name: Name of the sheet to read.

    Returns:
        Raw DataFrame read from the sheet.

    Raises:
        ValueError: If the sheet is empty or cannot be read.
    """
    logger.info("Reading sheet '%s' from '%s'", sheet_name, file_path)
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            **READ_EXCEL_KWARGS,
        )
    except Exception as exc:
        raise ValueError(f"Failed to read sheet '{sheet_name}' from '{file_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Sheet '{sheet_name}' in '{file_path}' is empty.")

    logger.info("  -> Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def _parse_datetimes(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Convert known datetime columns from string to datetime64[ns].

    Args:
        df: DataFrame with string-typed datetime columns.
        sheet_name: Sheet name used to look up which columns to parse.

    Returns:
        DataFrame with parsed datetime columns (in-place modifications on a copy).
    """
    cols = DATETIME_COLUMNS.get(sheet_name, [])
    if not cols:
        return df

    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            null_count = df[col].isna().sum()
            if null_count:
                logger.warning(
                    "  -> Column '%s' has %d unparsable datetime values", col, null_count
                )
    return df


def _normalize_jan_details(df: pd.DataFrame) -> pd.DataFrame:
    """Rename January-specific columns in 订单明细 to unified names.

    Jan uses '求和项:预计发货数量' and '求和项:预计发货数量-EA'.
    We rename them to '预计发货数量' and '预计发货数量EA' to match Feb-Mar.

    Also reorders columns to match the unified schema:
        订单单号, 商品编码, 商品名称, 温区, 预计发货数量, 单位,
        预计发货数量EA, 单位.1

    Args:
        df: Raw January 订单明细 DataFrame.

    Returns:
        DataFrame with unified column names and order.
    """
    df = df.rename(columns=JAN_DETAILS_COL_MAP)

    # Reorder to match unified schema: put 预计发货数量 before 单位
    # Jan original order: 订单单号, 商品编码, 商品名称, 温区, 单位, 求和项:预计发货数量, 单位.1, 求和项:预计发货数量-EA
    # After rename:       订单单号, 商品编码, 商品名称, 温区, 单位, 预计发货数量, 单位.1, 预计发货数量EA
    # Desired order:      订单单号, 商品编码, 商品名称, 温区, 预计发货数量, 单位, 预计发货数量EA, 单位.1
    desired_order = [
        "订单单号", "商品编码", "商品名称", "温区",
        "预计发货数量", "单位", "预计发货数量EA", "单位.1",
    ]
    missing = [c for c in desired_order if c not in df.columns]
    if missing:
        raise ValueError(f"Normalized Jan details missing expected columns: {missing}")
    return df[desired_order]


def _normalize_feb_mar_details(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Feb-Mar 订单明细 column order matches unified schema.

    Feb-Mar original order: 订单单号, 商品编码, 商品名称, 温区, 预计发货数量, 单位, 预计发货数量EA, 单位.1
    This already matches the desired order, but we enforce it explicitly.

    Args:
        df: Raw Feb-Mar 订单明细 DataFrame.

    Returns:
        DataFrame with unified column order.
    """
    desired_order = [
        "订单单号", "商品编码", "商品名称", "温区",
        "预计发货数量", "单位", "预计发货数量EA", "单位.1",
    ]
    missing = [c for c in desired_order if c not in df.columns]
    if missing:
        raise ValueError(f"Feb-Mar details missing expected columns: {missing}")
    return df[desired_order]


def _load_and_normalize() -> Dict[str, pd.DataFrame]:
    """Load both Excel files, normalize columns, concatenate, and cache.

    Returns:
        Dictionary with keys 'df_orders', 'df_details', 'df_logistics'.

    Raises:
        FileNotFoundError: If source files are missing.
        ValueError: If sheets are empty or columns are unexpected.
    """
    validate_paths()

    # ------------------------------------------------------------------
    # 1. Load raw sheets
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading supply chain data")
    logger.info("=" * 60)

    jan_orders = _read_sheet(FILE_JAN, SHEET_ORDERS)
    jan_details = _read_sheet(FILE_JAN, SHEET_DETAILS)
    jan_logistics = _read_sheet(FILE_JAN, SHEET_LOGISTICS)

    feb_orders = _read_sheet(FILE_FEB_MAR, SHEET_ORDERS)
    feb_details = _read_sheet(FILE_FEB_MAR, SHEET_DETAILS)
    feb_logistics = _read_sheet(FILE_FEB_MAR, SHEET_LOGISTICS)

    # ------------------------------------------------------------------
    # 2. Normalize column names for 订单明细 (details)
    # ------------------------------------------------------------------
    logger.info("Normalizing column names for 订单明细 ...")
    jan_details = _normalize_jan_details(jan_details)
    feb_details = _normalize_feb_mar_details(feb_details)

    # ------------------------------------------------------------------
    # 3. Parse datetime columns
    # ------------------------------------------------------------------
    logger.info("Parsing datetime columns ...")
    jan_orders = _parse_datetimes(jan_orders, SHEET_ORDERS)
    feb_orders = _parse_datetimes(feb_orders, SHEET_ORDERS)
    jan_logistics = _parse_datetimes(jan_logistics, SHEET_LOGISTICS)
    feb_logistics = _parse_datetimes(feb_logistics, SHEET_LOGISTICS)

    # ------------------------------------------------------------------
    # 4. Concatenate Jan + Feb-Mar
    # ------------------------------------------------------------------
    logger.info("Concatenating January + February-March data ...")
    df_orders = pd.concat([jan_orders, feb_orders], ignore_index=True)
    df_details = pd.concat([jan_details, feb_details], ignore_index=True)
    df_logistics = pd.concat([jan_logistics, feb_logistics], ignore_index=True)

    logger.info(
        "Final shapes -> orders: %s, details: %s, logistics: %s",
        df_orders.shape,
        df_details.shape,
        df_logistics.shape,
    )

    # ------------------------------------------------------------------
    # 5. Post-load type casting for numeric columns
    # ------------------------------------------------------------------
    df_orders = _cast_orders_types(df_orders)
    df_details = _cast_details_types(df_details)

    # Verify key columns exist
    _validate_final_columns(df_orders, df_details, df_logistics)

    return {
        "df_orders": df_orders,
        "df_details": df_details,
        "df_logistics": df_logistics,
    }


def _cast_orders_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns in orders table from str to appropriate types.

    Columns cast:
        - 求和项:预计发货数量EA -> float64
        - 预计总箱数 -> float64

    Args:
        df: Concatenated orders DataFrame with string-typed numeric columns.

    Returns:
        DataFrame with numeric columns properly typed.
    """
    df = df.copy()
    numeric_cols = ["求和项:预计发货数量EA", "预计总箱数"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _cast_details_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns in details table from str to appropriate types.

    Columns cast:
        - 预计发货数量 -> float64
        - 预计发货数量EA -> float64

    Args:
        df: Concatenated details DataFrame with string-typed numeric columns.

    Returns:
        DataFrame with numeric columns properly typed.
    """
    df = df.copy()
    numeric_cols = ["预计发货数量", "预计发货数量EA"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _validate_final_columns(
    df_orders: pd.DataFrame,
    df_details: pd.DataFrame,
    df_logistics: pd.DataFrame,
) -> None:
    """Validate that all expected columns exist after loading and normalization.

    Args:
        df_orders: Final orders DataFrame.
        df_details: Final details DataFrame.
        df_logistics: Final logistics DataFrame.

    Raises:
        ValueError: If any expected column is missing.
    """
    expected_orders = [
        "订单单号", "订单类型", "货主编码", "货主", "仓库",
        "收货门店", "省市区", "求和项:预计发货数量EA", "预计总箱数",
        "创建人", "创建时间",
    ]
    expected_details = [
        "订单单号", "商品编码", "商品名称", "温区", "预计发货数量",
        "单位", "预计发货数量EA", "单位.1",
    ]
    expected_logistics = ["订单号", "操作时间", "操作记录", "操作人"]

    for df, name, expected in [
        (df_orders, "df_orders", expected_orders),
        (df_details, "df_details", expected_details),
        (df_logistics, "df_logistics", expected_logistics),
    ]:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name} is missing expected columns after load: {missing}. "
                f"Actual columns: {list(df.columns)}"
            )

    logger.info("Column validation passed for all 3 tables.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_data(force_reload: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the three unified DataFrames (cached after first call).

    This is the primary entry point for accessing data.  It implements a
    module-level singleton cache so that the Excel files are only read once
    per process lifetime.

    Args:
        force_reload: If True, discard the cache and reload from disk.

    Returns:
        Tuple of (df_orders, df_details, df_logistics).

    Raises:
        FileNotFoundError: If source Excel files are missing.
        ValueError: If data normalization or validation fails.
    """
    global _data_cache, _load_error

    if force_reload or _data_cache is None:
        logger.info("Loading data from source Excel files ...")
        try:
            _data_cache = _load_and_normalize()
            _load_error = None
            logger.info("Data loaded and cached successfully.")
        except Exception as exc:
            _load_error = exc
            _data_cache = None
            logger.error("Failed to load data: %s", exc)
            raise

    if _load_error and _data_cache is None:
        raise _load_error

    return (
        _data_cache["df_orders"],
        _data_cache["df_details"],
        _data_cache["df_logistics"],
    )


def clear_cache() -> None:
    """Clear the module-level data cache.

    Subsequent calls to :func:`get_data` will reload from disk.
    """
    global _data_cache, _load_error
    _data_cache = None
    _load_error = None
    logger.info("Data cache cleared.")


# ---------------------------------------------------------------------------
# Utility query functions (useful for schema_manager and agent prompts)
# ---------------------------------------------------------------------------

def get_date_range(df_orders: Optional[pd.DataFrame] = None) -> Tuple[datetime, datetime]:
    """Return the minimum and maximum creation dates in the orders table.

    Args:
        df_orders: Orders DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        (min_date, max_date) as datetime objects.
    """
    if df_orders is None:
        df_orders, _, _ = get_data()
    min_date = df_orders["创建时间"].min()
    max_date = df_orders["创建时间"].max()
    return min_date, max_date


def get_customer_list(df_orders: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted list of unique customer names (货主).

    Args:
        df_orders: Orders DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Sorted list of customer name strings.
    """
    if df_orders is None:
        df_orders, _, _ = get_data()
    return sorted(df_orders["货主"].dropna().unique().tolist())


def get_customer_code_list(df_orders: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted list of unique customer codes (货主编码).

    Args:
        df_orders: Orders DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Sorted list of customer code strings.
    """
    if df_orders is None:
        df_orders, _, _ = get_data()
    return sorted(df_orders["货主编码"].dropna().unique().tolist())


def get_product_list(df_details: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted list of unique product names (商品名称).

    Args:
        df_details: Details DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Sorted list of product name strings.
    """
    if df_details is None:
        _, df_details, _ = get_data()
    return sorted(df_details["商品名称"].dropna().unique().tolist())


def get_warehouse_list(df_orders: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted list of unique warehouse names (仓库).

    Args:
        df_orders: Orders DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Sorted list of warehouse name strings.
    """
    if df_orders is None:
        df_orders, _, _ = get_data()
    return sorted(df_orders["仓库"].dropna().unique().tolist())


def get_store_list(df_orders: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted list of unique receiving store names (收货门店).

    Args:
        df_orders: Orders DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Sorted list of store name strings.
    """
    if df_orders is None:
        df_orders, _, _ = get_data()
    return sorted(df_orders["收货门店"].dropna().unique().tolist())


def get_temperature_zone_list(df_details: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted list of unique temperature zones (温区).

    Args:
        df_details: Details DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Sorted list of temperature zone strings.
    """
    if df_details is None:
        _, df_details, _ = get_data()
    return sorted(df_details["温区"].dropna().unique().tolist())


def get_order_type_list(df_orders: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted list of unique order types (订单类型).

    Args:
        df_orders: Orders DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Sorted list of order type strings.
    """
    if df_orders is None:
        df_orders, _, _ = get_data()
    return sorted(df_orders["订单类型"].dropna().unique().tolist())


def get_unit_list(df_details: Optional[pd.DataFrame] = None) -> List[str]:
    """Return sorted list of unique units (单位).

    Args:
        df_details: Details DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Sorted list of unit strings.
    """
    if df_details is None:
        _, df_details, _ = get_data()
    return sorted(df_details["单位"].dropna().unique().tolist())


def get_order_count(df_orders: Optional[pd.DataFrame] = None) -> int:
    """Return total number of unique orders.

    Args:
        df_orders: Orders DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Integer count of unique order IDs.
    """
    if df_orders is None:
        df_orders, _, _ = get_data()
    return int(df_orders["订单单号"].nunique())


def get_details_row_count(df_details: Optional[pd.DataFrame] = None) -> int:
    """Return total number of detail line items.

    Args:
        df_details: Details DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Integer count of detail rows.
    """
    if df_details is None:
        _, df_details, _ = get_data()
    return len(df_details)


def get_logistics_row_count(df_logistics: Optional[pd.DataFrame] = None) -> int:
    """Return total number of logistics operation records.

    Args:
        df_logistics: Logistics DataFrame. If None, calls :func:`get_data` internally.

    Returns:
        Integer count of logistics rows.
    """
    if df_logistics is None:
        _, _, df_logistics = get_data()
    return len(df_logistics)


# ---------------------------------------------------------------------------
# Module-level convenience: eager load on first import (optional)
# ---------------------------------------------------------------------------
# Lazy loading is preferred; data is loaded on first call to get_data().
# This avoids slow imports and import-side effects.
