"""
ChatBI Agent Configuration Module
=================================
Centralized configuration constants for the supply chain ChatBI agent.
Used in: 绝配-港大AI赛 (Shanghai Juepei/HKU/SHS AI Competition) Question 1.

All paths, column mappings, LLM parameters, and execution settings are defined here
to ensure consistency across data_loader, schema_manager, and agent modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = PROJECT_ROOT  # same folder as this file

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
# Default: look for data files in ./data/ directory (relative to project root)
# Override with env var CHATBI_DATA_DIR or modify these paths
DATA_DIR: Path = Path(os.getenv("CHATBI_DATA_DIR", PROJECT_ROOT / "data"))

# January 2026 data
FILE_JAN: str = str(DATA_DIR / "测试客户下单量1月V2.xlsx")

# February-March 2026 data
FILE_FEB_MAR: str = str(DATA_DIR / "测试客户下单2-3月V2.xlsx")

# ---------------------------------------------------------------------------
# Excel sheet names (shared by both files)
# ---------------------------------------------------------------------------
SHEET_ORDERS: str = "订单表"          # Order basic info table
SHEET_DETAILS: str = "订单明细"        # Order line items table
SHEET_LOGISTICS: str = "物流信息"      # Order operation log table

SHEET_NAMES: List[str] = [SHEET_ORDERS, SHEET_DETAILS, SHEET_LOGISTICS]

# ---------------------------------------------------------------------------
# Column name normalization mappings
# ---------------------------------------------------------------------------
# Jan file uses '求和项:预计发货数量' and '求和项:预计发货数量-EA'
# Feb-Mar file uses '预计发货数量' and '预计发货数量EA'
# We normalize both to the Feb-Mar names (cleaner, no '求和项:' prefix).

# Mapping for Jan 订单明细 sheet columns -> unified names
JAN_DETAILS_COL_MAP: Dict[str, str] = {
    "求和项:预计发货数量": "预计发货数量",
    "求和项:预计发货数量-EA": "预计发货数量EA",
}

# Mapping for Feb-Mar 订单明细 sheet columns -> unified names (identity, already clean)
FEB_MAR_DETAILS_COL_MAP: Dict[str, str] = {
    # No changes needed for Feb-Mar
}

# ---------------------------------------------------------------------------
# Unified column names (post-normalization) for each table
# ---------------------------------------------------------------------------
# Orders table columns (same in both files)
ORDERS_COLUMNS: List[str] = [
    "订单单号",           # Unique order ID, format CO######
    "订单类型",           # '销售出库'(sales) or '其他出库'(other)
    "货主编码",           # Customer code — 'C01','C02','C03'
    "货主",               # Customer name — '客户1','客户2','客户3'
    "仓库",               # Warehouse e.g. '深圳5仓','武汉1仓'
    "收货门店",           # Receiving store name
    "省市区",             # Province-City-District, e.g. '广东省-深圳市-龙华区'
    "求和项:预计发货数量EA",  # Total estimated shipping quantity (EA units)
    "预计总箱数",          # Estimated total box count
    "创建人",             # Always '外部客户'
    "创建时间",           # Creation datetime, range 2026-01-01 to 2026-03-31
]

# Details table columns (after normalization)
DETAILS_COLUMNS: List[str] = [
    "订单单号",           # Order ID (foreign key to orders)
    "商品编码",           # Product code, format GS######
    "商品名称",           # Product name e.g. '精选超甜嫩青豆-BW'
    "温区",               # '冷冻'(frozen), '冷藏'(cold), '常温'(normal)
    "预计发货数量",        # Estimated shipping quantity
    "单位",               # Unit — '包','件','个','卷','捆','提','条','桶','瓶','箱','袋'
    "预计发货数量EA",      # Quantity in EA units
    "单位.1",             # Same as 单位 (duplicate unit column)
]

# Logistics table columns (same in both files)
LOGISTICS_COLUMNS: List[str] = [
    "订单号",             # Order ID — NOTE: column name is 订单号 NOT 订单单号!
    "操作时间",           # Operation datetime
    "操作记录",           # Operation record description
    "操作人",             # Operator name
]

# ---------------------------------------------------------------------------
# Data loading configuration
# ---------------------------------------------------------------------------
# pandas read_excel options
READ_EXCEL_KWARGS: Dict = {
    "engine": "openpyxl",
    "dtype": str,          # Read all as str first, then cast in loader
}

# Datetime columns to parse for each sheet
DATETIME_COLUMNS: Dict[str, List[str]] = {
    SHEET_ORDERS: ["创建时间"],
    SHEET_DETAILS: [],     # No datetime columns in details
    SHEET_LOGISTICS: ["操作时间"],
}

# ---------------------------------------------------------------------------
# Execution configuration
# ---------------------------------------------------------------------------
CODE_EXEC_TIMEOUT: int = 30          # seconds; max time for single code execution
MAX_RETRIES: int = 3                 # max self-correction retries
RETRY_DELAY_BASE: float = 1.0        # base delay (seconds) between retries

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
# Default to Ollama local model for competition (no cloud dependency)
# Override with env vars or CLI flags
LLM_MODEL: str = os.getenv("CHATBI_MODEL", "qwen2.5-coder:7b")
LLM_TEMPERATURE: float = 0.1         # low temp for deterministic code
LLM_TOP_P: float = 0.95              # nucleus sampling
LLM_MAX_TOKENS: int = 2048           # max output tokens for code + explanation

# Ollama configuration (local inference)
OLLAMA_ENABLED: bool = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", LLM_MODEL)  # preferred Ollama model tag

# Cloud API fallback (only used if Ollama is disabled)
CLOUD_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
CLOUD_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLOUD_MODEL: str = os.getenv("CLOUD_MODEL", "gpt-4o-mini")

# Fallback model (optional, for redundancy)
LLM_FALLBACK_MODEL: str = "gpt-3.5-turbo"

# ---------------------------------------------------------------------------
# Prompt engineering constants
# ---------------------------------------------------------------------------
MAX_SCHEMA_SAMPLE_VALUES: int = 2    # sample values per column in schema prompt
MAX_FEW_SHOT_EXAMPLES: int = 3       # number of few-shot Q->code examples

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("CHATBI_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_file_paths() -> List[str]:
    """Return ordered list of source Excel file paths (Jan first, then Feb-Mar)."""
    return [FILE_JAN, FILE_FEB_MAR]


def validate_paths() -> None:
    """Raise FileNotFoundError if any required source file is missing."""
    missing: List[str] = []
    for p in get_file_paths():
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError(
            f"Missing required data file(s): {missing}. "
            f"Please ensure both January and February-March Excel files are uploaded."
        )
