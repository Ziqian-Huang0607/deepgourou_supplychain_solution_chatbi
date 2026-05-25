"""
ChatBI Query Engine - PAL-based pandas code generation for Chinese NL queries.

Architecture:
    Prompt (schema + few-shots + question) -> OpenAI-compatible API
    -> parse Python code -> CodeExecutor -> result

Design decisions (from deep research):
    * Model: Qwen2.5-Coder-7B-Instruct via vLLM / OpenAI-compatible endpoint
    * Temperature=0.1, top_p=0.95 for deterministic code generation
    * Self-correction: feed execution error back to LLM, retry <= 2
    * Single-pass direct generation (NOT ReAct, NOT LangChain agent)
    * Extract `result` variable from executed code
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, APITimeoutError

from code_executor import CodeExecutor, ExecutionResult

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
# Ollama is the default (local, no cloud dependency, better for competition)
DEFAULT_BASE_URL: str = os.getenv("OPENAI_BASE_URL",
    os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1")
DEFAULT_API_KEY: str = os.getenv("OPENAI_API_KEY", "ollama")
DEFAULT_MODEL: str = os.getenv("CHATBI_MODEL",
    os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"))
MAX_RETRIES: int = 1  # 1 retry = max 2 total attempts (speed vs accuracy tradeoff)
REQUEST_TIMEOUT: float = 60.0

SYSTEM_PROMPT: str = (
    "You are a pandas data analysis expert for supply-chain order data. "
    "Given the user's Chinese question and the pre-loaded dataframes, "
    "generate clean, correct, executable pandas code. "
    "The code must store the final answer in a variable named `result`. "
    "Add comments in Chinese to explain key steps. "
    "Do NOT use print(), external files, or undefined variables.\n\n"
    "## Date Interpretation Rules (MUST FOLLOW)\n"
    "- 'X月Y日' = specific date, use pd.Timestamp('2026-X-Y')\n"
    "- '前N天/前七天' = N days BEFORE a reference date, EXCLUSIVE of the reference date. "
    "Example: if period is '1月20到27日', then '前七天' = pd.Timestamp('2026-01-13') to pd.Timestamp('2026-01-19')\n"
    "- '后N天' = N days AFTER a reference date, EXCLUSIVE\n"
    "- ALWAYS create explicit date variables with comments showing actual calendar dates\n"
    "- Use .dt.date for date-level comparison\n\n"
    "## Multi-Part Question Rules (MUST FOLLOW)\n"
    "If a question has multiple parts (indicated by '其中', '以及', commas, or multiple questions):\n"
    "1. Identify EACH sub-question in code comments BEFORE coding\n"
    "2. Solve sub-questions sequentially with clear variable names\n"
    "3. Combine ALL answers into the final `result` (use tuple or dict)\n"
    "4. NEVER stop after answering only the first part\n\n"
    "## Table Join Rules\n"
    "- df_orders['订单单号'] links to df_details['订单单号']\n"
    "- df_orders['订单单号'] links to df_logistics['订单号'] (NOTE: column name is different!)\n"
    "- Always use .nunique() when counting orders (to avoid counting duplicate rows)"
)

# ---------------------------------------------------------------------------
# Schema description (REAL column names from actual data)
# ---------------------------------------------------------------------------
SCHEMA_DESCRIPTION: str = """\
## 数据表说明

以下3个DataFrame已经预加载到执行环境中，可直接使用：

### 1. df_orders - 订单表 (30,815行 x 11列)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| 订单单号 | str | 唯一订单编号，格式CO######，如CO036611274 |
| 订单类型 | str | '销售出库'(销售发货) 或 '其他出库'(非销售出库) |
| 货主编码 | str | 客户编码：C01, C02, C03 |
| 货主 | str | 客户名称：客户1, 客户2, 客户3 |
| 仓库 | str | 发货仓库，如'深圳5仓','武汉1仓','广州6仓' |
| 收货门店 | str | 收货门店名称，如'深圳新龙大厦店' |
| 省市区 | str | 收货地址，格式'广东省-深圳市-龙华区' |
| 求和项:预计发货数量EA | float | 订单预计发货总数量(EA计量单位) |
| 预计总箱数 | float | 订单预计发货总箱数 |
| 创建人 | str | 固定值'外部客户' |
| 创建时间 | datetime | 订单创建时间，范围2026-01-01至2026-03-31 |

### 2. df_details - 订单明细表 (437,204行 x 8列)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| 订单单号 | str | 关联df_orders的'订单单号' |
| 商品编码 | str | 商品唯一编码，格式GS###### |
| 商品名称 | str | 商品名称，如'精选超甜嫩青豆-BW','窖香脆卜装-BW' |
| 温区 | str | '冷冻','冷藏','常温' |
| 预计发货数量 | float | 该商品预计发货数量 |
| 单位 | str | 计量单位：包,件,个,卷,捆,提,条,桶,瓶,箱,袋 |
| 预计发货数量EA | float | 该商品预计发货数量(EA计量单位) |
| 单位.1 | str | 计量单位(重复列) |

**注意**: df_orders与df_details通过'订单单号'关联，一对多关系。

### 3. df_logistics - 物流信息表 (385,817行 x 4列)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| 订单号 | str | 关联df_orders的'订单单号'(注意列名不同！此处为'订单号') |
| 操作时间 | datetime | 物流操作时间 |
| 操作记录 | str | 操作描述，如'订单已审核...','订单已创建...' |
| 操作人 | str | 操作人员姓名 |

**注意**: df_orders['订单单号'] == df_logistics['订单号'] 关联(列名不同！)

### 关键操作记录关键词
- '订单已创建,订单来源:客户对接...' -> 订单创建
- '订单已审核,开始进入下一个环节...' -> 订单已审核/已处理
- '订单【COXXX】生成波次...' -> 波次已生成
- '订单【COXXX】已复核...' -> 已复核
- 包含'签收'的记录 -> 已签收

### 常见统计口径
- '已处理订单' = 操作记录中包含'订单已审核'的订单
- '已发运订单' = 操作记录中包含'已发运'的订单
- '已签收订单' = 操作记录中包含'已签收'或'签收'的订单

### 全局常量
- 客户：客户1, 客户2, 客户3
- 日期范围：2026-01-01 至 2026-03-31
- 商品：288种不同商品
- 所有时间字段均为 pandas datetime64 类型

### 代码规范
1. 直接使用变量 df_orders, df_details, df_logistics (已预加载)
2. 最终答案必须保存在变量 `result` 中
3. 用中文注释说明关键步骤
4. 不要写 print()，不要读写文件，不要使用未定义的变量
5. 日期筛选时如果要比较"某天的日期"，使用 .dt.date 进行日期级比较
6. 关联表时注意列名差异：df_orders用'订单单号'，df_logistics用'订单号'
7. 统计订单数量时按订单单号去重
"""

# ---------------------------------------------------------------------------
# Few-shot examples (REAL domain-specific examples with ACTUAL column names)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "question": "1月20日当天有多少个配送订单被处理？",
        "code": textwrap.dedent('''\
            # 从物流信息中筛选1月20日的审核记录
            target_date = pd.Timestamp('2026-01-20').date()
            mask_date = df_logistics['操作时间'].dt.date == target_date
            mask_audit = df_logistics['操作记录'].str.contains('订单已审核', na=False)
            result = df_logistics[mask_date & mask_audit]['订单号'].nunique()
        '''),
    },
    {
        "question": "1月20到27日的处理订单数量相比前七天变化是多少？",
        "code": textwrap.dedent('''\
            # Step 1: Define current period
            period_start = pd.Timestamp('2026-01-20').date()  # 1月20日
            period_end = pd.Timestamp('2026-01-27').date()    # 1月27日

            # Step 2: Calculate "前七天" = 7 days BEFORE period_start, EXCLUSIVE
            prev_start = pd.Timestamp('2026-01-13').date()    # 1月13日
            prev_end = pd.Timestamp('2026-01-19').date()      # 1月19日

            # Step 3: Count orders in current period
            audit = df_logistics[df_logistics['操作记录'].str.contains('订单已审核', na=False)]
            curr = audit[(audit['操作时间'].dt.date >= period_start) & (audit['操作时间'].dt.date <= period_end)]
            curr_cnt = curr['订单号'].nunique()

            # Step 4: Count orders in previous period
            prev = audit[(audit['操作时间'].dt.date >= prev_start) & (audit['操作时间'].dt.date <= prev_end)]
            prev_cnt = prev['订单号'].nunique()

            # Step 5: Calculate change
            change = curr_cnt - prev_cnt
            pct = round((change / prev_cnt) * 100, 2) if prev_cnt > 0 else 0
            result = f"增加{change}单，增幅{pct}%。"
        '''),
    },
    {
        "question": "1月前7天哪个客户下的订单最多，其中什么商品数量最多，是多少？",
        "code": textwrap.dedent('''\
            # Sub-question 1: Which customer ordered the most in first 7 days of Jan?
            jan7 = df_orders[df_orders['创建时间'].dt.date <= pd.Timestamp('2026-01-07').date()]
            cust_orders = jan7.groupby('货主')['订单单号'].nunique().sort_values(ascending=False)
            top_customer = cust_orders.index[0]
            top_orders = int(cust_orders.iloc[0])

            # Sub-question 2: For that customer, which product had highest quantity?
            cust_order_ids = jan7[jan7['货主'] == top_customer]['订单单号'].unique()
            cust_details = df_details[df_details['订单单号'].isin(cust_order_ids)]
            product_qty = cust_details.groupby('商品名称')['预计发货数量EA'].sum().sort_values(ascending=False)
            top_product = product_qty.index[0]
            top_qty = int(product_qty.iloc[0])

            # Combine all answers
            result = f"{top_customer}的订单最多（{top_orders}单）；其中{top_product}数量最多（{top_qty}EA）。"
        '''),
    },
    {
        "question": "哪个仓库处理的订单数量最多？",
        "code": textwrap.dedent('''\
            # 按仓库分组统计唯一订单数量
            warehouse_counts = df_orders.groupby('仓库')['订单单号'].nunique()
            result = warehouse_counts.idxmax()
        '''),
    },
    {
        "question": "2月份每个客户的订单总量分别是多少？",
        "code": textwrap.dedent('''\
            # 筛选2月份订单并按客户分组统计
            feb = df_orders[df_orders['创建时间'].dt.month == 2]
            result = feb.groupby('货主')['求和项:预计发货数量EA'].sum().to_dict()
        '''),
    },
]


def _build_few_shot_text(examples: List[Dict[str, str]]) -> str:
    """Serialize few-shot examples into prompt text."""
    lines: List[str] = []
    for idx, ex in enumerate(examples, 1):
        lines.append(f"### Example {idx}")
        lines.append(f"Q: {ex['question']}")
        lines.append("Code:")
        lines.append("```python")
        lines.append(ex["code"].strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Assembles the full conversation prompt for the code-generation LLM."""

    def __init__(
        self,
        schema: str = SCHEMA_DESCRIPTION,
        few_shots: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.schema = schema
        self.few_shots = few_shots or FEW_SHOT_EXAMPLES
        self.system_prompt = system_prompt

    def build(self, user_question: str) -> List[Dict[str, str]]:
        """Build message list for the chat completion API."""
        user_content = textwrap.dedent(f"""\
            {self.schema}

            {_build_few_shot_text(self.few_shots)}
            ### 用户问题
            Q: {user_question}
            Code:
            ```python
            # 请在此处编写解答代码，最终结果保存到变量 result 中
        """)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_retry_prompt(
        self,
        original_question: str,
        previous_code: str,
        error_message: str,
    ) -> List[Dict[str, str]]:
        """Build a retry prompt with error context for self-correction."""
        user_content = textwrap.dedent(f"""\
            {self.schema}

            ### 用户问题
            Q: {original_question}

            ### 之前生成的代码（执行出错）
            ```python
            {previous_code}
            ```

            ### 错误信息
            {error_message}

            请修正上述代码中的错误，重新生成正确的代码。
            注意：DataFrame变量名为 df_orders, df_details, df_logistics
            最终结果必须保存在变量 `result` 中，并用中文注释说明关键步骤。
            Code:
            ```python
            # 修正后的代码
        """)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

class CodeParser:
    """Extract Python code blocks from LLM markdown responses."""

    CODE_BLOCK_RE = re.compile(
        r"```(?:python)?\s*\n(.*?)\n```",
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def parse(cls, raw_text: str) -> str:
        """Extract the first Python code block from the LLM response."""
        if not raw_text or not raw_text.strip():
            raise ValueError("Empty response from LLM.")

        matches = cls.CODE_BLOCK_RE.findall(raw_text)
        if matches:
            code = matches[0].strip()
            logger.debug("Extracted code block (%d chars).", len(code))
            return code

        # Fallback: if the entire response looks like code
        stripped = raw_text.strip()
        if stripped and ("import " in stripped or "df_" in stripped or "result" in stripped):
            logger.warning("No markdown fences found; treating entire response as code.")
            return stripped

        raise ValueError(f"No Python code block found in LLM response:\n{raw_text[:500]}")


# ---------------------------------------------------------------------------
# Query Engine
# ---------------------------------------------------------------------------

class ChatBIQueryEngine:
    """End-to-end PAL query engine: Chinese NL -> pandas code -> safe execution -> answer."""

    def __init__(
        self,
        df_orders,
        df_details,
        df_logistics,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        max_retries: int = MAX_RETRIES,
        request_timeout: float = REQUEST_TIMEOUT,
        executor: Optional[CodeExecutor] = None,
    ) -> None:
        self.df_orders = df_orders
        self.df_details = df_details
        self.df_logistics = df_logistics
        self.base_url = base_url or DEFAULT_BASE_URL
        self.api_key = api_key or DEFAULT_API_KEY
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        self._client: OpenAI = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout,
        )
        self._prompt_builder = PromptBuilder()
        self._executor = executor or CodeExecutor(
            df_orders=df_orders,
            df_details=df_details,
            df_logistics=df_logistics,
        )

        logger.info(
            "ChatBIQueryEngine initialized | model=%s | temp=%.2f | retries=%d",
            self.model, self.temperature, self.max_retries,
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Answer a Chinese natural-language question about the supply-chain data.

        Returns dict with: success, result, error, code, retries
        """
        if not question or not question.strip():
            return {
                "success": False, "result": None, "error": "Empty question.",
                "code": "", "retries": 0,
            }

        question = question.strip()
        logger.info("Processing question: %s", question)

        # Phase 1 - initial generation
        messages = self._prompt_builder.build(question)
        code = self._call_llm_and_parse(messages)
        execution_result = self._executor.execute(code)
        retries = 0

        # Phase 2 - self-correction loop
        while not execution_result.success and retries < self.max_retries:
            retries += 1
            sanitized_error = self._sanitize_error(execution_result.error)
            logger.warning(
                "Execution failed (attempt %d/%d): %s",
                retries, self.max_retries + 1, sanitized_error[:200],
            )

            retry_messages = self._prompt_builder.build_retry_prompt(
                original_question=question,
                previous_code=code,
                error_message=sanitized_error,
            )
            try:
                code = self._call_llm_and_parse(retry_messages)
            except Exception as exc:
                logger.error("Retry LLM call failed: %s", exc)
                break

            execution_result = self._executor.execute(code)

        # Phase 3 - assemble final response
        if execution_result.success:
            return {
                "success": True,
                "result": execution_result.result,
                "error": "",
                "code": execution_result.code,
                "retries": retries,
            }

        return {
            "success": False, "result": None,
            "error": execution_result.error,
            "code": execution_result.code, "retries": retries,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI-compatible chat completion endpoint."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            if usage:
                logger.debug(
                    "Token usage: prompt=%d, completion=%d, total=%d",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
                )
            return content.strip()

        except APITimeoutError:
            logger.error("LLM request timed out after %.1fs.", self.request_timeout)
            raise
        except APIError as exc:
            logger.error("LLM API error: %s", exc)
            raise

    def _call_llm_and_parse(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM and immediately parse out the Python code."""
        raw = self._call_llm(messages)
        logger.debug("Raw LLM response:\n%s", raw[:800])
        return CodeParser.parse(raw)

    @staticmethod
    def _sanitize_error(error: str) -> str:
        """Sanitize execution error messages before feeding back to the LLM."""
        if not error:
            return "Unknown execution error."
        sanitized = " ".join(error.split())
        max_len = 600
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len] + " ... [truncated]"
        return sanitized


def create_engine(
    df_orders,
    df_details,
    df_logistics,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> ChatBIQueryEngine:
    """Factory function to create a ChatBIQueryEngine with sensible defaults."""
    return ChatBIQueryEngine(
        df_orders=df_orders,
        df_details=df_details,
        df_logistics=df_logistics,
        base_url=base_url,
        api_key=api_key,
        model=model,
        **kwargs,
    )
