#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatBI Agent - Main Orchestration Module
========================================
Entry point for the 绝配-港大AI赛 ChatBI competition agent.

Orchestrates data loading, schema extraction, LLM code generation,
safe code execution, self-correction, answer formatting, and caching.

Usage:
    python main.py --question "1月20日当天有多少个配送订单被处理？"
    python main.py --batch questions.txt --output answers.json
    python main.py --interactive
    python main.py --test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger("ChatBI")

# ---------------------------------------------------------------------------
# Module imports with graceful degradation
# ---------------------------------------------------------------------------
_MODULES_AVAILABLE: Dict[str, bool] = {}

def _try_import(module_name: str):
    """Attempt to import a module and track availability."""
    try:
        __import__(module_name)
        _MODULES_AVAILABLE[module_name] = True
        return True
    except Exception:
        _MODULES_AVAILABLE[module_name] = False
        return False

try:
    import config
    _MODULES_AVAILABLE["config"] = True
except Exception as _exc:
    logger.warning("config.py not found: %s", _exc)
    config = None  # type: ignore
    _MODULES_AVAILABLE["config"] = False

try:
    import data_loader
    _MODULES_AVAILABLE["data_loader"] = True
except Exception as _exc:
    logger.warning("data_loader.py not found: %s", _exc)
    data_loader = None  # type: ignore
    _MODULES_AVAILABLE["data_loader"] = False

try:
    import schema_manager
    _MODULES_AVAILABLE["schema_manager"] = True
except Exception as _exc:
    logger.warning("schema_manager.py not found: %s", _exc)
    schema_manager = None  # type: ignore
    _MODULES_AVAILABLE["schema_manager"] = False

try:
    from query_engine import ChatBIQueryEngine, create_engine
    _MODULES_AVAILABLE["query_engine"] = True
except Exception as _exc:
    logger.warning("query_engine.py not found: %s", _exc)
    ChatBIQueryEngine = None  # type: ignore
    create_engine = None  # type: ignore
    _MODULES_AVAILABLE["query_engine"] = False

try:
    from code_executor import CodeExecutor, ExecutionResult
    _MODULES_AVAILABLE["code_executor"] = True
except Exception as _exc:
    logger.warning("code_executor.py not found: %s", _exc)
    CodeExecutor = None  # type: ignore
    ExecutionResult = None  # type: ignore
    _MODULES_AVAILABLE["code_executor"] = False

try:
    import self_correction
    _MODULES_AVAILABLE["self_correction"] = True
except Exception as _exc:
    logger.warning("self_correction.py not found: %s", _exc)
    self_correction = None  # type: ignore
    _MODULES_AVAILABLE["self_correction"] = False

try:
    import answer_formatter
    _MODULES_AVAILABLE["answer_formatter"] = True
except Exception as _exc:
    logger.warning("answer_formatter.py not found: %s", _exc)
    answer_formatter = None  # type: ignore
    _MODULES_AVAILABLE["answer_formatter"] = False

try:
    import result_cache
    _MODULES_AVAILABLE["result_cache"] = True
except Exception as _exc:
    logger.warning("result_cache.py not found: %s", _exc)
    result_cache = None  # type: ignore
    _MODULES_AVAILABLE["result_cache"] = False

try:
    import ollama_client
    from ollama_client import get_ollama_engine, OllamaClient
    _MODULES_AVAILABLE["ollama_client"] = True
except Exception as _exc:
    logger.warning("ollama_client.py not found: %s", _exc)
    ollama_client = None  # type: ignore
    get_ollama_engine = None  # type: ignore
    OllamaClient = None  # type: ignore
    _MODULES_AVAILABLE["ollama_client"] = False

try:
    from dotenv import load_dotenv
    load_dotenv()
    _MODULES_AVAILABLE["dotenv"] = True
except Exception:
    _MODULES_AVAILABLE["dotenv"] = False


# ---------------------------------------------------------------------------
# Timing / profiling utilities
# ---------------------------------------------------------------------------
@dataclass
class TimingProfile:
    """Stores timing information for each pipeline stage."""
    stage: str
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineTimer:
    """Context manager for timing pipeline stages."""
    def __init__(self, stage: str, profile_list: List[TimingProfile]):
        self.stage = stage
        self.profile_list = profile_list
        self.start: float = 0.0
        self.metadata: Dict[str, Any] = {}
        self.elapsed: float = 0.0

    def __enter__(self) -> "PipelineTimer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = (time.perf_counter() - self.start) * 1000.0
        self.profile_list.append(
            TimingProfile(
                stage=self.stage,
                elapsed_ms=round(self.elapsed, 2),
                metadata=self.metadata,
            )
        )
        logger.info("[timer] Stage '%s' finished in %.2f ms", self.stage, self.elapsed)


@contextmanager
def timer(stage: str, profile_list: List[TimingProfile]):
    t = PipelineTimer(stage, profile_list)
    try:
        yield t
    finally:
        t.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    """Structured result from the ChatBI agent."""
    question: str
    answer: str = ""
    code: str = ""
    success: bool = False
    error: str = ""
    retries: int = 0
    timings: List[TimingProfile] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "code": self.code,
            "success": self.success,
            "error": self.error,
            "retries": self.retries,
            "timings": [{"stage": t.stage, "elapsed_ms": t.elapsed_ms, **t.metadata} for t in self.timings],
            "from_cache": self.from_cache,
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "BASE_URL": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "MODEL": os.getenv("CHATBI_MODEL", "gpt-4o-mini"),
    "MAX_RETRIES": int(os.getenv("CHATBI_MAX_RETRIES", "2")),
    "CACHE_ENABLED": os.getenv("CHATBI_CACHE_ENABLED", "true").lower() == "true",
    "CACHE_FILE": os.getenv("CHATBI_CACHE_FILE", ".chatbi_cache.json"),
    "TEMPERATURE": float(os.getenv("CHATBI_TEMPERATURE", "0.1")),
    "TIMEOUT": int(os.getenv("CHATBI_TIMEOUT", "120")),
}


def _get_cfg(key: str) -> Any:
    """Retrieve config value from config.py or fallback to defaults/env."""
    if config is not None:
        try:
            return getattr(config, key)
        except AttributeError:
            pass
    return DEFAULT_CONFIG.get(key)


# ---------------------------------------------------------------------------
# ChatBI Agent
# ---------------------------------------------------------------------------
class ChatBIAgent:
    """
    Production-grade ChatBI agent for supply-chain natural-language querying.

    Pipeline:
        1. Check cache for existing answer
        2. Load data & build schema-aware prompt
        3. Generate pandas code via LLM
        4. Execute code safely in sandbox
        5. Self-correction on error (up to 2 retries)
        6. Format result into Chinese natural language
        7. Cache and return
    """

    TEST_QUESTIONS: List[str] = [
        "1月20日当天有多少个配送订单被处理？",
        "1月20到27日的处理订单数量相比前七天变化是多少？",
        "1月前7天哪个客户下的订单最多，其中什么商品种类数量最多，是多少",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        use_ollama: bool = True,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> None:
        logger.info("=" * 60)
        logger.info("ChatBI Agent initializing")
        logger.info("=" * 60)

        self.api_key: str = (api_key or _get_cfg("API_KEY") or "").strip()
        self.base_url: str = (base_url or _get_cfg("BASE_URL") or "").rstrip("/")
        self.model: str = (model or _get_cfg("MODEL") or "gpt-4o-mini")
        self.temperature: float = _get_cfg("TEMPERATURE") or 0.1
        self.timeout: int = _get_cfg("TIMEOUT") or 120

        self.max_retries: int = max_retries if max_retries is not None else (
            _get_cfg("MAX_RETRIES") or 2
        )
        self.cache_enabled: bool = cache_enabled if cache_enabled is not None else (
            _get_cfg("CACHE_ENABLED") or True
        )

        self._df_orders: Optional[pd.DataFrame] = None
        self._df_details: Optional[pd.DataFrame] = None
        self._df_logistics: Optional[pd.DataFrame] = None
        self._schema: Optional[str] = None
        self._cache: Optional[Any] = None
        self._query_engine: Optional[Any] = None
        self._code_executor: Optional[Any] = None
        self._initialized: bool = False

        self.use_ollama: bool = use_ollama
        self.ollama_host: str = (ollama_host or _get_cfg("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.ollama_model: str = ollama_model or _get_cfg("OLLAMA_MODEL") or ""

        for mod, avail in _MODULES_AVAILABLE.items():
            logger.info("Module '%s': %s", mod, "OK" if avail else "MISSING")

        mode = "Ollama" if self.use_ollama else "Cloud API"
        logger.info("ChatBI Agent initialized (mode=%s, model=%s)", mode, self.model)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        logger.info("Lazy-initializing data, schema, and components...")

        if data_loader is not None:
            try:
                logger.info("Loading data via data_loader.get_data()...")
                self._df_orders, self._df_details, self._df_logistics = data_loader.get_data()
                logger.info(
                    "Data loaded: df_orders=%s, df_details=%s, df_logistics=%s",
                    self._df_orders.shape, self._df_details.shape, self._df_logistics.shape,
                )
            except Exception as exc:
                logger.error("data_loader failed: %s", exc)
                raise
        else:
            raise RuntimeError("data_loader module is required but not available.")

        if schema_manager is not None:
            try:
                logger.info("Building schema via schema_manager.get_combined_schema()...")
                self._schema = schema_manager.get_combined_schema()
                logger.info("Schema built (%d chars)", len(self._schema))
            except Exception as exc:
                logger.error("schema_manager failed: %s", exc)
                self._schema = self._fallback_schema()
        else:
            self._schema = self._fallback_schema()

        if self._df_orders is not None:
            if self.use_ollama and get_ollama_engine is not None:
                try:
                    logger.info("Initializing Ollama query engine...")
                    self._query_engine = get_ollama_engine(
                        df_orders=self._df_orders,
                        df_details=self._df_details,
                        df_logistics=self._df_logistics,
                        model=self.ollama_model or None,
                        host=self.ollama_host,
                        temperature=self.temperature,
                        max_retries=self.max_retries,
                        request_timeout=self.timeout,
                    )
                    logger.info("Ollama query engine initialized")
                except Exception as exc:
                    logger.error("Ollama init failed: %s", exc)
                    self._query_engine = None

            if self._query_engine is None and create_engine is not None:
                try:
                    logger.info("Initializing cloud API query engine...")
                    self._query_engine = create_engine(
                        df_orders=self._df_orders,
                        df_details=self._df_details,
                        df_logistics=self._df_logistics,
                        base_url=self.base_url,
                        api_key=self.api_key,
                        model=self.model,
                        temperature=self.temperature,
                        max_retries=self.max_retries,
                        request_timeout=self.timeout,
                    )
                    logger.info("Cloud query engine initialized")
                except Exception as exc:
                    logger.error("Cloud query engine init failed: %s", exc)
                    self._query_engine = None
        else:
            self._query_engine = None

        if CodeExecutor is not None and self._df_orders is not None:
            try:
                self._code_executor = CodeExecutor(
                    df_orders=self._df_orders,
                    df_details=self._df_details,
                    df_logistics=self._df_logistics,
                )
                logger.info("Code executor initialized")
            except Exception as exc:
                logger.error("Code executor init failed: %s", exc)
                self._code_executor = None
        else:
            self._code_executor = None

        if self.cache_enabled and result_cache is not None:
            try:
                self._cache = result_cache.get_default_cache()
                logger.info("Cache enabled")
            except Exception as exc:
                logger.error("result_cache init failed: %s", exc)
                self._cache = None
        else:
            self._cache = None

        self._initialized = True
        logger.info("Lazy initialization complete.")

    def _fallback_schema(self) -> str:
        lines = ["# DataFrames (fallback schema)", ""]
        for name, df in [("df_orders", self._df_orders), ("df_details", self._df_details), ("df_logistics", self._df_logistics)]:
            if df is not None:
                lines.append(f"## {name}")
                lines.append(f"- Shape: {df.shape}")
                lines.append(f"- Columns: {list(df.columns)}")
                lines.append("")
        return "\n".join(lines)

    def answer(self, question: str) -> AgentResult:
        timings: List[TimingProfile] = []
        result = AgentResult(question=question)

        try:
            self._ensure_initialized()
        except Exception as exc:
            logger.critical("Initialization failed: %s", exc)
            result.error = f"Initialization failed: {exc}"
            return result

        with timer("cache_lookup", timings):
            cached = self._check_cache(question)
            if cached is not None:
                result.answer = cached.get("answer", "")
                result.code = cached.get("code", "")
                result.success = True
                result.from_cache = True
                result.timings = timings
                return result

        if self._query_engine is not None:
            with timer("query_execute", timings) as pt:
                try:
                    query_result = self._query_engine.ask(question)
                    result.code = query_result.get("code", "")
                    result.retries = query_result.get("retries", 0)
                    pt.metadata["retries"] = result.retries

                    if query_result.get("success"):
                        with timer("format_answer", timings):
                            nl_answer = self._format_answer(query_result["result"], question)
                            result.answer = nl_answer
                        result.success = True
                    else:
                        result.error = query_result.get("error", "Unknown error")

                except Exception as exc:
                    logger.error("Query engine error: %s", exc)
                    result.error = f"Query engine error: {exc}"
        else:
            result.error = "Query engine not available"

        result.timings = timings

        if result.success:
            with timer("cache_write", timings):
                self._save_to_cache(result)

        return result

    def _check_cache(self, question: str) -> Optional[Dict[str, Any]]:
        if self._cache is None:
            return None
        try:
            return self._cache.get(question)
        except Exception as exc:
            logger.warning("Cache lookup error: %s", exc)
            return None

    def _save_to_cache(self, result: AgentResult) -> None:
        if self._cache is None or not result.success:
            return
        try:
            self._cache.set(result.question, {"answer": result.answer, "code": result.code})
        except Exception as exc:
            logger.warning("Cache write error: %s", exc)

    def _format_answer(self, execution_result: Any, question: str) -> str:
        if answer_formatter is not None:
            try:
                if hasattr(answer_formatter, 'format_answer'):
                    return answer_formatter.format_answer(execution_result, question)
                if hasattr(answer_formatter, 'AnswerFormatter'):
                    return answer_formatter.AnswerFormatter().format(execution_result, question)
            except Exception as exc:
                logger.error("answer_formatter failed: %s", exc)
        return self._fallback_format(execution_result, question)

    def _fallback_format(self, execution_result: Any, question: str) -> str:
        if execution_result is None:
            return "查询结果为空。"
        if isinstance(execution_result, pd.DataFrame):
            if execution_result.empty:
                return "查询结果为空数据表。"
            return f"查询结果如下：\n{execution_result.to_string(index=False)}"
        if isinstance(execution_result, pd.Series):
            return f"查询结果：\n{execution_result.to_string()}"
        if isinstance(execution_result, (int, float, np.integer, np.floating)):
            rounded = round(float(execution_result), 2)
            if rounded == int(rounded):
                return f"{int(rounded)}"
            return f"{rounded}"
        if isinstance(execution_result, (list, tuple)):
            if len(execution_result) == 0:
                return "查询结果为空列表。"
            if len(execution_result) == 2 and isinstance(execution_result[0], (int, float)) and isinstance(execution_result[1], (int, float)):
                v1, v2 = execution_result
                if abs(v2) <= 1:
                    return f"增加{v1}单，增幅{v2*100:.2f}%。"
                return f"{v1}, {v2}"
            items = "\n".join(f"- {item}" for item in execution_result[:50])
            if len(execution_result) > 50:
                items += f"\n... (共 {len(execution_result)} 项)"
            return f"查询结果共 {len(execution_result)} 项：\n{items}"
        if isinstance(execution_result, dict):
            items = "\n".join(f"- {k}: {v}" for k, v in execution_result.items())
            return f"查询结果：\n{items}"
        return str(execution_result)

    def answer_batch(self, questions: List[str], progress_every: int = 10) -> List[AgentResult]:
        results: List[AgentResult] = []
        total = len(questions)
        logger.info("Batch processing %d questions...", total)

        for idx, q in enumerate(questions, 1):
            if idx % progress_every == 0 or idx == 1:
                logger.info("[%d/%d] Processing: %s", idx, total, q[:80])
            try:
                res = self.answer(q)
            except Exception as exc:
                logger.error("Unhandled error on question '%s': %s", q, exc)
                res = AgentResult(question=q, answer="", success=False, error=f"Unhandled: {exc}")
            results.append(res)

        logger.info("Batch complete: %d/%d succeeded", sum(r.success for r in results), total)
        return results


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ChatBI Agent",
        description="绝配-港大AI赛 - ChatBI Agent for supply-chain NL querying.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --question '1月20日当天有多少个配送订单被处理？'\n"
            "  python main.py --batch questions.txt --output answers.json\n"
            "  python main.py --interactive\n"
            "  python main.py --test\n"
        ),
    )
    parser.add_argument("--question", "-q", type=str, default=None, help="Single question")
    parser.add_argument("--batch", "-b", type=str, default=None, help="Batch file (one Q per line)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL")
    parser.add_argument("--test", "-t", action="store_true", help="Run built-in test questions")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file")
    parser.add_argument("--format", "-f", type=str, choices=["json", "jsonl"], default="json")

    ollama_group = parser.add_argument_group("Ollama (Local LLM) Options")
    ollama_group.add_argument(
        "--ollama", action="store_true", default=True,
        help="Use Ollama local model (default: True). Use --no-ollama for cloud API.")
    ollama_group.add_argument(
        "--no-ollama", action="store_true",
        help="Disable Ollama, use cloud API instead.")
    ollama_group.add_argument(
        "--ollama-host", type=str, default=None,
        help="Ollama server URL (default: http://localhost:11434).")
    ollama_group.add_argument(
        "--ollama-model", type=str, default=None,
        help="Ollama model tag, e.g. 'qwen2.5-coder:7b' (auto-detect if omitted).")
    ollama_group.add_argument(
        "--ollama-setup", action="store_true",
        help="Print Ollama setup guide and exit.")

    cloud_group = parser.add_argument_group("Cloud API (Fallback) Options")
    cloud_group.add_argument("--api-key", type=str, default=None, help="OpenAI API key.")
    cloud_group.add_argument("--base-url", type=str, default=None, help="API base URL.")
    cloud_group.add_argument("--model", type=str, default=None, help="Cloud model name.")
    cloud_group.add_argument("--max-retries", type=int, default=None,
                             help="Max self-correction retries.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    parser.add_argument("--profile", action="store_true", help="Print timing profile.")
    return parser


def run_single(agent: ChatBIAgent, question: str, args: argparse.Namespace) -> None:
    result = agent.answer(question)
    print("\n" + "=" * 60)
    print(f"Question: {result.question}")
    print(f"Success:  {result.success}")
    if result.from_cache:
        print("[Cache Hit]")
    if result.answer:
        print(f"\nAnswer:\n{result.answer}")
    if result.code:
        print(f"\nGenerated Code:\n{result.code}")
    if result.error:
        print(f"\nError: {result.error}")
    if result.retries:
        print(f"Retries: {result.retries}")
    if args.profile:
        print("\n--- Timing Profile ---")
        total = sum(t.elapsed_ms for t in result.timings)
        for t in result.timings:
            pct = (t.elapsed_ms / total * 100) if total else 0
            print(f"  {t.stage:25s} {t.elapsed_ms:>10.2f} ms ({pct:5.1f}%)")
        print(f"  {'TOTAL':25s} {total:>10.2f} ms")
    print("=" * 60 + "\n")
    if args.output:
        _write_output([result], args.output, args.format)


def run_batch(agent: ChatBIAgent, filepath: str, args: argparse.Namespace) -> None:
    path = Path(filepath)
    if not path.exists():
        logger.error("Batch file not found: %s", filepath)
        sys.exit(1)
    questions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.strip().startswith("#")]
    results = agent.answer_batch(questions)
    succeeded = sum(r.success for r in results)
    print(f"\nBatch complete: {succeeded}/{len(results)} succeeded")
    if args.output:
        _write_output(results, args.output, args.format)
    else:
        for r in results:
            status = "OK" if r.success else "FAIL"
            print(f"[{status}] {r.question[:70]:70s} -> {r.answer[:80]}")


def run_interactive(agent: ChatBIAgent) -> None:
    print("=" * 60)
    print("ChatBI Agent - Interactive Mode")
    print("Type 'quit' or 'exit' to quit, 'help' for commands")
    print("=" * 60)
    while True:
        try:
            question = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "help":
            print("Commands: help, quit | Type your question in Chinese")
            continue
        if not question:
            continue
        result = agent.answer(question)
        if result.success:
            print(f"A> {result.answer}")
        else:
            print(f"Error: {result.error}")


def run_test(agent: ChatBIAgent, args: argparse.Namespace) -> None:
    print("=" * 60)
    print("Running built-in test questions")
    print("=" * 60)
    results = agent.answer_batch(agent.TEST_QUESTIONS, progress_every=1)
    print("\n--- Results ---")
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"[{status}] {r.question[:60]}")
        print(f"       Answer: {r.answer[:100] if r.answer else 'N/A'}")
        if r.error:
            print(f"       Error: {r.error[:100]}")
    succeeded = sum(r.success for r in results)
    print(f"\n{succeeded}/{len(results)} tests passed")
    if args.output:
        _write_output(results, args.output, args.format)


def _write_output(results: List[AgentResult], filepath: str, fmt: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    logger.info("Results written to %s", path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if getattr(args, "ollama_setup", False):
        if ollama_client is not None:
            ollama_client.print_ollama_setup_guide()
        else:
            print("ollama_client module not available.")
        sys.exit(0)

    use_ollama = args.ollama and not getattr(args, "no_ollama", False)

    if not any([args.question, args.batch, args.interactive, args.test]):
        parser.print_help()
        sys.exit(0)

    logger.info("Creating agent | Ollama=%s | model=%s",
                use_ollama, args.ollama_model or args.model or "auto-detect")

    agent = ChatBIAgent(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_retries=args.max_retries,
        cache_enabled=not args.no_cache,
        use_ollama=use_ollama,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
    )

    if args.question:
        run_single(agent, args.question, args)
    elif args.batch:
        run_batch(agent, args.batch, args)
    elif args.interactive:
        run_interactive(agent)
    elif args.test:
        run_test(agent, args)


if __name__ == "__main__":
    main()
