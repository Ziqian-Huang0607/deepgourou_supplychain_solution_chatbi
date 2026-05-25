"""
ChatBI Agent - 绝配-港大AI赛 Question 1 Solution
===============================================

A production-grade ChatBI agent that answers natural language questions
about supply-chain order data using Program-Aided Language (PAL) with
local Ollama LLM inference.

Quick Start (Ollama):
    >>> from chatbi_agent import ChatBIAgent
    >>> agent = ChatBIAgent()
    >>> result = agent.answer("1月20日当天有多少个配送订单被处理？")
    >>> print(result.answer)

Modules:
    config          - Centralized configuration
    data_loader     - Excel data loading & normalization
    schema_manager  - Schema extraction for LLM prompts
    query_engine    - Core PAL code generation engine
    code_executor   - Safe sandboxed code execution
    self_correction - Error recovery & retry logic
    answer_formatter - Natural language answer generation
    result_cache    - Query result caching
    ollama_client   - Local Ollama LLM integration
    main            - CLI entry point
"""

__version__ = "1.0.0"
__author__ = "Competition Team"

# Lazy imports to avoid heavy dependencies on import
__all__ = [
    "ChatBIAgent",
    "AgentResult",
    "get_data",
    "get_combined_schema",
    "create_engine",
    "CodeExecutor",
    "OllamaClient",
    "get_ollama_engine",
]
