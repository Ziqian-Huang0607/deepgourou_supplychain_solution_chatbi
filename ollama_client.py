"""
Ollama Client Module — Local LLM inference for the ChatBI Agent.
===============================================================

This module provides an OpenAI-compatible client wrapper for Ollama,
with automatic model detection, smart fallback, and competition-optimized
settings.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1
which means the standard `openai` library works directly. This module
adds convenience features on top:

    * Auto-detect available models
    * Recommend best code model for this competition
    * Pull models if not present
    * Health check and connection validation
    * Fallback chain: primary -> fallback -> error

Usage:
    >>> from ollama_client import OllamaClient, get_ollama_engine
    >>> engine = get_ollama_engine()  # auto-detect best model
    >>> result = engine.ask("1月20日有多少订单被审核？")

Environment Variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL: Specific model to use (default: auto-detect)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")

# Recommended models for this competition (ranked by code quality)
RECOMMENDED_MODELS: List[str] = [
    # Tier 1: Best code generation at 7B (HIGHEST RECOMMENDATION)
    "qwen2.5-coder:7b",
    "qwen2.5-coder:7b-instruct",
    "qwen2.5-coder:7b-instruct-q4_0",
    "qwen2.5-coder:7b-instruct-q4_K_M",
    # Tier 2: Strong alternatives
    "codellama:7b",
    "codellama:7b-instruct",
    "deepseek-coder:6.7b",
    "deepseek-coder-v2:16b",
    # Tier 3: Smaller/faster options
    "qwen2.5-coder:3b",
    "qwen2.5-coder:1.5b",
    "codegemma:2b",
    # Tier 4: General models (work but not optimized for code)
    "qwen2.5:7b",
    "llama3.1:8b",
    "mistral:7b",
]

# OpenAI-compatible API path
OLLAMA_API_PATH: str = "/v1/chat/completions"


# ---------------------------------------------------------------------------
# Health check & model detection
# ---------------------------------------------------------------------------

def check_ollama_running(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """Check if Ollama server is reachable.

    Args:
        host: Ollama server base URL.

    Returns:
        True if server responds, False otherwise.
    """
    try:
        req = urllib.request.Request(
            f"{host}/api/tags",
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.debug("Ollama health check failed: %s", exc)
        return False


def list_local_models(host: str = DEFAULT_OLLAMA_HOST) -> List[str]:
    """List all models currently available in Ollama.

    Args:
        host: Ollama server base URL.

    Returns:
        List of model tag strings, e.g. ['qwen2.5-coder:7b', 'llama3.1:8b'].
    """
    try:
        req = urllib.request.Request(
            f"{host}/api/tags",
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", m.get("model", "")) for m in data.get("models", [])]
            return [m for m in models if m]
    except Exception as exc:
        logger.warning("Failed to list Ollama models: %s", exc)
        return []


def select_best_model(
    available: List[str],
    recommendations: List[str] = RECOMMENDED_MODELS,
) -> Optional[str]:
    """Select the best available model from recommendations.

    Iterates through the recommendation list and returns the first
    model that is present in *available*.

    Args:
        available: List of locally installed model tags.
        recommendations: Priority-ordered list of preferred models.

    Returns:
        Best matching model tag, or None if no overlap.
    """
    available_set = set(available)
    for rec in recommendations:
        # Exact match
        if rec in available_set:
            return rec
        # Partial match (e.g. 'qwen2.5-coder:7b' matches 'qwen2.5-coder:7b-instruct')
        for av in available_set:
            if rec in av or av in rec:
                return av
    return None


def recommend_model_for_competition(host: str = DEFAULT_OLLAMA_HOST) -> str:
    """Return the best available model for this competition.

    Detects installed models and returns the highest-quality code model.
    If no recommended models are found, returns the first available model.

    Args:
        host: Ollama server base URL.

    Returns:
        Model tag string, e.g. 'qwen2.5-coder:7b'.

    Raises:
        RuntimeError: If Ollama is not running or no models are installed.
    """
    if not check_ollama_running(host):
        raise RuntimeError(
            f"Ollama server not reachable at {host}.\n"
            "Please start Ollama first:  ollama serve\n"
            "Or install it:  https://ollama.com/download"
        )

    available = list_local_models(host)
    if not available:
        raise RuntimeError(
            "Ollama is running but no models are installed.\n"
            "Install a recommended model:\n"
            "  ollama pull qwen2.5-coder:7b      # BEST for this competition\n"
            "  ollama pull qwen2.5-coder:3b      # faster, smaller\n"
            "  ollama pull codellama:7b          # alternative\n"
        )

    # User-specified model takes precedence
    if DEFAULT_OLLAMA_MODEL:
        if DEFAULT_OLLAMA_MODEL in available:
            logger.info("Using user-specified model: %s", DEFAULT_OLLAMA_MODEL)
            return DEFAULT_OLLAMA_MODEL
        logger.warning(
            "User-specified model '%s' not found in Ollama. "
            "Installed models: %s",
            DEFAULT_OLLAMA_MODEL, available,
        )

    best = select_best_model(available)
    if best:
        logger.info("Auto-selected best model: %s", best)
        return best

    # Fallback: just use the first available model
    fallback = available[0]
    logger.warning(
        "No recommended models found. Using first available: %s. "
        "For best results, run: ollama pull qwen2.5-coder:7b",
        fallback,
    )
    return fallback


# ---------------------------------------------------------------------------
# Model metadata helpers
# ---------------------------------------------------------------------------

def get_model_info(model_name: str, host: str = DEFAULT_OLLAMA_HOST) -> Dict[str, Any]:
    """Get metadata about a specific Ollama model.

    Args:
        model_name: Model tag, e.g. 'qwen2.5-coder:7b'.
        host: Ollama server base URL.

    Returns:
        Dictionary with model metadata.
    """
    try:
        req = urllib.request.Request(
            f"{host}/api/show",
            data=json.dumps({"name": model_name}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Failed to get model info for %s: %s", model_name, exc)
        return {}


def estimate_param_size(model_name: str) -> str:
    """Estimate parameter size from model name for competition scoring.

    Args:
        model_name: Model tag string.

    Returns:
        Human-readable parameter size, e.g. '7B'.
    """
    name_lower = model_name.lower()
    if "1.5b" in name_lower or "1.5_b" in name_lower:
        return "1.5B"
    if "3b" in name_lower and "7b" not in name_lower and "14b" not in name_lower:
        return "3B"
    if "6.7b" in name_lower:
        return "6.7B"
    if "7b" in name_lower:
        return "7B"
    if "8b" in name_lower:
        return "8B"
    if "9b" in name_lower:
        return "9B"
    if "14b" in name_lower:
        return "14B"
    if "16b" in name_lower:
        return "16B"
    if "32b" in name_lower:
        return "32B"
    if "70b" in name_lower:
        return "70B"
    return "unknown"


# ---------------------------------------------------------------------------
# Ollama engine wrapper (OpenAI-compatible)
# ---------------------------------------------------------------------------

class OllamaClient:
    """OpenAI-compatible client wrapper for Ollama.

    Uses the standard `openai` library with Ollama's OpenAI-compatible
    endpoint. This provides a drop-in replacement for cloud API usage.

    Args:
        model: Ollama model tag. If None, auto-detects best available.
        host: Ollama server base URL.
        temperature: Sampling temperature (default 0.1 for deterministic code).
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: str = DEFAULT_OLLAMA_HOST,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model or recommend_model_for_competition(host)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Build OpenAI-compatible client
        self._init_openai_client()

        logger.info(
            "OllamaClient initialized | model=%s | host=%s | temp=%.2f | ~%s params",
            self.model, self.host, self.temperature, estimate_param_size(self.model),
        )

    def _init_openai_client(self) -> None:
        """Initialize the underlying OpenAI client pointing at Ollama."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for Ollama integration. "
                "Install it:  pip install openai"
            ) from exc

        self._client = OpenAI(
            base_url=f"{self.host}/v1",
            api_key="ollama",  # required but unused by Ollama
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send chat messages to Ollama and return response content.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            top_p: Override default top_p.
            max_tokens: Override default max_tokens.

        Returns:
            Response content string from the model.

        Raises:
            RuntimeError: On API errors.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature if temperature is not None else self.temperature,
                top_p=top_p if top_p is not None else self.top_p,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
            content = response.choices[0].message.content or ""

            # Log token usage if available
            usage = response.usage
            if usage:
                logger.debug(
                    "Ollama token usage: prompt=%d, completion=%d, total=%d",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
                )
            return content.strip()

        except Exception as exc:
            raise RuntimeError(f"Ollama API error: {exc}") from exc

    def generate_code(self, prompt_messages: List[Dict[str, str]]) -> str:
        """Convenience: generate code from prompt messages.

        Args:
            prompt_messages: Full message list (system + user).

        Returns:
            Generated code string (raw LLM output).
        """
        return self.chat(prompt_messages)

    def ping(self) -> bool:
        """Check if the Ollama server and model are ready."""
        try:
            self.chat([{"role": "user", "content": "Hi"}], max_tokens=5)
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"OllamaClient(model='{self.model}', host='{self.host}', "
            f"temp={self.temperature}, top_p={self.top_p})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_ollama_client(
    model: Optional[str] = None,
    host: Optional[str] = None,
    **kwargs: Any,
) -> OllamaClient:
    """Factory: create an OllamaClient with auto-detection.

    Args:
        model: Model tag (auto-detect if None).
        host: Ollama host URL (default from OLLAMA_HOST env var).
        **kwargs: Additional arguments passed to OllamaClient.

    Returns:
        Configured OllamaClient instance.
    """
    return OllamaClient(
        model=model,
        host=host or DEFAULT_OLLAMA_HOST,
        **kwargs,
    )


def get_ollama_engine(
    df_orders,
    df_details,
    df_logistics,
    model: Optional[str] = None,
    host: Optional[str] = None,
    **kwargs: Any,
):
    """Factory: create a full ChatBIQueryEngine backed by Ollama.

    This is the one-liner to get a competition-ready engine:

        >>> from ollama_client import get_ollama_engine
        >>> from data_loader import get_data
        >>> df_o, df_d, df_l = get_data()
        >>> engine = get_ollama_engine(df_o, df_d, df_l)
        >>> result = engine.ask("1月20日有多少订单被审核？")

    Args:
        df_orders, df_details, df_logistics: Pre-loaded DataFrames.
        model: Ollama model tag (auto-detect if None).
        host: Ollama host URL.
        **kwargs: Extra args for ChatBIQueryEngine.

    Returns:
        Configured ChatBIQueryEngine instance.
    """
    from query_engine import create_engine

    ollama_host = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
    ollama_model = model or DEFAULT_OLLAMA_MODEL or recommend_model_for_competition(ollama_host)

    logger.info("=" * 60)
    logger.info("Creating Ollama-backed ChatBI engine")
    logger.info("Model: %s (~%s params)", ollama_model, estimate_param_size(ollama_model))
    logger.info("Host:  %s", ollama_host)
    logger.info("=" * 60)

    return create_engine(
        df_orders=df_orders,
        df_details=df_details,
        df_logistics=df_logistics,
        base_url=f"{ollama_host}/v1",
        api_key="ollama",
        model=ollama_model,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def print_ollama_setup_guide() -> None:
    """Print step-by-step setup instructions for Ollama."""
    guide = """
╔══════════════════════════════════════════════════════════════════╗
║               Ollama Setup Guide for ChatBI Agent                ║
╚══════════════════════════════════════════════════════════════════╝

1. INSTALL OLLAMA
   ───────────────
   macOS:    brew install ollama
   Linux:    curl -fsSL https://ollama.com/install.sh | sh
   Windows:  https://ollama.com/download

2. START OLLAMA SERVER
   ────────────────────
   ollama serve

   (Keep this terminal open. Or run as a background service.)

3. PULL A RECOMMENDED MODEL
   ─────────────────────────
   # BEST for this competition (7B code model):
   ollama pull qwen2.5-coder:7b

   # Smaller/faster alternatives:
   ollama pull qwen2.5-coder:3b
   ollama pull qwen2.5-coder:1.5b

   # Other good options:
   ollama pull codellama:7b
   ollama pull deepseek-coder:6.7b

4. VERIFY INSTALLATION
   ────────────────────
   ollama list                    # Should show your model
   ollama run qwen2.5-coder:7b    # Test interactively

5. RUN THE CHATBI AGENT
   ─────────────────────
   # Auto-detect model
   python main.py --question "1月20日有多少订单被审核？" --ollama

   # Specify model explicitly
   python main.py --question "..." --ollama-model qwen2.5-coder:7b

   # Custom Ollama host
   python main.py --question "..." --ollama --ollama-host http://192.168.1.100:11434

╚══════════════════════════════════════════════════════════════════╝
"""
    print(guide)


if __name__ == "__main__":
    # When run directly, show setup guide and diagnostics
    print_ollama_setup_guide()

    print("\n--- Diagnostics ---")
    host = DEFAULT_OLLAMA_HOST
    print(f"OLLAMA_HOST: {host}")

    if check_ollama_running(host):
        print("✅ Ollama server is RUNNING")
        models = list_local_models(host)
        if models:
            print(f"Installed models: {models}")
            best = select_best_model(models)
            print(f"Recommended model for competition: {best}")
        else:
            print("⚠️  No models installed. Run: ollama pull qwen2.5-coder:7b")
    else:
        print("❌ Ollama server is NOT RUNNING")
        print("   Start it: ollama serve")
