"""
Result cache module for the ChatBI agent.

Provides a persistent, thread-safe cache that:
1. Hashes the question string to produce a cache key
2. Checks for a previously computed answer before invoking code generation
3. Serialises answers to disk as JSON
4. Invalidates entries automatically when the underlying data file changes
   (detected via modification time)
5. Is fully thread-safe via ``threading.RLock``

Usage:
    cache = ResultCache(
        cache_file="/tmp/chatbi_cache.json",
        data_files=["/data/orders.parquet"],
    )

    # Lookup
    answer = cache.get("1月20日有多少配送订单？")
    if answer is not None:
        print("Cache hit:", answer)
    else:
        result = expensive_computation()
        cache.set("1月20日有多少配送订单？", result)

    # Persist to disk
    cache.save()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON helpers for pandas / numpy objects
# ---------------------------------------------------------------------------

class _ChatBIJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that serialises common pandas / numpy types.
    """

    def default(self, obj: Any) -> Any:
        # Pandas Series → {index: value} dict
        try:
            import pandas as pd
            if isinstance(obj, pd.Series):
                return {
                    "__type__": "pandas.Series",
                    "data": obj.to_dict(),
                }
            if isinstance(obj, pd.DataFrame):
                return {
                    "__type__": "pandas.DataFrame",
                    "data": obj.to_dict(orient="records"),
                    "columns": list(obj.columns),
                    "index": list(obj.index),
                }
        except ImportError:  # pragma: no cover
            pass

        # NumPy scalars / arrays
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return {
                    "__type__": "numpy.ndarray",
                    "data": obj.tolist(),
                    "dtype": str(obj.dtype),
                }
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:  # pragma: no cover
            pass

        # Python bytes
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "data": obj.decode("utf-8", errors="replace")}

        # Fallback
        return super().default(obj)


def _json_decode_hook(dct: Dict[str, Any]) -> Any:
    """JSON object hook that reconstructs pandas / numpy objects."""
    type_tag = dct.get("__type__")
    if type_tag is None:
        return dct

    if type_tag == "pandas.Series":
        try:
            import pandas as pd
            return pd.Series(dct["data"])
        except ImportError:  # pragma: no cover
            return dct["data"]

    if type_tag == "pandas.DataFrame":
        try:
            import pandas as pd
            df = pd.DataFrame(dct["data"], columns=dct.get("columns"))
            return df
        except ImportError:  # pragma: no cover
            return dct["data"]

    if type_tag == "numpy.ndarray":
        try:
            import numpy as np
            return np.array(dct["data"], dtype=dct.get("dtype"))
        except ImportError:  # pragma: no cover
            return dct["data"]

    if type_tag == "bytes":
        return dct["data"].encode("utf-8")

    return dct


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """
    A single cached result.

    Attributes
    ----------
    key :
        SHA-256 hex digest of the normalised question.
    question :
        The original question string (stored for debugging / inspection).
    result :
        The cached answer payload (any JSON-serialisable value).
    timestamp :
        Unix timestamp when the entry was created.
    data_mtime :
        Dict mapping data file paths → their mtime at the moment of caching.
        Used for invalidation.
    hit_count :
        Number of times this entry has been served from cache.
    """

    key: str
    question: str
    result: Any
    timestamp: float = field(default_factory=time.time)
    data_mtime: Dict[str, float] = field(default_factory=dict)
    hit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON encoding."""
        return {
            "key": self.key,
            "question": self.question,
            "result": self.result,
            "timestamp": self.timestamp,
            "data_mtime": self.data_mtime,
            "hit_count": self.hit_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheEntry":
        """Reconstruct from a dict (with pandas decoding)."""
        return cls(
            key=d["key"],
            question=d["question"],
            result=_json_decode_hook(d["result"])
            if isinstance(d.get("result"), dict)
            else d.get("result"),
            timestamp=d.get("timestamp", 0.0),
            data_mtime=d.get("data_mtime", {}),
            hit_count=d.get("hit_count", 0),
        )


# ---------------------------------------------------------------------------
# Key builder
# ---------------------------------------------------------------------------

class CacheKeyBuilder:
    """
    Builds deterministic cache keys from question strings.

    Normalisation rules:
    - Collapse whitespace
    - Lower-case (Chinese has no case, but English portions matter)
    - Strip trailing punctuation
    """

    @staticmethod
    def normalise(question: str) -> str:
        """Normalise *question* for stable hashing."""
        q = question.strip().lower()
        # Collapse all whitespace runs to a single space
        q = " ".join(q.split())
        # Strip trailing question marks / punctuation
        q = q.rstrip("？?。.!！")
        return q

    @classmethod
    def build(cls, question: str) -> str:
        """Return a SHA-256 hex digest of the normalised question."""
        normalised = cls.normalise(question)
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main cache class
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    """Configuration for ``ResultCache``."""

    cache_file: Optional[str] = ".chatbi_cache.json"
    data_files: Sequence[str] = field(default_factory=list)
    max_entries: int = 10_000          # LRU eviction trigger
    save_on_every_write: bool = True   # Auto-persist after each set()
    ttl_seconds: Optional[float] = None  # None = no expiry


class ResultCache:
    """
    Persistent, thread-safe result cache with data-file mtime invalidation.

    Parameters
    ----------
    config :
        ``CacheConfig`` instance.  If *None*, sensible defaults are used.

    Example::

        cache = ResultCache(
            CacheConfig(
                cache_file="/app/cache/chatbi.json",
                data_files=["/data/orders.parquet", "/data/inventory.csv"],
            )
        )
        cache.load()   # warm-start from disk

        answer = cache.get("1月20日有多少配送订单？")
        if answer is None:
            answer = run_expensive_pipeline(...)
            cache.set("1月20日有多少配送订单？", answer)

        cache.save()   # flush to disk
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self._cfg = config or CacheConfig()
        self._lock = threading.RLock()
        self._store: Dict[str, CacheEntry] = {}
        self._key_builder = CacheKeyBuilder()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ------------------------------------------------------------------ #
    # Public API — core operations
    # ------------------------------------------------------------------ #

    def get(self, question: str) -> Optional[Any]:
        """
        Lookup a cached result for *question*.

        Returns ``None`` on miss or if the cached entry has been invalidated
        (data file changed or TTL expired).
        """
        key = self._key_builder.build(question)

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # TTL check
            if self._cfg.ttl_seconds is not None:
                age = time.time() - entry.timestamp
                if age > self._cfg.ttl_seconds:
                    logger.debug("Cache entry expired (age=%.0fs)", age)
                    del self._store[key]
                    self._misses += 1
                    return None

            # Data file mtime check
            if not self._is_fresh(entry):
                logger.debug("Cache entry invalidated by data file change")
                del self._store[key]
                self._misses += 1
                return None

            # Cache hit
            entry.hit_count += 1
            self._hits += 1
            logger.debug("Cache hit for key=%s... (hits=%d)", key[:8], self._hits)
            return entry.result

    def set(self, question: str, result: Any) -> None:
        """
        Store *result* as the answer to *question*.

        Automatically captures current data-file mtimes for later invalidation.
        If auto-save is enabled, flushes to disk immediately.
        """
        key = self._key_builder.build(question)
        data_mtime = self._current_data_mtimes()

        entry = CacheEntry(
            key=key,
            question=question,
            result=result,
            timestamp=time.time(),
            data_mtime=data_mtime,
            hit_count=0,
        )

        with self._lock:
            self._store[key] = entry
            self._maybe_evict()

        if self._cfg.save_on_every_write:
            self.save()

    def delete(self, question: str) -> bool:
        """Remove the entry for *question*. Returns ``True`` if it existed."""
        key = self._key_builder.build(question)
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Drop all cached entries from memory (does NOT delete disk file)."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            logger.info("Cleared %d cache entries from memory", count)

    def invalidate_all(self) -> None:
        """Remove all entries AND delete the disk cache file."""
        with self._lock:
            self._store.clear()
            if self._cfg.cache_file and os.path.exists(self._cfg.cache_file):
                try:
                    os.remove(self._cfg.cache_file)
                    logger.info("Deleted cache file: %s", self._cfg.cache_file)
                except OSError as exc:
                    logger.warning("Failed to delete cache file: %s", exc)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def load(self) -> int:
        """
        Load cache entries from disk.  Returns the number of entries loaded.

        Safe to call even if the cache file does not yet exist.
        """
        path = self._cfg.cache_file
        if not path or not os.path.exists(path):
            return 0

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cache from %s: %s", path, exc)
            return 0

        if not isinstance(raw, dict) or "entries" not in raw:
            logger.warning("Cache file malformed, ignoring.")
            return 0

        loaded = 0
        with self._lock:
            for entry_dict in raw["entries"]:
                try:
                    entry = CacheEntry.from_dict(entry_dict)
                    # Skip stale entries immediately on load
                    if self._cfg.ttl_seconds is not None:
                        age = time.time() - entry.timestamp
                        if age > self._cfg.ttl_seconds:
                            continue
                    self._store[entry.key] = entry
                    loaded += 1
                except Exception as exc:
                    logger.debug("Skipping malformed cache entry: %s", exc)

        logger.info("Loaded %d entries from %s", loaded, path)
        return loaded

    def save(self) -> None:
        """Persist current in-memory cache to disk as JSON."""
        path = self._cfg.cache_file
        if not path:
            return

        # Ensure parent directory exists
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:
                logger.warning("Cannot create cache directory %s: %s", parent, exc)
                return

        with self._lock:
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "entries": [
                    entry.to_dict() for entry in self._store.values()
                ],
            }

        try:
            # Atomic write: write to temp file then rename
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, cls=_ChatBIJSONEncoder)
            os.replace(tmp_path, path)
            logger.debug("Cache saved to %s (%d entries)", path, len(self._store))
        except OSError as exc:
            logger.warning("Failed to save cache to %s: %s", path, exc)

    # ------------------------------------------------------------------ #
    # Inspection / statistics
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries_in_memory": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hit_rate(),
                "evictions": self._evictions,
                "cache_file": self._cfg.cache_file,
            }

    def keys(self) -> List[str]:
        """Return all cache keys (hex digests)."""
        with self._lock:
            return list(self._store.keys())

    def questions(self) -> List[str]:
        """Return all cached questions (for debugging)."""
        with self._lock:
            return [e.question for e in self._store.values()]

    def peek(self, question: str) -> Optional[CacheEntry]:
        """Return the raw ``CacheEntry`` for *question* (or None)."""
        key = self._key_builder.build(question)
        with self._lock:
            return self._store.get(key)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_fresh(self, entry: CacheEntry) -> bool:
        """
        Return ``True`` if none of the tracked data files have changed
        since *entry* was created.
        """
        if not entry.data_mtime:
            # No data files tracked → assume fresh
            return True

        current = self._current_data_mtimes()
        for filepath, cached_mtime in entry.data_mtime.items():
            actual_mtime = current.get(filepath)
            if actual_mtime is None:
                # File was deleted → invalidate
                return False
            if actual_mtime != cached_mtime:
                # File was modified → invalidate
                return False
        return True

    def _current_data_mtimes(self) -> Dict[str, float]:
        """Return a dict of {filepath: mtime} for all tracked data files."""
        result: Dict[str, float] = {}
        for filepath in self._cfg.data_files:
            try:
                st = os.stat(filepath)
                result[filepath] = st.st_mtime
            except OSError:
                # File may not exist yet — store 0 as sentinel
                result[filepath] = 0.0
        return result

    def _maybe_evict(self) -> None:
        """LRU eviction if we exceed ``max_entries``."""
        if len(self._store) <= self._cfg.max_entries:
            return

        # Sort by (hit_count, timestamp) ascending — evict least valuable
        sorted_items = sorted(
            self._store.items(),
            key=lambda kv: (kv[1].hit_count, kv[1].timestamp),
        )
        to_evict = len(sorted_items) - self._cfg.max_entries + self._cfg.max_entries // 10
        # Evict 10% over the limit to avoid thrashing
        for key, _ in sorted_items[:to_evict]:
            del self._store[key]
            self._evictions += 1
        logger.info("Evicted %d cache entries (LRU)", to_evict)

    def _hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total


# ---------------------------------------------------------------------------
# Decorator-style helper
# ---------------------------------------------------------------------------

def cached(
    cache: ResultCache,
    key_func: Optional[Callable[..., str]] = None,
):
    """
    Decorator that caches the return value of a function keyed by its
    first string argument (assumed to be the question).

    Example::

        cache = ResultCache()

        @cached(cache)
        def answer_question(question: str, df: pd.DataFrame) -> str:
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Default: first positional arg is the question
            if key_func is not None:
                question = key_func(*args, **kwargs)
            elif args:
                question = str(args[0])
            else:
                return func(*args, **kwargs)

            cached_result = cache.get(question)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            cache.set(question, result)
            return result

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Convenience: module-level singleton helpers
# ---------------------------------------------------------------------------

_default_cache: Optional[ResultCache] = None


def get_default_cache() -> ResultCache:
    """Return a lazily-initialised module-level default cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = ResultCache()
        _default_cache.load()
    return _default_cache


def configure_default_cache(
    cache_file: Optional[str] = None,
    data_files: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> ResultCache:
    """Reconfigure and return the module-level default cache."""
    global _default_cache
    cfg = CacheConfig()
    if cache_file is not None:
        cfg.cache_file = cache_file
    if data_files is not None:
        cfg.data_files = list(data_files)
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    _default_cache = ResultCache(cfg)
    _default_cache.load()
    return _default_cache
