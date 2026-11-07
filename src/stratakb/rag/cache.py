"""Query result cache with TTL for the StrataKB RAG pipeline.

LRU + TTL cache keyed by (query, layers, top_k, metadata_filter). Cleared whole
by the retriever when the layers stamp changes (see core/stamp_watcher.py).
"""
from __future__ import annotations

from cachetools import TTLCache

import structlog

logger = structlog.get_logger()


class QueryCache:
    """In-memory query result cache with TTL expiry."""

    def __init__(self, maxsize: int = 500, ttl: int = 600):
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._hits = 0
        self._misses = 0

    def _make_key(
        self, query: str, layers: list[str] | None, top_k: int,
        metadata_filter: dict[str, str] | None = None,
    ) -> tuple:
        mf = tuple(sorted(metadata_filter.items())) if metadata_filter else ()
        return (query, tuple(sorted(layers or [])), top_k, mf)

    def get(
        self, query: str, layers: list[str] | None, top_k: int,
        metadata_filter: dict[str, str] | None = None,
    ):
        key = self._make_key(query, layers, top_k, metadata_filter)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            logger.debug("cache_hit", query=query[:50])
            return result
        self._misses += 1
        return None

    def put(
        self, query: str, layers: list[str] | None, top_k: int, results: list,
        metadata_filter: dict[str, str] | None = None,
    ):
        key = self._make_key(query, layers, top_k, metadata_filter)
        self._cache[key] = results

    def invalidate_layer(self, layer_name: str) -> int:
        """Remove all cached entries that include the given layer."""
        keys_to_remove = [
            k for k in self._cache
            if layer_name in k[1]  # k[1] is the layers tuple
        ]
        for k in keys_to_remove:
            del self._cache[k]
        return len(keys_to_remove)

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 2) if total > 0 else 0.0,
            "size": len(self._cache),
        }
