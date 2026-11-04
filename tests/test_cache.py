import pytest
from stratakb.rag.cache import QueryCache


def test_cache_hit():
    cache = QueryCache(maxsize=10, ttl=60)
    cache.put("q1", ["sepa"], 5, [{"content": "result"}])
    result = cache.get("q1", ["sepa"], 5)
    assert result is not None
    assert result[0]["content"] == "result"


def test_cache_miss():
    cache = QueryCache(maxsize=10, ttl=60)
    result = cache.get("q1", ["sepa"], 5)
    assert result is None


def test_cache_different_layers_miss():
    cache = QueryCache(maxsize=10, ttl=60)
    cache.put("q1", ["sepa"], 5, [{"content": "result"}])
    result = cache.get("q1", ["iso20022"], 5)
    assert result is None


def test_cache_different_metadata_filter_miss():
    cache = QueryCache(maxsize=10, ttl=60)
    cache.put("q1", ["sepa"], 5, [{"content": "result"}], metadata_filter={"doc_type": "csv"})
    result = cache.get("q1", ["sepa"], 5, metadata_filter={"doc_type": "pdf"})
    assert result is None


def test_cache_stats():
    cache = QueryCache(maxsize=10, ttl=60)
    cache.get("q1", None, 5)  # miss
    cache.put("q1", None, 5, [{"content": "r"}])
    cache.get("q1", None, 5)  # hit
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_invalidate_layer():
    cache = QueryCache(maxsize=10, ttl=60)
    cache.put("q1", ["sepa"], 5, [{"content": "r"}])
    cache.put("q2", ["iso20022"], 5, [{"content": "r"}])
    cache.invalidate_layer("sepa")
    assert cache.get("q1", ["sepa"], 5) is None
    assert cache.get("q2", ["iso20022"], 5) is not None
