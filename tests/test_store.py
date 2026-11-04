"""Tests for learning store CRUD operations.

These are unit tests that mock the database. Integration tests
against a real database run separately.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from stratakb.learning.store import LearningStore


@pytest.fixture
def store():
    return LearningStore(database_url="postgresql://test:test@localhost/test")


@pytest.mark.asyncio
async def test_find_synonyms_returns_empty_when_no_matches(store):
    with patch.object(store, "_fetch", new_callable=AsyncMock, return_value=[]):
        result = await store.find_synonyms("unknown term")
    assert result == []


@pytest.mark.asyncio
async def test_find_rewrites_returns_none_when_no_match(store):
    with patch.object(store, "_fetch", new_callable=AsyncMock, return_value=[]):
        result = await store.find_rewrites("some query", min_success_rate=0.7)
    assert result is None


@pytest.mark.asyncio
async def test_find_synonyms_returns_matches(store):
    mock_rows = [
        {"term": "AC01", "synonym": "Incorrect Account Number", "layer": "iso20022", "confidence": 0.9}
    ]
    with patch.object(store, "_fetch", new_callable=AsyncMock, return_value=mock_rows):
        result = await store.find_synonyms("AC01")
    assert len(result) == 1
    assert result[0]["synonym"] == "Incorrect Account Number"


@pytest.mark.asyncio
async def test_apply_synonyms_enriches_query(store):
    synonyms = [
        {"term": "AC01", "synonym": "Incorrect Account Number", "layer": "iso20022", "confidence": 0.9},
        {"term": "kai", "synonym": "and", "layer": None, "confidence": 0.95},
    ]
    result = store.apply_synonyms("AC01 kai AM04", synonyms)
    assert "Incorrect Account Number" in result
    assert "AC01" in result  # original preserved too


def test_learned_rewrite_queries_from_json_array():
    from stratakb.learning.store import LearnedRewrite

    r = LearnedRewrite("q", '["a b", "c d"]', [], 0.9, 5)
    assert r.queries == ["a b", "c d"]


def test_learned_rewrite_queries_from_plain_text():
    from stratakb.learning.store import LearnedRewrite

    assert LearnedRewrite("q", "sepa reject codes", [], 0.9, 5).queries == ["sepa reject codes"]
    assert LearnedRewrite("q", "[not json", [], 0.9, 5).queries == ["[not json"]


@pytest.mark.asyncio
async def test_store_feedback_reports_failure(store):
    with patch.object(store, "_execute", new_callable=AsyncMock, return_value=None):
        assert await store.store_feedback("not-a-uuid", "explicit", "helpful", 1.0) is False
    with patch.object(store, "_execute", new_callable=AsyncMock, return_value="INSERT 0 1"):
        assert await store.store_feedback("id", "explicit", "helpful", 1.0) is True


@pytest.mark.asyncio
async def test_recent_interactions_decode_results(store):
    rows = [{"id": "i1", "results": '[{"chunk_id": "c1"}]'}]
    with patch.object(store, "_fetch", new_callable=AsyncMock, return_value=rows):
        recent = await store.get_recent_interactions("s1")
    assert recent[0]["results"] == [{"chunk_id": "c1"}]


@pytest.mark.asyncio
async def test_rewrite_bypass_returns_all_learned_queries():
    from stratakb.learning.store import LearnedRewrite
    from stratakb.rag.query_rewriter import rewrite_query

    learning_store = MagicMock()
    learning_store.find_synonyms = AsyncMock(return_value=[])
    learning_store.find_rewrites = AsyncMock(
        return_value=LearnedRewrite("q", '["a b", "c d"]', ["sepa"], 0.95, 10)
    )
    queries = await rewrite_query("q", anthropic_api_key="key", learning_store=learning_store)
    assert queries == ["a b", "c d"]
