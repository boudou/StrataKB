"""Tests for the promotion/pattern mining engine."""

import json

import pytest
from unittest.mock import AsyncMock, patch
from stratakb.learning.promoter import Promoter
from stratakb.learning.store import LearningStore


@pytest.fixture
def promoter():
    store = LearningStore(database_url="postgresql://test:test@localhost/test")
    return Promoter(store)


@pytest.mark.asyncio
async def test_decay_reduces_success_rate(promoter):
    """Stale rewrites should have their success_rate decayed."""
    with patch.object(
        promoter.store, "_execute", new_callable=AsyncMock, return_value="UPDATE 3"
    ) as mock_exec:
        decayed = await promoter.decay_stale_rewrites(unused_days=60, decay_factor=0.9)
        mock_exec.assert_called_once()
        sql = mock_exec.call_args[0][0]
        assert "success_rate" in sql
        assert "0.9" in sql or "decay_factor" in sql.lower() or "$" in sql
        assert decayed == 3


@pytest.mark.asyncio
async def test_decay_count_zero_when_store_unavailable(promoter):
    with patch.object(promoter.store, "_execute", new_callable=AsyncMock, return_value=None):
        assert await promoter.decay_stale_rewrites() == 0


@pytest.mark.asyncio
async def test_promote_rewrites_counts_evidence(promoter):
    """The most common rewrite wins; total_count is how often the question was asked."""
    rewrite = ["sepa rejection reason codes", "pacs.002 reject status"]
    evidence = [
        {"question": "Why was my payment rejected?", "rewritten_queries": json.dumps(rewrite),
         "fallback_query": None, "layers": ["sepa"]},
        {"question": "why was my payment rejected? ", "rewritten_queries": json.dumps(rewrite),
         "fallback_query": None, "layers": ["sepa"]},
        # A rewrite that equals the question teaches nothing
        {"question": "AC04", "rewritten_queries": json.dumps(["AC04"]),
         "fallback_query": None, "layers": None},
    ]
    asked = [{"pattern": "why was my payment rejected?", "n": 4}]

    fetch = AsyncMock(side_effect=[evidence, asked])
    upsert = AsyncMock()
    with patch.object(promoter.store, "_fetch", fetch), \
         patch.object(promoter.store, "upsert_learned_rewrite", upsert):
        promoted = await promoter.promote_rewrites()

    assert promoted == 1
    kwargs = upsert.call_args.kwargs
    assert kwargs["pattern"] == "why was my payment rejected?"
    assert json.loads(kwargs["optimized_rewrite"]) == rewrite
    assert kwargs["success_count"] == 2
    assert kwargs["total_count"] == 4


@pytest.mark.asyncio
async def test_promote_rewrites_uses_reformulation_when_not_rewritten(promoter):
    evidence = [
        {"question": "payment bounced", "rewritten_queries": None,
         "fallback_query": "SCT reject reason codes", "layers": ["sepa"]},
    ]
    fetch = AsyncMock(side_effect=[evidence, []])
    upsert = AsyncMock()
    with patch.object(promoter.store, "_fetch", fetch), \
         patch.object(promoter.store, "upsert_learned_rewrite", upsert):
        await promoter.promote_rewrites()

    kwargs = upsert.call_args.kwargs
    assert json.loads(kwargs["optimized_rewrite"]) == ["SCT reject reason codes"]
    assert kwargs["total_count"] == 1


@pytest.mark.asyncio
async def test_layer_affinities_from_corrections(promoter):
    question = "settlement cut-off times"
    rows = [
        {"question": question.title(), "right_layers": ["sepa"], "wrong_layers": ["iso20022"]},
        {"question": question, "right_layers": ["sepa"], "wrong_layers": ["iso20022", "sepa"]},
    ]
    upsert = AsyncMock()
    with patch.object(promoter.store, "_fetch", AsyncMock(return_value=rows)), \
         patch.object(promoter.store, "upsert_layer_affinity", upsert):
        updated = await promoter.update_layer_affinities()

    assert updated == 2
    calls = [c.kwargs for c in upsert.call_args_list]
    scores = {c["layer_name"]: c["affinity_score"] for c in calls}
    assert scores["sepa"] == pytest.approx(3 / 4)       # 2 right, 0 wrong
    assert scores["iso20022"] == pytest.approx(1 / 4)   # 0 right, 2 wrong
    assert {c["query_pattern"] for c in calls} == {question}
