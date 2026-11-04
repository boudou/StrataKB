import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from stratakb.rag.retriever import (
    _build_hybrid_sql,
    _fuse_rrf,
    _hybrid_search,
    _reserve_keyword_slots,
)


def test_rrf_fusion_combines_scores():
    """RRF should combine semantic and keyword ranks into a single score."""
    semantic = [("id1", 1), ("id2", 2), ("id3", 3)]
    keyword = [("id2", 1), ("id4", 2), ("id1", 3)]

    fused = _fuse_rrf(semantic, keyword, k=60, sem_weight=0.7, kw_weight=0.3)

    ids = [f[0] for f in fused]
    # id1: sem rank 1 (0.7/61) + kw rank 3 (0.3/63) = highest combined
    # id2: sem rank 2 (0.7/62) + kw rank 1 (0.3/61) = close second
    assert ids[0] == "id1", f"Expected id1 first, got {ids[0]}"
    assert ids[1] == "id2", f"Expected id2 second, got {ids[1]}"
    assert set(ids) == {"id1", "id2", "id3", "id4"}


def test_rrf_fusion_semantic_only():
    """When keyword returns nothing, semantic results should still work."""
    semantic = [("id1", 1), ("id2", 2)]
    keyword = []
    fused = _fuse_rrf(semantic, keyword, k=60, sem_weight=0.7, kw_weight=0.3)
    assert len(fused) == 2
    assert fused[0][0] == "id1"


def test_build_hybrid_sql_includes_fulltext():
    """Hybrid SQL should contain both tsv and embedding search CTEs."""
    sql, params = _build_hybrid_sql(
        query_vector="[1.0,2.0]",
        query_text="SEPA rejection AC01",
        layers=["sepa"],
        top_k=5,
    )
    assert "tsv" in sql
    assert "embedding" in sql
    assert "rrf" in sql.lower() or "UNION ALL" in sql


# ==========================================
# Reserved keyword slots
# ==========================================

_FUSED = [("a", 0.05), ("b", 0.04), ("c", 0.03), ("d", 0.02), ("k1", 0.005), ("k2", 0.004)]


def test_reserve_keyword_slots_missing_keyword_replaces_lowest():
    kept = _reserve_keyword_slots(_FUSED, [("k1", 1), ("k2", 2)], top_k=3, slots=1)
    assert kept == [("a", 0.05), ("b", 0.04), ("k1", 0.005)]


def test_reserve_keyword_slots_present_keyword_leaves_results():
    kept = _reserve_keyword_slots(_FUSED, [("b", 1), ("k1", 2)], top_k=3, slots=1)
    assert kept == _FUSED[:3]


def test_reserve_keyword_slots_zero_slots():
    assert _reserve_keyword_slots(_FUSED, [("k1", 1)], top_k=3, slots=0) == _FUSED[:3]


def test_reserve_keyword_slots_top_k_one_keeps_fused_first():
    assert _reserve_keyword_slots(_FUSED, [("k1", 1)], top_k=1, slots=1) == _FUSED[:1]


def test_reserve_keyword_slots_two_missing_keep_fused_first():
    kept = _reserve_keyword_slots(_FUSED, [("k1", 1), ("k2", 2)], top_k=3, slots=2)
    assert kept == [("a", 0.05), ("k1", 0.005), ("k2", 0.004)]


def test_reserve_keyword_slots_protected_entry_is_not_dropped():
    # c is keyword #1 and already the lowest kept entry; adding k1 must drop b instead
    kept = _reserve_keyword_slots(_FUSED, [("c", 1), ("k1", 2)], top_k=3, slots=2)
    assert kept == [("a", 0.05), ("c", 0.03), ("k1", 0.005)]


def test_reserve_keyword_slots_fewer_fused_than_top_k():
    fused = [("a", 0.05), ("k1", 0.005)]
    assert _reserve_keyword_slots(fused, [("k1", 1)], top_k=5, slots=1) == fused


def test_reserve_keyword_slots_negative_slots_act_as_zero():
    assert _reserve_keyword_slots(_FUSED, [("k1", 1)], top_k=3, slots=-2) == _FUSED[:3]


# ==========================================
# Search queries with a mocked pool
# ==========================================


def _mock_pool(conn):
    tx = AsyncMock()
    tx.__aenter__.return_value = None
    tx.__aexit__.return_value = False
    conn.transaction = MagicMock(return_value=tx)
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn
    ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = ctx
    return pool


def _chunk_row(chunk_id):
    return {
        "id": uuid.UUID(chunk_id), "content": "text", "layer_name": "docs",
        "source_file": "f.md", "source_section": "", "doc_type": "", "metadata": {},
    }


@pytest.mark.asyncio
async def test_hybrid_search_sets_statement_timeout_and_logs_keyword_only():
    sem = [str(uuid.uuid4()) for _ in range(3)]
    kw = str(uuid.uuid4())
    hybrid_rows = [{"source": "semantic", "id": cid, "rank": i + 1} for i, cid in enumerate(sem)]
    hybrid_rows.append({"source": "fulltext", "id": kw, "rank": 1})
    conn = AsyncMock()
    conn.fetch.side_effect = [hybrid_rows, [_chunk_row(sem[0]), _chunk_row(kw)]]
    settings = SimpleNamespace(
        hnsw_ef_search=40, hybrid_rrf_k=60, hybrid_semantic_weight=0.7,
        hybrid_keyword_weight=0.3, hybrid_keyword_slots=1,
    )

    with capture_logs() as logs:
        results = await _hybrid_search(
            _mock_pool(conn), "AC04", "[1.0]", ["docs"], 2, None, settings
        )

    statements = [c.args[0] for c in conn.execute.call_args_list]
    assert any(s.startswith("SET LOCAL statement_timeout") for s in statements)
    assert [r.chunk_id for r in results] == [sem[0], kw]
    done = [e for e in logs if e["event"] == "hybrid_search_complete"]
    assert done[0]["keyword_only_results"] == 1


@pytest.mark.asyncio
async def test_vector_search_sets_statement_timeout():
    from stratakb.rag.retriever import search_chunks

    conn = AsyncMock()
    conn.fetch.return_value = []
    settings = MagicMock(hybrid_search_enabled=False, hnsw_ef_search=40, cache_enabled=False)

    with patch(
        "stratakb.rag.retriever.embed_texts", new_callable=AsyncMock, return_value=[[0.01] * 768]
    ), patch(
        "stratakb.rag.retriever.get_pool", new_callable=AsyncMock, return_value=_mock_pool(conn)
    ), patch(
        "stratakb.rag.retriever.fetch_layers_stamp", new_callable=AsyncMock, return_value=(0, None)
    ), patch("stratakb.config.get_settings", return_value=settings):
        await search_chunks("postgresql://test", "sk-test", "timeout probe")

    statements = [c.args[0] for c in conn.execute.call_args_list]
    assert any(s.startswith("SET LOCAL statement_timeout") for s in statements)
