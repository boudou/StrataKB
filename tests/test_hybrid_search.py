import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from stratakb.rag.retriever import (
    _build_hybrid_sql,
    _filter_conditions,
    _fuse_rrf,
    _hybrid_search,
    _reserve_keyword_slots,
    _search_fetch,
    _supports_iterative_scan,
    fuse_rrf,
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


def _chunk_row(chunk_id, similarity=0.5):
    return {
        "id": uuid.UUID(chunk_id), "content": "text", "layer_name": "docs",
        "source_file": "f.md", "source_section": "", "doc_type": "", "metadata": {},
        "similarity": similarity,
    }


@pytest.mark.asyncio
async def test_hybrid_search_sets_statement_timeout_and_logs_keyword_only():
    sem = [str(uuid.uuid4()) for _ in range(3)]
    kw = str(uuid.uuid4())
    hybrid_rows = [{"source": "semantic", "id": cid, "rank": i + 1} for i, cid in enumerate(sem)]
    hybrid_rows.append({"source": "fulltext", "id": kw, "rank": 1})
    conn = AsyncMock()
    conn.fetch.side_effect = [
        hybrid_rows, [_chunk_row(sem[0], similarity=0.73), _chunk_row(kw, similarity=None)],
    ]
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
    # The chunk query computes the cosine similarity against the query vector
    assert conn.fetch.call_args_list[1].args[2] == "[1.0]"
    assert results[0].similarity == 0.73
    assert results[1].similarity is None  # a chunk without an embedding
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


@pytest.mark.asyncio
async def test_search_fetch_sets_search_settings_in_one_transaction():
    conn = AsyncMock()
    conn.fetch.return_value = ["row"]

    rows = await _search_fetch(
        _mock_pool(conn), SimpleNamespace(hnsw_ef_search=40), "SELECT 1", [7]
    )

    statements = [c.args[0] for c in conn.execute.call_args_list]
    assert statements == [
        "SET LOCAL hnsw.ef_search = 40",
        "SET LOCAL statement_timeout = '10s'",
    ]
    conn.transaction.assert_called_once()
    assert conn.fetch.call_args.args == ("SELECT 1", 7)
    assert rows == ["row"]


@pytest.mark.asyncio
async def test_search_fetch_turns_on_iterative_scan_on_pgvector_08():
    conn = AsyncMock()
    conn.fetch.return_value = []
    conn.fetchval.return_value = "0.8.0"

    await _search_fetch(_mock_pool(conn), SimpleNamespace(hnsw_ef_search=40), "SELECT 1", [])

    statements = [c.args[0] for c in conn.execute.call_args_list]
    assert statements == [
        "SET LOCAL hnsw.ef_search = 40",
        "SET LOCAL statement_timeout = '10s'",
        "SET LOCAL hnsw.iterative_scan = relaxed_order",
    ]


@pytest.mark.asyncio
async def test_search_fetch_skips_iterative_scan_on_old_pgvector_and_warns_once(monkeypatch):
    monkeypatch.setattr("stratakb.rag.retriever._iterative_scan_warned", False)
    conn = AsyncMock()
    conn.fetch.return_value = []
    conn.fetchval.return_value = "0.7.4"
    settings = SimpleNamespace(hnsw_ef_search=40)

    with capture_logs() as logs:
        await _search_fetch(_mock_pool(conn), settings, "SELECT 1", [])
        await _search_fetch(_mock_pool(conn), settings, "SELECT 1", [])

    statements = [c.args[0] for c in conn.execute.call_args_list]
    assert not any("iterative_scan" in s for s in statements)
    assert len(statements) == 4
    warned = [e for e in logs if e["event"] == "hnsw_iterative_scan_unavailable"]
    assert len(warned) == 1
    assert warned[0]["pgvector_version"] == "0.7.4"


@pytest.mark.parametrize("version", ["0.8.0", "0.8.7", "1.0.0"])
def test_supports_iterative_scan_from_08(version):
    assert _supports_iterative_scan(version)


@pytest.mark.parametrize("version", ["0.7.4", "0.5", None, MagicMock()])
def test_supports_iterative_scan_rejects_older_or_unknown(version):
    assert not _supports_iterative_scan(version)


def test_filter_conditions_numbers_from_start_idx():
    assert _filter_conditions(["a", "b"], {"doc_type": "x"}, 3) == (
        ["layer_name IN ($3, $4)", "metadata->>'doc_type' = $5"],
        ["a", "b", "x"],
    )


def test_hybrid_sql_tsquery_follows_filter_params():
    sql, params = _build_hybrid_sql("[0.1]", "settlement", ["sepa"], 5, {"doc_type": "rule"})

    assert "plainto_tsquery('english', $5)" in sql
    assert params == ["[0.1]", 15, "sepa", "rule", "settlement"]


def test_hybrid_sql_ranks_semantic_leg_with_a_real_sort():
    sql, _ = _build_hybrid_sql("[0.1]", "settlement", None, 5)

    assert "ORDER BY distance + 0, id" in sql


@pytest.mark.asyncio
async def test_hybrid_search_parses_json_string_metadata():
    cid = str(uuid.uuid4())
    row = _chunk_row(cid)
    row["metadata"] = json.dumps({"k": "v"})
    row["source_section"] = None
    conn = AsyncMock()
    conn.fetch.side_effect = [[{"source": "semantic", "id": cid, "rank": 1}], [row]]
    settings = SimpleNamespace(
        hnsw_ef_search=40, hybrid_rrf_k=60, hybrid_semantic_weight=0.7,
        hybrid_keyword_weight=0.3, hybrid_keyword_slots=1,
    )

    results = await _hybrid_search(_mock_pool(conn), "q", "[1.0]", None, 3, None, settings)

    assert results[0].metadata == {"k": "v"}
    assert results[0].source_section == ""


# ==========================================
# Cosine similarity on real rows (requires DB)
# ==========================================


@pytest.mark.asyncio
async def test_hybrid_results_carry_cosine_similarity(monkeypatch, vacuumed_chunks):
    """The fused score is normalized to a top of 1.0; similarity stays the real cosine."""
    from stratakb.config import get_settings
    from stratakb.core.database import get_pool
    from stratakb.rag.retriever import search_chunks

    settings = get_settings()
    pool = await get_pool(settings.database_url)
    layer = f"hybsim_{uuid.uuid4().hex[:8]}"
    h = settings.embedding_dimensions // 2
    close = [1.0] * h + [0.0] * (settings.embedding_dimensions - h)
    orthogonal = [0.0] * h + [1.0] * (settings.embedding_dimensions - h)
    close_vector = "[" + ",".join(str(x) for x in close) + "]"
    orthogonal_vector = "[" + ",".join(str(x) for x in orthogonal) + "]"
    insert = """
        INSERT INTO strata_vectors.chunks (layer_name, content, embedding, source_file)
        VALUES ($1, $2, $3::vector, $4) RETURNING id
    """
    monkeypatch.setattr("stratakb.rag.retriever._get_cache", lambda: None)
    try:
        async with pool.acquire() as conn:
            for i in range(8):
                await conn.execute(
                    insert, layer, f"Note {i} on payment fees and reporting.",
                    close_vector, f"close_{i}",
                )
            ac04 = str(await conn.fetchval(
                insert, layer, "Reason code AC04 means the account is closed.",
                orthogonal_vector, "ac04",
            ))

        with patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            return_value=[close],
        ):
            results = await search_chunks(
                settings.database_url, None, "AC04", layers=[layer], top_k=2
            )

        assert results[0].score == 1.0
        assert abs(results[0].similarity - 1.0) < 1e-3
        keyword_only = next(r for r in results if r.chunk_id == ac04)
        assert abs(keyword_only.similarity) < 1e-3
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM strata_vectors.chunks WHERE layer_name = $1", layer)


# ==========================================
# Chunks without an embedding
# ==========================================


def test_semantic_leg_skips_chunks_without_embedding():
    """Only the semantic CTE leaves out NULL embeddings; the full-text CTE keeps them."""
    sql, _ = _build_hybrid_sql("[0.1,0.2]", "settlement", ["sepa"], top_k=3)
    semantic, fulltext = sql.split("fulltext AS (", 1)
    assert "embedding IS NOT NULL" in semantic.split("semantic AS (", 1)[1]
    assert "embedding IS NOT NULL" not in fulltext


@pytest.mark.asyncio
async def test_hybrid_still_finds_chunk_without_embedding(monkeypatch, vacuumed_chunks):
    """A chunk with a NULL embedding is still returned through the keyword leg."""
    from stratakb.config import get_settings
    from stratakb.core.database import get_pool
    from stratakb.rag.retriever import search_chunks

    settings = get_settings()
    pool = await get_pool(settings.database_url)
    layer = f"nullemb_{uuid.uuid4().hex[:8]}"
    query = [0.1] * settings.embedding_dimensions
    vector = "[" + ",".join(str(x) for x in query) + "]"
    insert = """
        INSERT INTO strata_vectors.chunks (layer_name, content, embedding, source_file)
        VALUES ($1, $2, $3::vector, $4) RETURNING id
    """
    monkeypatch.setattr("stratakb.rag.retriever._get_cache", lambda: None)
    try:
        async with pool.acquire() as conn:
            for i in range(3):
                await conn.execute(
                    insert, layer, f"Note {i} on payment fees and reporting.", vector, f"n_{i}",
                )
            ac04 = str(await conn.fetchval(
                insert, layer, "Reason code AC04 means the account is closed.", None, "ac04",
            ))

        with patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            return_value=[query],
        ):
            results = await search_chunks(
                settings.database_url, None, "AC04", layers=[layer], top_k=5
            )

        found = next(r for r in results if r.chunk_id == ac04)
        assert found.similarity is None
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM strata_vectors.chunks WHERE layer_name = $1", layer)


def test_fuse_rrf_three_lists_unweighted():
    lists = [
        [("a", 1), ("b", 2)],
        [("c", 1), ("a", 2)],
        [("a", 3), ("c", 2)],
    ]
    fused = dict(fuse_rrf(lists, k=60))

    assert next(iter(fused)) == "a"
    assert fused["a"] == 1 / 61 + 1 / 62 + 1 / 63
    assert fused["c"] == 1 / 61 + 1 / 62
    assert fused["b"] == 1 / 62


def test_fuse_rrf_counts_repeated_id_once_per_list():
    assert fuse_rrf([[("a", 1), ("a", 2)]], k=60) == [("a", 1 / 61)]


def test_fuse_rrf_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        fuse_rrf([[("a", 1)], [("b", 1)]], k=60, weights=[1.0])


def test_fuse_rrf_matches_two_leg_wrapper():
    semantic = [("id1", 1), ("id2", 2), ("id3", 3)]
    keyword = [("id2", 1), ("id4", 2), ("id1", 3)]

    assert _fuse_rrf(semantic, keyword, k=60, sem_weight=0.7, kw_weight=0.3) == fuse_rrf(
        [semantic, keyword], k=60, weights=[0.7, 0.3]
    )


def test_fuse_rrf_ties_keep_first_seen_order():
    fused = fuse_rrf([[("x", 1)], [("y", 1)], [("z", 1)]], k=60)

    assert [cid for cid, _ in fused] == ["x", "y", "z"]
