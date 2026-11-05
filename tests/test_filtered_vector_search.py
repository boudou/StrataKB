"""Vector search through the HNSW index returns every row it asks for (requires DB).

Small test tables get a sequential scan, so these tests force the HNSW plan
with a pool that turns sequential scans and sorts off. Without iterative
index scans, the index hands back at most hnsw.ef_search candidates and a
filter applied afterwards can leave none of them.
"""

import random
import uuid
from unittest.mock import AsyncMock

import asyncpg
import pytest

from stratakb.config import get_settings
from stratakb.core.database import get_pool
from stratakb.rag import retriever
from stratakb.rag.retriever import _build_hybrid_sql, _search_fetch, search_chunks

_LARGE = 280
_SMALL = 20


def _vec(values: list[float]) -> str:
    return "[" + ",".join(str(x) for x in values) + "]"


def _near(axis: int, dims: int, rng: random.Random) -> list[float]:
    v = [rng.uniform(-0.05, 0.05) for _ in range(dims)]
    v[axis] = 1.0
    return v


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 1 - dot / (na * nb)


@pytest.fixture
async def two_layers(vacuumed_chunks):
    """A large layer near the query and a small layer near an orthogonal axis."""
    settings = get_settings()
    dims = settings.embedding_dimensions
    rng = random.Random(12)
    suffix = uuid.uuid4().hex[:8]
    large, small = f"hnswbig_{suffix}", f"hnswsmall_{suffix}"
    query = [0.0] * dims
    query[0] = 1.0
    vectors: dict[str, list[float]] = {}
    pool = await get_pool(settings.database_url)
    insert = """
        INSERT INTO strata_vectors.chunks (layer_name, content, embedding, source_file)
        VALUES ($1, $2, $3::vector, $4) RETURNING id
    """
    try:
        async with pool.acquire() as conn:
            for layer, axis, count in ((large, 0, _LARGE), (small, 1, _SMALL)):
                for i in range(count):
                    v = _near(axis, dims, rng)
                    cid = await conn.fetchval(
                        insert, layer, f"Note {i} in {layer}.", _vec(v), f"{layer}_{i}"
                    )
                    vectors[str(cid)] = v
        yield {"large": large, "small": small, "query": query, "vectors": vectors}
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM strata_vectors.chunks WHERE layer_name = ANY($1::text[])",
                [large, small],
            )


@pytest.fixture
async def forced_pool():
    """A pool whose planner takes the HNSW index scan for ORDER BY ... LIMIT."""
    pool = await asyncpg.create_pool(
        get_settings().database_url, min_size=1, max_size=2,
        server_settings={"enable_seqscan": "off", "enable_sort": "off"},
    )
    try:
        yield pool
    finally:
        await pool.close()


def _semantic(rows) -> list:
    return [r for r in rows if r["source"] == "semantic"]


@pytest.mark.asyncio
async def test_forced_pool_plans_the_hnsw_index(two_layers, forced_pool):
    """Guard: without the index in the plan the other tests would pass on old code too."""
    sql, params = _build_hybrid_sql(
        _vec(two_layers["query"]), "zzqx", [two_layers["small"]], top_k=5
    )
    async with forced_pool.acquire() as conn:
        plan = await conn.fetch("EXPLAIN " + sql, *params)
    text = "\n".join(r[0] for r in plan)
    assert "idx_chunks_embedding_hnsw" in text


@pytest.mark.asyncio
async def test_filtered_semantic_leg_fills_its_limit(two_layers, forced_pool):
    settings = get_settings()
    small = two_layers["small"]
    sql, params = _build_hybrid_sql(_vec(two_layers["query"]), "zzqx", [small], top_k=5)

    rows = _semantic(await _search_fetch(forced_pool, settings, sql, params))

    assert len(rows) == 15
    assert all(str(r["id"]) in two_layers["vectors"] for r in rows)
    async with forced_pool.acquire() as conn:
        layers = await conn.fetch(
            "SELECT DISTINCT layer_name FROM strata_vectors.chunks WHERE id = ANY($1::uuid[])",
            [r["id"] for r in rows],
        )
    assert [r["layer_name"] for r in layers] == [small]


@pytest.mark.asyncio
async def test_unfiltered_semantic_leg_passes_ef_search(two_layers, forced_pool):
    settings = get_settings()
    sql, params = _build_hybrid_sql(_vec(two_layers["query"]), "zzqx", None, top_k=50)

    rows = _semantic(await _search_fetch(forced_pool, settings, sql, params))

    assert len(rows) == 150


@pytest.mark.asyncio
async def test_semantic_ranks_follow_distance(two_layers, forced_pool):
    settings = get_settings()
    query, vectors = two_layers["query"], two_layers["vectors"]
    sql, params = _build_hybrid_sql(
        _vec(query), "zzqx", [two_layers["large"], two_layers["small"]], top_k=50
    )

    rows = sorted(
        _semantic(await _search_fetch(forced_pool, settings, sql, params)),
        key=lambda r: r["rank"],
    )

    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    distances = [_cosine_distance(vectors[str(r["id"])], query) for r in rows]
    assert all(a <= b + 1e-6 for a, b in zip(distances, distances[1:], strict=False))


@pytest.mark.asyncio
async def test_vector_only_search_fills_top_k_in_order(two_layers, forced_pool, monkeypatch):
    settings = get_settings()
    vector_only = settings.model_copy(update={"hybrid_search_enabled": False})
    small = two_layers["small"]

    async def _forced(_url):
        return forced_pool

    monkeypatch.setattr(retriever, "get_pool", _forced)
    monkeypatch.setattr(
        retriever, "embed_texts", AsyncMock(return_value=[two_layers["query"]])
    )
    monkeypatch.setattr("stratakb.config.get_settings", lambda: vector_only)
    monkeypatch.setattr(retriever, "_get_cache", lambda: None)

    results = await search_chunks(
        settings.database_url, None, "orthogonal notes", layers=[small], top_k=10
    )

    assert len(results) == 10
    assert {r.layer_name for r in results} == {small}
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
