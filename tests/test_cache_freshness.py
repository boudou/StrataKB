"""Tests for cache freshness: the query and routing caches clear when the layers change.

All tests need the database; embeddings are mocked and each owner's watcher
gets a test clock, so time.monotonic is never patched.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
from structlog.testing import capture_logs

from stratakb.knowledge import layers as layers_mod
from stratakb.rag import retriever
from stratakb.rag.cache import QueryCache
from tests.helpers import FAKE_EMBEDDING, fake_embed, ingest

SENTINEL_RESULTS = ["cached-result"]
SENTINEL_ROUTING = {"patterns": {}, "descriptions": {"sentinel": "Sentinel"}}


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    """Fresh caches and stamp state in both modules, before and after each test."""

    def reset():
        retriever._watcher.reset()
        layers_mod._watcher.reset()
        layers_mod._routing_cache.clear()

    reset()
    monkeypatch.setattr(retriever, "_cache", QueryCache(maxsize=10, ttl=600))
    yield
    reset()


@pytest.fixture
def clock():
    """Give both watchers a test clock; set clock.value to move time."""
    state = MagicMock()
    state.value = 100.0
    with (
        patch.object(retriever._watcher, "clock", lambda: state.value),
        patch.object(layers_mod._watcher, "clock", lambda: state.value),
    ):
        yield state


def _fill_caches() -> None:
    retriever._get_cache().put("q", None, 5, SENTINEL_RESULTS)
    layers_mod._routing_cache["routing"] = SENTINEL_ROUTING


def _caches_empty() -> bool:
    return (
        retriever._get_cache().get("q", None, 5) is None
        and "routing" not in layers_mod._routing_cache
    )


async def _check_both(db_url: str) -> None:
    await retriever._watcher.check(db_url)
    await layers_mod._watcher.check(db_url)


async def _execute_elsewhere(db_url: str, sql: str, *args) -> None:
    """Run a statement over its own connection, as another process would."""
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql, *args)
    finally:
        await conn.close()


async def _layer_row(db_url: str, layer: str):
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetchrow(
        "SELECT doc_count, chunk_count, updated_at FROM strata_core.knowledge_layers "
        "WHERE name = $1",
        layer,
    )


async def _chunk_count(db_url: str, layer: str) -> int:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetchval(
        "SELECT count(*) FROM strata_vectors.chunks WHERE layer_name = $1", layer
    )


def _write_files(root: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (root / f"doc{i}.md").write_text(
            f"# Doc {i}\n\nSome text about topic {i}.\n", encoding="utf-8"
        )


async def test_stamp_change_clears_both_caches(db_url, layer, clock):
    _fill_caches()
    await _check_both(db_url)  # first read only records the stamp
    assert not _caches_empty()

    await _execute_elsewhere(
        db_url,
        "UPDATE strata_core.knowledge_layers SET updated_at = now() WHERE name = $1",
        layer,
    )
    clock.value = 103.0
    await _check_both(db_url)
    assert _caches_empty()


async def test_search_and_routing_call_the_check(db_url, layer, clock):
    from stratakb.knowledge.layers import get_layer_routing
    from stratakb.rag.retriever import search_chunks

    _fill_caches()
    with patch(
        "stratakb.rag.retriever.embed_texts",
        new_callable=AsyncMock,
        return_value=[FAKE_EMBEDDING],
    ) as embed:
        assert await search_chunks(db_url, "", "q") == SENTINEL_RESULTS
        assert await get_layer_routing(db_url) == SENTINEL_ROUTING
        embed.assert_not_awaited()

        await _execute_elsewhere(
            db_url,
            "UPDATE strata_core.knowledge_layers SET updated_at = now() WHERE name = $1",
            layer,
        )
        clock.value = 103.0
        results = await search_chunks(db_url, "", "q")
        routing = await get_layer_routing(db_url)

    assert results != SENTINEL_RESULTS
    embed.assert_awaited_once()
    assert layer in routing["descriptions"]
    assert "sentinel" not in routing["descriptions"]


async def test_no_stamp_read_within_two_seconds(db_url, layer, clock):
    from stratakb.core.database import fetch_layers_stamp

    with (
        patch.object(
            retriever._watcher, "fetch", new_callable=AsyncMock, wraps=fetch_layers_stamp
        ) as r_stamp,
        patch.object(
            layers_mod._watcher, "fetch", new_callable=AsyncMock, wraps=fetch_layers_stamp
        ) as l_stamp,
    ):
        await _check_both(db_url)
        clock.value = 101.5
        await _check_both(db_url)
        assert r_stamp.await_count == 1
        assert l_stamp.await_count == 1

        clock.value = 102.1
        await _check_both(db_url)
        assert r_stamp.await_count == 2
        assert l_stamp.await_count == 2


async def test_layer_delete_clears_both_caches(db_url, layer, clock):
    _fill_caches()
    await _check_both(db_url)

    await _execute_elsewhere(
        db_url, "DELETE FROM strata_core.knowledge_layers WHERE name = $1", layer
    )
    clock.value = 102.5
    await _check_both(db_url)
    assert _caches_empty()


async def test_interrupted_ingest_bumps_updated_at(db_url, layer, tmp_path):
    from stratakb.knowledge.ingestion import ingest_documents

    _write_files(tmp_path / "docs", 2)
    before = (await _layer_row(db_url, layer))["updated_at"]
    calls = 0

    async def embed(texts, **_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise asyncio.CancelledError
        return fake_embed(texts)

    with (
        patch("stratakb.knowledge.ingestion.embed_texts", side_effect=embed),
        pytest.raises(asyncio.CancelledError),
    ):
        await ingest_documents(
            db_url, "fake-key", layer_name=layer, source_path=str(tmp_path / "docs")
        )

    row = await _layer_row(db_url, layer)
    assert row["updated_at"] > before
    stored = await _chunk_count(db_url, layer)
    assert stored > 0
    assert row["chunk_count"] == stored
    assert row["doc_count"] == 1


async def test_failed_record_run_bumps_updated_at(db_url, layer, tmp_path):
    _write_files(tmp_path / "docs", 2)
    before = (await _layer_row(db_url, layer))["updated_at"]

    with (
        patch(
            "stratakb.knowledge.ingestion._record_run",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await ingest(db_url, layer, tmp_path / "docs")

    row = await _layer_row(db_url, layer)
    assert row["updated_at"] > before
    assert row["chunk_count"] == await _chunk_count(db_url, layer)
    assert row["doc_count"] == 2


async def test_clear_layer_recounts(db_url, layer, tmp_path):
    from stratakb.knowledge.ingestion import clear_layer

    _write_files(tmp_path / "docs", 1)
    await ingest(db_url, layer, tmp_path / "docs")
    before = await _layer_row(db_url, layer)
    assert before["chunk_count"] > 0

    await clear_layer(db_url, layer)

    after = await _layer_row(db_url, layer)
    assert after["doc_count"] == 0
    assert after["chunk_count"] == 0
    assert after["updated_at"] > before["updated_at"]


async def test_failed_stamp_read_serves_cache(db_url, layer, clock):
    from stratakb.knowledge.layers import get_layer_routing
    from stratakb.rag.retriever import search_chunks

    await _check_both(db_url)
    _fill_caches()
    clock.value = 103.0

    with (
        patch.object(
            retriever._watcher, "fetch", new_callable=AsyncMock, side_effect=OSError("down")
        ),
        patch.object(
            layers_mod._watcher, "fetch", new_callable=AsyncMock, side_effect=OSError("down")
        ),
        patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            side_effect=AssertionError("embedder must not be called"),
        ),
        capture_logs() as logs,
    ):
        assert await search_chunks(db_url, "", "q") == SENTINEL_RESULTS
        assert await get_layer_routing(db_url) == SENTINEL_ROUTING

    query_warnings = [e for e in logs if e["event"] == "query_cache_stamp_failed"]
    routing_warnings = [e for e in logs if e["event"] == "routing_cache_stamp_failed"]
    assert len(query_warnings) == 1
    assert len(routing_warnings) == 1
    for entry in query_warnings + routing_warnings:
        assert entry["log_level"] == "warning"
        assert entry["error_type"] == "OSError"


async def test_stats_update_failure_is_logged(db_url, layer):
    from stratakb.knowledge.ingestion import _update_layer_stats

    pool = MagicMock()
    pool.execute = AsyncMock(side_effect=OSError("down"))
    with (
        patch("stratakb.knowledge.ingestion.get_pool", new_callable=AsyncMock, return_value=pool),
        capture_logs() as logs,
    ):
        await _update_layer_stats(db_url, layer)

    events = [e for e in logs if e["event"] == "layer_stats_update_failed"]
    assert len(events) == 1
    assert events[0]["layer"] == layer
    assert events[0]["error_type"] == "OSError"
