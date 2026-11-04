"""Tests for the server lifespan (startup and shutdown).

Tests cover:
- The lifespan yields without waiting for the embedding model
- A failed model preload is logged, not raised
- No preload for a remote embedding backend
- The startup migration runs only after a successful database warm-up

No database and no model are needed: the pool, the migration and the model
load are patched.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from stratakb import mcp_server


def _fake_pool() -> MagicMock:
    """A pool whose acquire() yields a connection with an async fetchval."""
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=1)

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    return pool


@pytest.fixture
def db_ok():
    """Patch the database steps of the lifespan with a reachable fake pool."""
    with (
        patch("stratakb.mcp_server.get_pool", AsyncMock(return_value=_fake_pool())) as get_pool,
        patch("stratakb.mcp_server.ensure_schema", AsyncMock()) as migrate,
        patch("stratakb.mcp_server.close_pool", AsyncMock()),
    ):
        yield get_pool, migrate


@pytest.fixture
def local_backend(monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "local")


async def test_lifespan_does_not_wait_for_embedding_model(db_ok, local_backend):
    started = threading.Event()
    release = threading.Event()

    def slow_preload():
        started.set()
        release.wait(10)

    try:
        with patch("stratakb.mcp_server._preload_embedding_model_sync", slow_preload):
            t0 = time.monotonic()
            async with mcp_server._lifespan(mcp_server.mcp):
                assert time.monotonic() - t0 < 1.0
                assert started.wait(1.0)
                assert not release.is_set()
    finally:
        release.set()


async def test_preload_failure_is_logged(db_ok, local_backend):
    done = threading.Event()

    def failing_load(model_name):
        done.set()
        raise OSError("model not cached")

    load = MagicMock(side_effect=failing_load)
    with (
        patch("stratakb.knowledge.embeddings._get_local_model", load),
        capture_logs() as logs,
    ):
        async with mcp_server._lifespan(mcp_server.mcp):
            assert done.wait(5.0)
            # The thread logs right after the load raises
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not any(
                e["event"] == "embedding_preload_failed" for e in logs
            ):
                time.sleep(0.01)

    failed = [e for e in logs if e["event"] == "embedding_preload_failed"]
    assert len(failed) == 1
    assert "model not cached" in failed[0]["error"]


async def test_no_preload_for_openai_backend(db_ok, monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    preload = MagicMock()
    with patch("stratakb.mcp_server._preload_embedding_model_sync", preload):
        async with mcp_server._lifespan(mcp_server.mcp):
            pass
    preload.assert_not_called()


async def test_migration_skipped_when_database_unreachable(monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    with (
        patch("stratakb.mcp_server.get_pool", AsyncMock(side_effect=OSError("refused"))),
        patch("stratakb.mcp_server.ensure_schema", AsyncMock()) as migrate,
        patch("stratakb.mcp_server.close_pool", AsyncMock()),
        capture_logs() as logs,
    ):
        async with mcp_server._lifespan(mcp_server.mcp):
            pass

    migrate.assert_not_called()
    assert any(e["event"] == "db_warmup_failed" for e in logs)


async def test_migration_runs_after_warmup(db_ok, monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    _, migrate = db_ok
    async with mcp_server._lifespan(mcp_server.mcp):
        pass
    migrate.assert_awaited_once_with(mcp_server._settings.database_url)
