"""Tests for the server lifespan (startup and shutdown).

Tests cover:
- The lifespan yields without waiting for the embedding model
- A failed model preload is logged, not raised
- No preload for a remote embedding backend
- The startup migration runs only after a successful database warm-up
- The SQL tools role check runs before the lifespan yields
- An unread ./.env and a missing STRATAKB_CONFIG file are logged, without the path
- The lifespan yields without waiting for the embedding checks, and cancels them on exit
- The embedding checks learn whether the database warm-up succeeded

No database and no model are needed: the pool, the migration and the model
load are patched.
"""

from __future__ import annotations

import asyncio
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


@pytest.fixture(autouse=True)
def secure_sql_tools():
    """Keep the role check off the network and the global server's tools intact."""
    with patch("stratakb.mcp_server.secure_sql_tools", AsyncMock()) as secure:
        yield secure


@pytest.fixture(autouse=True)
def embedding_checks():
    """Keep the embedding checks off the network and off a real pool."""
    with patch("stratakb.mcp_server.run_embedding_checks", AsyncMock()) as checks:
        yield checks


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


async def test_lifespan_warns_unread_dotenv(db_ok, monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    with (
        patch("stratakb.mcp_server.dotenv_warning", return_value="hint text"),
        patch("stratakb.mcp_server.config_file_missing", return_value=False),
        capture_logs() as logs,
    ):
        async with mcp_server._lifespan(mcp_server.mcp):
            pass

    warned = [e for e in logs if e["event"] == "dotenv_not_read"]
    assert len(warned) == 1
    assert warned[0]["hint"] == "hint text"
    assert warned[0]["log_level"] == "warning"
    assert not any(e["event"] == "config_file_not_found" for e in logs)


async def test_lifespan_warns_missing_config_file(db_ok, monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    monkeypatch.setenv("STRATAKB_CONFIG", str(tmp_path / "missing-settings.env"))
    with (
        patch("stratakb.mcp_server.dotenv_warning", return_value=None),
        capture_logs() as logs,
    ):
        async with mcp_server._lifespan(mcp_server.mcp):
            pass

    missing = [e for e in logs if e["event"] == "config_file_not_found"]
    assert len(missing) == 1
    assert missing[0]["setting"] == "STRATAKB_CONFIG"
    assert "missing-settings" not in repr(logs)
    assert not any(e["event"] == "dotenv_not_read" for e in logs)


async def test_lifespan_quiet_without_config_problems(db_ok, monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    with (
        patch("stratakb.mcp_server.dotenv_warning", return_value=None),
        patch("stratakb.mcp_server.config_file_missing", return_value=False),
        capture_logs() as logs,
    ):
        async with mcp_server._lifespan(mcp_server.mcp):
            pass

    events = {e["event"] for e in logs}
    assert "dotenv_not_read" not in events
    assert "config_file_not_found" not in events


async def test_lifespan_secures_sql_tools(db_ok, secure_sql_tools, monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    async with mcp_server._lifespan(mcp_server.mcp):
        secure_sql_tools.assert_awaited_once_with(
            mcp_server.mcp, mcp_server._settings.readonly_database_url
        )


async def test_lifespan_does_not_wait_for_embedding_checks(db_ok, monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    started = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def blocking_checks(settings, db_ready):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            order.append("cancelled")
            raise

    async def close_pool():
        order.append("close_pool")

    with (
        patch("stratakb.mcp_server.run_embedding_checks", blocking_checks),
        patch("stratakb.mcp_server.close_pool", close_pool),
    ):
        t0 = time.monotonic()
        async with mcp_server._lifespan(mcp_server.mcp):
            assert time.monotonic() - t0 < 1.0
            await asyncio.wait_for(started.wait(), 1.0)
            assert not release.is_set()

    assert order == ["cancelled", "close_pool"]


async def test_embedding_checks_get_db_ready(db_ok, embedding_checks, monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    async with mcp_server._lifespan(mcp_server.mcp):
        await asyncio.sleep(0)
    embedding_checks.assert_awaited_once_with(mcp_server._settings, True)


async def test_embedding_checks_without_database(embedding_checks, monkeypatch):
    monkeypatch.setattr(mcp_server._settings, "embedding_backend", "openai")
    with (
        patch("stratakb.mcp_server.get_pool", AsyncMock(side_effect=OSError("refused"))),
        patch("stratakb.mcp_server.ensure_schema", AsyncMock()),
        patch("stratakb.mcp_server.close_pool", AsyncMock()),
    ):
        async with mcp_server._lifespan(mcp_server.mcp):
            await asyncio.sleep(0)
    embedding_checks.assert_awaited_once_with(mcp_server._settings, False)
