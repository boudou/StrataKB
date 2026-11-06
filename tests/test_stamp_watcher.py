"""Tests for the shared layers-stamp watcher behind the query and routing caches."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from structlog.testing import capture_logs

from stratakb.core.stamp_watcher import StampWatcher

URL = "postgresql://example/db"


async def test_first_read_records_and_change_clears():
    watcher = StampWatcher("test_cache")
    clear = MagicMock()
    fetch = AsyncMock(side_effect=[(1, None), (1, None), (2, None)])

    await watcher.check(URL, clear, now=100.0, fetch=fetch)
    await watcher.check(URL, clear, now=103.0, fetch=fetch)
    clear.assert_not_called()

    with capture_logs() as logs:
        await watcher.check(URL, clear, now=106.0, fetch=fetch)
    clear.assert_called_once()
    assert [e["event"] for e in logs] == ["test_cache_cleared"]


async def test_reads_at_most_every_two_seconds_per_url():
    watcher = StampWatcher("test_cache")
    fetch = AsyncMock(return_value=(1, None))

    await watcher.check(URL, MagicMock(), now=100.0, fetch=fetch)
    await watcher.check(URL, MagicMock(), now=101.9, fetch=fetch)
    assert fetch.await_count == 1

    await watcher.check("postgresql://other/db", MagicMock(), now=101.9, fetch=fetch)
    assert fetch.await_count == 2

    await watcher.check(URL, MagicMock(), now=102.0, fetch=fetch)
    assert fetch.await_count == 3


async def test_failed_read_logs_and_keeps_cache():
    watcher = StampWatcher("test_cache")
    clear = MagicMock()
    fetch = AsyncMock(side_effect=OSError("down"))

    with capture_logs() as logs:
        await watcher.check(URL, clear, now=100.0, fetch=fetch)
    clear.assert_not_called()
    assert logs == [
        {"event": "test_cache_stamp_failed", "log_level": "warning", "error_type": "OSError"}
    ]


async def test_owners_watch_separately():
    from stratakb.knowledge import layers as layers_mod
    from stratakb.rag import retriever

    assert retriever._watcher is not layers_mod._watcher
    assert retriever._stamps is retriever._watcher.stamps
    assert layers_mod._checked_at is layers_mod._watcher.checked_at
