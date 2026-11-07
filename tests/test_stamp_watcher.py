"""Tests for the shared layers-stamp watcher behind the query and routing caches."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from structlog.testing import capture_logs

from stratakb.core.stamp_watcher import StampWatcher

URL = "postgresql://example/db"


def _make_watcher(fetch: AsyncMock) -> tuple[StampWatcher, MagicMock, MagicMock]:
    """A watcher with test event names; returns it, its clear mock and its clock mock."""
    clear = MagicMock()
    clock = MagicMock(return_value=100.0)
    watcher = StampWatcher(
        clear,
        cleared_event="test_cache_cleared",
        failed_event="test_cache_stamp_failed",
        clock=clock,
        fetch=fetch,
    )
    return watcher, clear, clock


async def test_first_read_records_and_change_clears():
    fetch = AsyncMock(side_effect=[(1, None), (1, None), (2, None)])
    watcher, clear, clock = _make_watcher(fetch)

    await watcher.check(URL)
    clock.return_value = 103.0
    await watcher.check(URL)
    clear.assert_not_called()

    clock.return_value = 106.0
    with capture_logs() as logs:
        await watcher.check(URL)
    clear.assert_called_once()
    assert [e["event"] for e in logs] == ["test_cache_cleared"]


async def test_reads_at_most_every_two_seconds_per_url():
    fetch = AsyncMock(return_value=(1, None))
    watcher, _, clock = _make_watcher(fetch)

    await watcher.check(URL)
    clock.return_value = 101.9
    await watcher.check(URL)
    assert fetch.await_count == 1

    await watcher.check("postgresql://other/db")
    assert fetch.await_count == 2

    clock.return_value = 102.0
    await watcher.check(URL)
    assert fetch.await_count == 3


async def test_failed_read_logs_and_keeps_cache():
    fetch = AsyncMock(side_effect=OSError("down"))
    watcher, clear, _ = _make_watcher(fetch)

    with capture_logs() as logs:
        await watcher.check(URL)
    clear.assert_not_called()
    assert logs == [
        {"event": "test_cache_stamp_failed", "log_level": "warning", "error_type": "OSError"}
    ]


async def test_reset_forgets_stamps_and_read_times():
    fetch = AsyncMock(side_effect=[(1, None), (2, None)])
    watcher, clear, _ = _make_watcher(fetch)

    await watcher.check(URL)
    watcher.reset()

    # Same clock reading: only the reset allows this read, and it only records
    await watcher.check(URL)
    assert fetch.await_count == 2
    clear.assert_not_called()
    assert watcher.stamps == {URL: (2, None)}


async def test_owners_watch_separately():
    from stratakb.knowledge import layers as layers_mod
    from stratakb.rag import retriever

    assert retriever._watcher is not layers_mod._watcher
    assert retriever._watcher.stamps is not layers_mod._watcher.stamps
    assert retriever._watcher.cleared_event == "query_cache_cleared"
    assert layers_mod._watcher.failed_event == "routing_cache_stamp_failed"
