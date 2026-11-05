"""Tests for get_pool: concurrent and cancelled first calls (no database)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from stratakb.core import database
from stratakb.core.database import close_pool, get_pool

URL = "postgresql://user:secret@db.example:5432/strata"


class FakePool:
    """Stands in for an uninitialised asyncpg Pool: awaiting it initialises."""

    def __init__(self, gate: asyncio.Event, error: BaseException | None = None):
        self.gate = gate
        self.error = error
        self.started = asyncio.Event()
        self.terminated = 0
        self.closed = 0

    def __await__(self):
        return self._init().__await__()

    async def _init(self) -> FakePool:
        self.started.set()
        await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self

    def terminate(self) -> None:
        self.terminated += 1

    async def close(self) -> None:
        self.closed += 1


class FakeFactory:
    """Replaces asyncpg.create_pool: a plain function returning a FakePool."""

    def __init__(self, error: BaseException | None = None):
        self.gate = asyncio.Event()
        self.error = error
        self.pools: list[FakePool] = []

    def __call__(self, *args, **kwargs) -> FakePool:
        pool = FakePool(self.gate, self.error if not self.pools else None)
        self.pools.append(pool)
        return pool


@pytest.fixture
async def clean_pools():
    await close_pool()
    yield
    await close_pool()


async def test_concurrent_first_calls_create_one_pool(clean_pools):
    factory = FakeFactory()
    with patch.object(database.asyncpg, "create_pool", factory):
        tasks = [asyncio.create_task(get_pool(URL)) for _ in range(10)]
        await asyncio.sleep(0)
        factory.gate.set()
        results = await asyncio.gather(*tasks)

    assert len(factory.pools) == 1
    assert all(r is factory.pools[0] for r in results)
    assert database._pools[URL] is factory.pools[0]


async def test_cancelled_creation_leaves_nothing_behind(clean_pools):
    factory = FakeFactory()
    with patch.object(database.asyncpg, "create_pool", factory):
        task = asyncio.create_task(get_pool(URL))
        await asyncio.sleep(0)
        await factory.pools[0].started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert URL not in database._pools
        assert factory.pools[0].terminated == 1

        factory.gate.set()
        pool = await get_pool(URL)

    assert len(factory.pools) == 2
    assert pool is factory.pools[1]
    assert database._pools[URL] is pool


async def test_cancelled_leader_lets_waiter_create(clean_pools):
    factory = FakeFactory()
    with patch.object(database.asyncpg, "create_pool", factory):
        leader = asyncio.create_task(get_pool(URL))
        await asyncio.sleep(0)
        await factory.pools[0].started.wait()
        waiter = asyncio.create_task(get_pool(URL))
        await asyncio.sleep(0)
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        factory.gate.set()
        pool = await waiter

    assert factory.pools[0].terminated == 1
    assert len(factory.pools) == 2
    assert pool is factory.pools[1]
    assert list(database._pools) == [URL]
    assert database._pools[URL] is pool


async def test_failed_creation_is_terminated_and_retried(clean_pools):
    factory = FakeFactory(error=OSError("connection refused"))
    factory.gate.set()
    with patch.object(database.asyncpg, "create_pool", factory):
        with pytest.raises(OSError, match="connection refused"):
            await get_pool(URL)

        assert URL not in database._pools
        assert factory.pools[0].terminated == 1

        pool = await get_pool(URL)

    assert pool is factory.pools[1]
    assert database._pools[URL] is pool


async def test_close_pool_clears_locks(clean_pools):
    factory = FakeFactory()
    factory.gate.set()
    with patch.object(database.asyncpg, "create_pool", factory):
        pool = await get_pool(URL)
    assert database._locks

    await close_pool()

    assert database._pools == {}
    assert database._locks == {}
    assert pool.closed == 1
