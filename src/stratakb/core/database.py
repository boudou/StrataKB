"""PostgreSQL connection pool management using asyncpg."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import datetime

import asyncpg
import structlog

from stratakb.core.urls import url_host_port

logger = structlog.get_logger()

# What fetch_layers_stamp returns: (layer count, newest updated_at)
Stamp = tuple[int, datetime | None]

# One pool per URL: the SQL tools may connect as a separate read-only role
_pools: dict[str, asyncpg.Pool] = {}
# One lock per URL: concurrent first calls wait their turn, and each re-checks _pools,
# so only one pool is made per URL (after a failed creation the next caller tries again)
_locks: dict[str, asyncio.Lock] = {}


@contextlib.asynccontextmanager
async def connect_once(database_url: str, *, timeout: float) -> AsyncIterator[asyncpg.Connection]:
    """One connection outside the pools, with a deadline at both ends.

    Connects within `timeout` seconds and closes within `timeout` seconds when
    the body finishes. When the body raises or is cancelled the connection is
    dropped with terminate(): a graceful close() first cancels a running query
    over a new connection, which has no deadline.
    """
    conn = await asyncpg.connect(database_url, timeout=timeout)
    try:
        yield conn
    except BaseException:
        conn.terminate()
        raise
    await conn.close(timeout=timeout)


async def get_pool(database_url: str) -> asyncpg.Pool:
    """Get or create the connection pool for a database URL."""
    pool = _pools.get(database_url)
    if pool is not None:
        return pool
    async with _locks.setdefault(database_url, asyncio.Lock()):
        pool = _pools.get(database_url)
        if pool is not None:
            return pool
        logger.info("creating_db_pool", url=url_host_port(database_url))
        pool = asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=20,
            command_timeout=30,
        )
        try:
            await pool
        except BaseException:
            # Cancelled or failed while connecting: close what was opened
            with contextlib.suppress(Exception):
                pool.terminate()
            raise
        _pools[database_url] = pool
        logger.info("db_pool_created")
    return pool


async def fetch_layers_stamp(database_url: str) -> Stamp:
    """Return (layer count, newest updated_at) of strata_core.knowledge_layers.

    Caches compare this stamp to notice layer changes made by any process.
    """
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        "SELECT count(*) AS layers, max(updated_at) AS latest "
        "FROM strata_core.knowledge_layers"
    )
    return (row["layers"], row["latest"])


async def close_pool() -> None:
    """Close all connection pools."""
    # Locks bind to the event loop they first wait on; start fresh each time
    _locks.clear()
    while _pools:
        _, pool = _pools.popitem()
        await pool.close()
        logger.info("db_pool_closed")


async def check_connection(database_url: str) -> dict[str, str]:
    """Check database connectivity and return status."""
    try:
        pool = await get_pool(database_url)
        async with pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            # Check pgvector extension
            pgvector = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            # Check schemas exist
            schemas = await conn.fetch(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name IN ('strata_core', 'strata_vectors', 'strata_learning')"
            )
            schema_names = [r["schema_name"] for r in schemas]

        return {
            "status": "ok",
            "postgres_version": version.split(",")[0] if version else "unknown",
            "pgvector_version": pgvector or "not installed",
            "schemas": schema_names,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
