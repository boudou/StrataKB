"""PostgreSQL connection pool management using asyncpg."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from urllib.parse import urlsplit

import asyncpg
import structlog

logger = structlog.get_logger()

# One pool per URL: the SQL tools may connect as a separate read-only role
_pools: dict[str, asyncpg.Pool] = {}
# One lock per URL, so concurrent first calls share a single creation
_locks: dict[str, asyncio.Lock] = {}


def url_host_port(database_url: str) -> str:
    """host:port of a database URL, never its user, password or database name."""
    try:
        parts = urlsplit(database_url)
        host = parts.hostname or "localhost"
        port = parts.port or 5432
    except ValueError:
        return "localhost:5432"
    return f"{host}:{port}"


async def get_pool(database_url: str) -> asyncpg.Pool:
    """Get or create the connection pool for a database URL."""
    pool = _pools.get(database_url)
    if pool is not None:
        return pool
    async with _locks.setdefault(database_url, asyncio.Lock()):
        pool = _pools.get(database_url)
        if pool is not None:
            return pool
        logger.info("creating_db_pool", url=database_url.split("@")[-1])
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


async def fetch_layers_stamp(database_url: str) -> tuple[int, datetime | None]:
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
