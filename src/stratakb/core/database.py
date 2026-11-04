"""PostgreSQL connection pool management using asyncpg."""

from __future__ import annotations

from datetime import datetime

import asyncpg
import structlog

logger = structlog.get_logger()

# One pool per URL: the SQL tools may connect as a separate read-only role
_pools: dict[str, asyncpg.Pool] = {}


async def get_pool(database_url: str) -> asyncpg.Pool:
    """Get or create the connection pool for a database URL."""
    pool = _pools.get(database_url)
    if pool is None:
        logger.info("creating_db_pool", url=database_url.split("@")[-1])
        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=20,
            command_timeout=30,
        )
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
