"""Shared test fixtures."""

from __future__ import annotations

import asyncio

import asyncpg
import pytest


@pytest.fixture(scope="session", autouse=True)
def _migrate_test_db():
    """Bring the test database up to the current schema once per session.

    Does nothing when DATABASE_URL is unset or the database is unreachable,
    so tests that need no database still run. A MigrationError fails loudly.
    """
    from stratakb.config import get_settings
    from stratakb.core import migrations

    url = get_settings().database_url
    if not url:
        return
    try:
        asyncio.run(migrations.ensure_schema(url))
    except (OSError, asyncpg.PostgresError):
        pass
    finally:
        migrations._checked.clear()


@pytest.fixture
async def vacuumed_chunks():
    """Vacuum the chunks table before a test that ranks through the HNSW index.

    Many tests insert and then delete the same fake embeddings. Until vacuum
    repairs the HNSW graph, rows added afterwards can be unreachable from its
    entry point, so an index scan returns fewer rows than the table holds.
    Autovacuum does this in a running server; a test can run first.
    """
    from stratakb.config import get_settings

    conn = await asyncpg.connect(get_settings().database_url)
    try:
        await conn.execute("VACUUM strata_vectors.chunks")
    finally:
        await conn.close()


@pytest.fixture(autouse=True)
async def _reset_db_pool():
    """Reset the global connection pools after each test.

    A pool is bound to the event loop that created it. Since pytest-asyncio
    creates a new loop per test by default, we must close the pools so the
    next test creates fresh ones on its own loop.
    """
    yield
    from stratakb.core.database import close_pool
    await close_pool()
