"""Shared test fixtures."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING
from uuid import uuid4

import asyncpg
import pytest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

# Asserts in the shared helpers report like asserts in a test module.
pytest.register_assert_rewrite("tests.helpers")


async def call_tool(server: FastMCP, name: str, /, **arguments) -> dict:
    """Call tool `name` on a built server, through its role and logging decorators."""
    return await server._tool_manager.get_tool(name).fn(**arguments)


@pytest.fixture
def server():
    """A StrataKB server built from the environment's settings."""
    from stratakb.config import get_settings
    from stratakb.mcp_server import build_server

    return build_server(get_settings())


@pytest.fixture
def db_url():
    """The test database URL from the environment's settings.

    A module may define its own `db_url` fixture, which overrides this one.
    """
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def layer(db_url):
    """A fresh layer with a unique name and 500/50 chunking, deleted afterwards.

    A module may define its own `layer` fixture, which overrides this one.
    """
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_layer_{uuid4().hex[:10]}"
    await create_layer(
        db_url,
        name=name,
        display_name="Test Layer",
        chunk_config={"chunk_size": 500, "chunk_overlap": 50},
    )
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


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
