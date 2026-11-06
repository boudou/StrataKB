"""Shared test fixtures."""

from __future__ import annotations

import asyncio
import contextlib
import os
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest

# Asserts in the shared helpers report like asserts in a test module.
pytest.register_assert_rewrite("tests.helpers")


@pytest.fixture
def no_sdk_env(monkeypatch):
    """Keep SDK keys, base-URL overrides and the rewrite model from the shell out of a test."""
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "QUERY_REWRITE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def server():
    """A StrataKB server built from the environment's settings.

    Call its tools with `tests.helpers.call_tool`. A module may define its own
    `server` fixture, which overrides this one; it must also yield a built server.
    """
    from stratakb.config import get_settings
    from stratakb.mcp_server import build_server

    return build_server(get_settings())


@pytest.fixture
def db_url():
    """The test database URL from the environment's settings."""
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
def folder_db_url(db_url):
    """db_url, for a test that adds a tmp_path folder.

    tmp_path can sit under a system folder (%LOCALAPPDATA%): treat it as normal.
    """
    with patch("stratakb.knowledge.add_folder.broad_reason", return_value=None):
        yield db_url


@pytest.fixture
async def cleanup(db_url):
    """Layer names to delete after the test."""
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer

    names: list[str] = []
    yield names
    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


@pytest.fixture
def isolated_git(monkeypatch, tmp_path):
    """Keep the machine's git config and any repository above tmp_path out of the tests."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))


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


@pytest.fixture
def clear_checked():
    """Forget which databases ensure_schema has checked, before and after a test."""
    from stratakb.core import migrations

    migrations._checked.clear()
    yield
    migrations._checked.clear()


@pytest.fixture
async def restore_schema(db_url):
    """Put the shared test schema back after a test that drops parts of it.

    Replays every migration file as the owner (they are idempotent), so the
    restore does not depend on ensure_schema working, then records the versions.
    """
    from stratakb.core import migrations
    from tests.helpers import owner_execute

    yield
    for m in migrations.MIGRATIONS:
        await owner_execute(db_url, migrations._load_sql(m.file))
    migrations._checked.clear()
    await migrations.ensure_schema(db_url)


async def _new_database(db_url: str, name: str):
    """Yield the URL of a new, empty database name next to db_url, then drop it."""
    from tests.helpers import owner_execute

    parts = urlsplit(db_url)
    try:
        await owner_execute(db_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await owner_execute(db_url, f'CREATE DATABASE "{name}"')
    except asyncpg.InsufficientPrivilegeError:
        pytest.skip("the owner may not create databases")
    try:
        yield urlunsplit(parts._replace(path=f"/{name}"))
    finally:
        await owner_execute(db_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
async def fresh_url(db_url):
    """URL of a new, empty database next to db_url, dropped afterwards."""
    parts = urlsplit(db_url)
    async for url in _new_database(db_url, f"{parts.path.lstrip('/')}_fresh"):
        yield url


@pytest.fixture
async def reserved_url(db_url):
    """URL of a new, empty database named template_<db_url's database>, dropped afterwards."""
    parts = urlsplit(db_url)
    async for url in _new_database(db_url, f"template_{parts.path.lstrip('/')}"):
        yield url


@pytest.fixture
async def probe_url(db_url):
    """URL for a new login role with no privileges on the StrataKB schemas."""
    from tests.helpers import owner_execute

    parts = urlsplit(db_url)
    role = f"strata_probe_{parts.path.lstrip('/')}"
    password = "probe_password"
    drop = f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
                DROP OWNED BY "{role}";
                DROP ROLE "{role}";
            END IF;
        END $$;
    """
    try:
        await owner_execute(db_url, drop)
        await owner_execute(db_url, f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}'")
    except asyncpg.InsufficientPrivilegeError:
        pytest.skip("the owner may not create roles")
    try:
        netloc = f"{role}:{password}@{parts.hostname}:{parts.port}"
        yield urlunsplit(parts._replace(netloc=netloc))
    finally:
        await owner_execute(db_url, drop)


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
