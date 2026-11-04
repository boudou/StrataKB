"""Tests for the numbered schema migrations (stratakb.core.migrations).

All tests need the database. The non-owner tests also need READONLY_DATABASE_URL
on the same database.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from structlog.testing import capture_logs

from stratakb.core import migrations


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
def readonly_url(db_url):
    """READONLY_DATABASE_URL, only when it points at the same database as db_url."""
    from stratakb.config import get_settings

    url = get_settings().readonly_database_url
    if not url:
        pytest.skip("READONLY_DATABASE_URL is not set")
    if urlsplit(url).path != urlsplit(db_url).path:
        pytest.skip("READONLY_DATABASE_URL points at another database")
    return url


@pytest.fixture(autouse=True)
def _clear_checked():
    migrations._checked.clear()
    yield
    migrations._checked.clear()


async def _owner_execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _recorded(db_url: str) -> set[int]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch("SELECT version FROM strata_core.schema_migrations")
    finally:
        await conn.close()
    return {row["version"] for row in rows}


def _with_extra(version: int, name: str, sql: str):
    """Patch MIGRATIONS with one extra, never-present migration whose file is sql."""
    real_load = migrations._load_sql

    def load(file_name: str) -> str:
        return sql if file_name == name else real_load(file_name)

    return (
        patch.object(
            migrations, "MIGRATIONS", [*migrations.MIGRATIONS, (version, name, "SELECT false")]
        ),
        patch.object(migrations, "_load_sql", side_effect=load),
    )


async def test_sql_files_apply_twice(db_url):
    for name in ("04_ingested_files.sql", "05_add_folder.sql"):
        sql = migrations._load_sql(name)
        await _owner_execute(db_url, sql)
        await _owner_execute(db_url, sql)
    assert {4, 5} <= await _recorded(db_url)


async def test_records_existing_objects_without_running(db_url):
    await _owner_execute(db_url, "DROP TABLE strata_core.schema_migrations")
    with patch.object(migrations, "_load_sql", side_effect=AssertionError("file was run")):
        await migrations.ensure_schema(db_url)
    assert await _recorded(db_url) == {4, 5}
    assert db_url in migrations._checked


async def test_second_call_is_noop(db_url):
    await migrations.ensure_schema(db_url)
    with patch("stratakb.core.migrations.asyncpg.connect") as connect:
        await migrations.ensure_schema(db_url)
    connect.assert_not_called()


async def test_applies_missing_05(db_url):
    await _owner_execute(
        db_url,
        """
        DROP INDEX strata_core.knowledge_layers_root_path_key;
        ALTER TABLE strata_core.knowledge_layers
            DROP COLUMN origin, DROP COLUMN owner, DROP COLUMN root_path;
        DELETE FROM strata_core.schema_migrations WHERE version = 5;
        """,
    )
    await migrations.ensure_schema(db_url)

    conn = await asyncpg.connect(db_url)
    try:
        columns = {
            row["column_name"]
            for row in await conn.fetch(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'strata_core' AND table_name = 'knowledge_layers'
                """
            )
        }
        index = await conn.fetchval(
            "SELECT to_regclass('strata_core.knowledge_layers_root_path_key')::text"
        )
    finally:
        await conn.close()
    assert {"origin", "owner", "root_path"} <= columns
    assert index == "strata_core.knowledge_layers_root_path_key"
    assert await _recorded(db_url) == {4, 5}


async def test_concurrent_calls(db_url):
    await _owner_execute(db_url, "DROP TABLE strata_core.schema_migrations")
    await asyncio.gather(migrations.ensure_schema(db_url), migrations.ensure_schema(db_url))
    assert await _recorded(db_url) == {4, 5}


async def test_broken_file_raises_naming_it(db_url):
    extra_migrations, extra_load = _with_extra(9001, "90_broken.sql", "SELEC 1")
    with (
        extra_migrations,
        extra_load,
        pytest.raises(migrations.MigrationError, match="90_broken.sql") as exc,
    ):
        await migrations.ensure_schema(db_url)
    assert "psql -f" in str(exc.value)
    assert db_url not in migrations._checked
    assert 9001 not in await _recorded(db_url)


async def test_bad_password_passes_through(db_url):
    parts = urlsplit(db_url)
    user = parts.username or "strata"
    netloc = f"{user}:wrong-password@{parts.hostname}:{parts.port}"
    bad_url = urlunsplit(parts._replace(netloc=netloc))
    with pytest.raises(asyncpg.InvalidPasswordError):
        await migrations.ensure_schema(bad_url)
    assert bad_url not in migrations._checked


async def test_non_owner_schema_present(db_url, readonly_url):
    await _owner_execute(db_url, "DROP TABLE strata_core.schema_migrations")
    try:
        with capture_logs() as logs:
            await migrations.ensure_schema(readonly_url)
        events = [entry for entry in logs if entry["event"] == "migration_not_recorded"]
        assert len(events) == 1
        assert events[0]["versions"] == [4, 5]
        assert readonly_url in migrations._checked
    finally:
        await migrations.ensure_schema(db_url)
    assert await _recorded(db_url) == {4, 5}


async def test_non_owner_schema_missing(db_url, readonly_url):
    await migrations.ensure_schema(db_url)
    migrations._checked.clear()
    extra_migrations, extra_load = _with_extra(
        9002, "92_probe.sql", "CREATE TABLE strata_core.migration_probe (id INT)"
    )
    with (
        extra_migrations,
        extra_load,
        pytest.raises(migrations.MigrationError, match="may not update it") as exc,
    ):
        await migrations.ensure_schema(readonly_url)
    assert "92_probe.sql" in str(exc.value)
    assert readonly_url not in migrations._checked
