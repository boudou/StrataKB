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
    for name in (
        "01_init.sql",
        "02_learning.sql",
        "04_ingested_files.sql",
        "05_add_folder.sql",
        "06_readonly_grants.sql",
    ):
        sql = migrations._load_sql(name)
        await _owner_execute(db_url, sql)
        await _owner_execute(db_url, sql)
    assert {4, 5, 6} <= await _recorded(db_url)


@pytest.fixture
async def fresh_url(db_url):
    """URL of a new, empty database next to db_url, dropped afterwards."""
    parts = urlsplit(db_url)
    name = f"{parts.path.lstrip('/')}_fresh"
    try:
        await _owner_execute(db_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await _owner_execute(db_url, f'CREATE DATABASE "{name}"')
    except asyncpg.InsufficientPrivilegeError:
        pytest.skip("the owner may not create databases")
    try:
        yield urlunsplit(parts._replace(path=f"/{name}"))
    finally:
        await _owner_execute(db_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


async def test_builds_an_empty_database(fresh_url):
    await migrations.ensure_schema(fresh_url)

    assert await _recorded(fresh_url) == {1, 2, 4, 5, 6}
    conn = await asyncpg.connect(fresh_url)
    try:
        for table in (
            "strata_core.knowledge_layers",
            "strata_core.ingested_files",
            "strata_vectors.chunks",
            "strata_learning.query_sessions",
        ):
            assert await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table), table
        columns = {
            (row["table_name"], row["column_name"])
            for row in await conn.fetch(
                """
                SELECT table_name, column_name FROM information_schema.columns
                WHERE (table_schema, table_name, column_name) IN (
                    ('strata_core', 'knowledge_layers', 'root_path'),
                    ('strata_vectors', 'chunks', 'file_id')
                )
                """
            )
        }
        assert columns == {("knowledge_layers", "root_path"), ("chunks", "file_id")}

        if await conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = 'strata_readonly'"):
            readable = {
                row["t"]
                for row in await conn.fetch(
                    """
                    SELECT format('%I.%I', n.nspname, c.relname) AS t
                    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname IN ('strata_core', 'strata_vectors', 'strata_learning')
                      AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                      AND has_table_privilege('strata_readonly', c.oid, 'SELECT')
                    """
                )
            }
            assert readable == set(_READABLE)
            assert not await conn.fetchval(
                "SELECT has_schema_privilege('strata_readonly', 'strata_learning', 'USAGE')"
            )
    finally:
        await conn.close()

    migrations._checked.clear()
    with capture_logs() as logs:
        await migrations.ensure_schema(fresh_url)
    assert not [entry for entry in logs if entry["event"] == "applying_migration"]


async def test_records_01_and_02_on_an_existing_database(db_url):
    await migrations.ensure_schema(db_url)
    migrations._checked.clear()
    await _owner_execute(
        db_url, "DELETE FROM strata_core.schema_migrations WHERE version IN (1, 2)"
    )
    with patch.object(migrations, "_load_sql", side_effect=AssertionError("file was run")):
        await migrations.ensure_schema(db_url)
    assert {1, 2} <= await _recorded(db_url)


async def test_non_owner_empty_database_names_01(fresh_url, readonly_url):
    url = urlunsplit(urlsplit(readonly_url)._replace(path=urlsplit(fresh_url).path))
    with pytest.raises(migrations.MigrationError) as exc:
        await migrations.ensure_schema(url)
    assert "01_init.sql" in str(exc.value)
    assert url not in migrations._checked


async def test_records_existing_objects_without_running(db_url):
    await _owner_execute(db_url, "DROP TABLE strata_core.schema_migrations")
    with patch.object(migrations, "_load_sql", side_effect=AssertionError("file was run")):
        await migrations.ensure_schema(db_url)
    assert await _recorded(db_url) == {1, 2, 4, 5, 6}
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
    assert await _recorded(db_url) == {1, 2, 4, 5, 6}


async def test_concurrent_calls(db_url):
    await _owner_execute(db_url, "DROP TABLE strata_core.schema_migrations")
    await asyncio.gather(migrations.ensure_schema(db_url), migrations.ensure_schema(db_url))
    assert await _recorded(db_url) == {1, 2, 4, 5, 6}


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
        assert events[0]["versions"] == [1, 2, 4, 5, 6]
        assert readonly_url in migrations._checked
    finally:
        await migrations.ensure_schema(db_url)
    assert await _recorded(db_url) == {1, 2, 4, 5, 6}


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


_READABLE = (
    "strata_core.knowledge_layers",
    "strata_core.ingestion_runs",
    "strata_core.ingested_files",
    "strata_vectors.chunks",
)

# The grants db/03_readonly_role.sql made before 06
_OLD_GRANTS = """
GRANT USAGE ON SCHEMA strata_core, strata_vectors, strata_learning TO strata_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA strata_core, strata_vectors, strata_learning
    TO strata_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA strata_core, strata_vectors, strata_learning
    GRANT SELECT ON TABLES TO strata_readonly;
DELETE FROM strata_core.schema_migrations WHERE version = 6;
"""


def _presence_query(version: int) -> str:
    return next(query for v, _, query in migrations.MIGRATIONS if v == version)


@pytest.fixture
async def readonly_role(db_url):
    """Skip unless the strata_readonly role exists; restore 06's grants afterwards."""
    conn = await asyncpg.connect(db_url)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = 'strata_readonly'")
    finally:
        await conn.close()
    if not exists:
        pytest.skip("the strata_readonly role does not exist")
    yield
    await _owner_execute(db_url, migrations._load_sql("06_readonly_grants.sql"))


async def _readonly_privileges(db_url: str) -> dict:
    conn = await asyncpg.connect(db_url)
    try:
        tables = {
            row["t"]: row["readable"]
            for row in await conn.fetch(
                """
                SELECT format('%I.%I', n.nspname, c.relname) AS t,
                       has_table_privilege('strata_readonly', c.oid, 'SELECT') AS readable
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname IN ('strata_core', 'strata_vectors', 'strata_learning')
                  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                """
            )
        }
        learning_usage = await conn.fetchval(
            "SELECT has_schema_privilege('strata_readonly', 'strata_learning', 'USAGE')"
        )
        default_acls = await conn.fetchval(
            """
            SELECT count(*) FROM pg_default_acl d
            CROSS JOIN LATERAL aclexplode(d.defaclacl) a
            WHERE a.grantee = 'strata_readonly'::regrole
            """
        )
    finally:
        await conn.close()
    return {"tables": tables, "learning_usage": learning_usage, "default_acls": default_acls}


def _assert_closed(privileges: dict) -> None:
    tables = privileges["tables"]
    assert {t for t, readable in tables.items() if readable} == set(_READABLE)
    for t in ("strata_core.audit_log", "strata_core.users", "strata_core.schema_migrations"):
        assert tables[t] is False
    learning = [t for t in tables if t.startswith("strata_learning.")]
    assert learning
    assert not any(tables[t] for t in learning)
    assert privileges["learning_usage"] is False
    assert privileges["default_acls"] == 0


async def test_readonly_grants_migrate_old_install(db_url, readonly_role):
    await _owner_execute(db_url, _OLD_GRANTS)
    conn = await asyncpg.connect(db_url)
    try:
        assert await conn.fetchval(_presence_query(6)) is False
    finally:
        await conn.close()

    await migrations.ensure_schema(db_url)

    _assert_closed(await _readonly_privileges(db_url))
    assert 6 in await _recorded(db_url)


async def test_readonly_role_select(db_url, readonly_role):
    await migrations.ensure_schema(db_url)
    conn = await asyncpg.connect(db_url)
    try:
        if not await conn.fetchval(
            "SELECT rolsuper OR pg_has_role(current_user, 'strata_readonly', 'MEMBER') "
            "FROM pg_roles WHERE rolname = current_user"
        ):
            pytest.skip("the owner may not SET ROLE strata_readonly")
        for table in ("strata_core.audit_log", "strata_core.users"):
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with conn.transaction():
                    await conn.execute("SET LOCAL ROLE strata_readonly")
                    await conn.fetch(f"SELECT 1 FROM {table} LIMIT 1")
        for table in ("strata_vectors.chunks", "strata_core.knowledge_layers"):
            async with conn.transaction():
                await conn.execute("SET LOCAL ROLE strata_readonly")
                await conn.fetch(f"SELECT 1 FROM {table} LIMIT 1")
    finally:
        await conn.close()


async def test_readonly_new_table_not_readable(db_url, readonly_role):
    await _owner_execute(db_url, _OLD_GRANTS)
    try:
        await migrations.ensure_schema(db_url)
        await _owner_execute(db_url, "CREATE TABLE strata_core.grant_probe (id INT)")
        conn = await asyncpg.connect(db_url)
        try:
            assert not await conn.fetchval(
                "SELECT has_table_privilege('strata_readonly', 'strata_core.grant_probe', 'SELECT')"
            )
        finally:
            await conn.close()
    finally:
        await _owner_execute(db_url, "DROP TABLE IF EXISTS strata_core.grant_probe")


async def test_readonly_presence_without_ingested_files(db_url, readonly_role):
    conn = await asyncpg.connect(db_url)
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute(
                "ALTER TABLE strata_core.ingested_files RENAME TO ingested_files_probe"
            )
            assert await conn.fetchval(_presence_query(6)) is False
        finally:
            await tx.rollback()
    finally:
        await conn.close()
