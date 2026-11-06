"""Tests for the numbered schema migrations (stratakb.core.migrations).

All tests need the database. The non-owner tests also need READONLY_DATABASE_URL
on the same database.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from structlog.testing import capture_logs

from stratakb.core import migrations

# Every version a complete schema records. Read once from the real list, so
# tests that patch migrations.MIGRATIONS still compare against it.
ALL_VERSIONS = {m.version for m in migrations.MIGRATIONS}


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


@pytest.fixture
async def restore_schema(db_url):
    """Put the shared test schema back after a test that drops parts of it.

    Replays every migration file as the owner (they are idempotent), so the
    restore does not depend on ensure_schema working, then records the versions.
    """
    yield
    for m in migrations.MIGRATIONS:
        await _owner_execute(db_url, migrations._load_sql(m.file))
    migrations._checked.clear()
    await migrations.ensure_schema(db_url)


async def _recorded(db_url: str) -> set[int]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch("SELECT version FROM strata_core.schema_migrations")
    finally:
        await conn.close()
    return {row["version"] for row in rows}


@contextlib.contextmanager
def _with_extra(version: int, name: str, sql: str):
    """Patch MIGRATIONS with one extra, never-present migration whose file is sql."""
    real_load = migrations._load_sql

    def load(file_name: str) -> str:
        return sql if file_name == name else real_load(file_name)

    extra = migrations.Migration(version, name, "SELECT false")
    with (
        patch.object(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, extra]),
        patch.object(migrations, "_load_sql", side_effect=load),
    ):
        yield


def test_schema_migrations_ddl_matches_05():
    sql = migrations._load_sql("05_add_folder.sql")
    found = re.search(
        r"CREATE TABLE IF NOT EXISTS strata_core\.schema_migrations \(.*?\);", sql, re.S
    )
    assert found
    in_05 = found.group(0).removesuffix(";")
    assert " ".join(in_05.split()) == " ".join(migrations._SCHEMA_MIGRATIONS_DDL.split())


_REPO = Path(__file__).parents[1]


def _granted(sql: str, privilege: str) -> set[str]:
    """Every name in the GRANT <privilege> ON ... TO strata_readonly statements."""
    sql = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    names: set[str] = set()
    pattern = rf"GRANT {privilege} ON (?:SCHEMA )?(.+?) TO strata_readonly"
    for found in re.finditer(pattern, sql, re.S):
        names.update(name.strip() for name in found.group(1).split(","))
    return names


def test_readonly_lists_agree():
    pairs = migrations._READONLY_TABLES
    assert len(set(pairs)) == len(pairs)
    in_module = {f"{schema}.{table}" for schema, table in pairs}

    in_06 = _granted(migrations._load_sql("06_readonly_grants.sql"), "SELECT")
    db_03 = (_REPO / "db" / "03_readonly_role.sql").read_text(encoding="utf-8")
    in_03 = _granted(db_03, "SELECT")
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    sentence = re.search(r"The `strata_readonly` role can read(.*?)It cannot read", readme, re.S)
    assert sentence
    in_readme = set(re.findall(r"`(strata_\w+\.\w+)`", sentence.group(1)))

    assert in_06 == in_module
    assert in_03 == in_module
    assert set(_READABLE) == in_module
    assert in_readme == in_module
    assert _granted(migrations._load_sql("08_readonly_documents.sql"), "SELECT") == {
        f"{schema}.{table}" for schema, table in migrations._READONLY_TABLES_AFTER_06
    }

    schemas = {schema for schema, _ in pairs}
    assert set(migrations._READABLE_SCHEMAS) == schemas
    assert _granted(migrations._load_sql("06_readonly_grants.sql"), "USAGE") == schemas
    assert _granted(db_03, "USAGE") == schemas

    # 06 also names the StrataKB schemas by hand where it revokes
    sql_06 = "\n".join(
        line
        for line in migrations._load_sql("06_readonly_grants.sql").splitlines()
        if not line.lstrip().startswith("--")
    )
    all_schemas = set(migrations._STRATA_SCHEMAS)
    acl_list = re.search(r"n\.nspname IN \((.+?)\)", sql_06, re.S)
    assert acl_list
    assert {name.strip().strip("'") for name in acl_list.group(1).split(",")} == all_schemas
    for pattern, expected in (
        (r"REVOKE ALL ON ALL TABLES IN SCHEMA (.+?) FROM", all_schemas),
        (r"REVOKE ALL ON SCHEMA (.+?) FROM", all_schemas - schemas),
    ):
        found = re.search(pattern, sql_06, re.S)
        assert found
        assert {name.strip() for name in found.group(1).split(",")} == expected


def test_usage_refused_names_readable_schemas():
    text = migrations._USAGE_REFUSED.format(error="x")
    assert all(schema in text for schema in migrations._READABLE_SCHEMAS)
    assert "strata_learning" not in text


def test_stage_errors_cover_every_stage():
    assert set(migrations._STAGE_ERRORS) == set(migrations._Stage)
    for stage in migrations._Stage:
        errors = migrations._STAGE_ERRORS[stage]
        assert errors.lock and errors.refused and errors.failed


def test_error_for_picks_the_family():
    file_lock = migrations._error_for(
        migrations._Stage.FILE, asyncpg.LockNotAvailableError("x"), "07_documents.sql"
    )
    assert "table lock" in str(file_lock)
    assert "07_documents.sql" in str(file_lock)

    read_refused = migrations._error_for(
        migrations._Stage.READ, asyncpg.InsufficientPrivilegeError("x"), ""
    )
    assert "USAGE" in str(read_refused)

    # Not an InsufficientPrivilegeError, but refused all the same
    table_refused = migrations._error_for(
        migrations._Stage.TABLE, asyncpg.ReadOnlySQLTransactionError("x"), "05_add_folder.sql"
    )
    assert "may not update it" in str(table_refused)
    assert "05_add_folder.sql" in str(table_refused)

    failed = migrations._error_for(
        migrations._Stage.FILE, asyncpg.PostgresError("bad {x}"), "07_documents.sql"
    )
    assert "bad {x}" in str(failed)
    assert "Schema migration 07_documents.sql failed" in str(failed)


async def _schema_snapshot(db_url: str) -> dict:
    """Columns, indexes, constraints, recorded versions and read-only privileges."""
    conn = await asyncpg.connect(db_url)
    try:
        columns = await conn.fetch(
            """
            SELECT table_schema, table_name, column_name, data_type, is_nullable,
                   column_default
            FROM information_schema.columns
            WHERE table_schema IN ('strata_core', 'strata_vectors', 'strata_learning')
            ORDER BY 1, 2, 3
            """
        )
        indexes = await conn.fetch(
            """
            SELECT schemaname, indexname, indexdef FROM pg_indexes
            WHERE schemaname IN ('strata_core', 'strata_vectors', 'strata_learning')
            ORDER BY 1, 2
            """
        )
        constraints = await conn.fetch(
            """
            SELECT c.conrelid::regclass::text AS rel, c.conname,
                   pg_get_constraintdef(c.oid) AS def
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname IN ('strata_core', 'strata_vectors', 'strata_learning')
            ORDER BY 1, 2
            """
        )
        versions = await conn.fetch(
            "SELECT version, name FROM strata_core.schema_migrations ORDER BY version"
        )
        has_role = await conn.fetchval(
            "SELECT 1 FROM pg_roles WHERE rolname = 'strata_readonly'"
        )
    finally:
        await conn.close()
    return {
        "columns": [tuple(row) for row in columns],
        "indexes": [tuple(row) for row in indexes],
        "constraints": [tuple(row) for row in constraints],
        "versions": [tuple(row) for row in versions],
        "privileges": await _readonly_privileges(db_url) if has_role else None,
    }


async def test_sql_files_apply_twice(db_url):
    await migrations.ensure_schema(db_url)
    for m in migrations.MIGRATIONS:
        await _owner_execute(db_url, migrations._load_sql(m.file))
    first = await _schema_snapshot(db_url)
    for m in migrations.MIGRATIONS:
        await _owner_execute(db_url, migrations._load_sql(m.file))
    assert await _schema_snapshot(db_url) == first
    assert first["columns"]
    assert {row[0] for row in first["columns"]} == {
        "strata_core",
        "strata_vectors",
        "strata_learning",
    }
    assert await _recorded(db_url) == ALL_VERSIONS


async def test_presence_true_on_current_schema(db_url):
    await migrations.ensure_schema(db_url)
    conn = await asyncpg.connect(db_url)
    try:
        for m in migrations.MIGRATIONS:
            assert await conn.fetchval(m.presence) is True, m.file
    finally:
        await conn.close()


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


async def _new_database(db_url: str, name: str):
    """Yield the URL of a new, empty database name next to db_url, then drop it."""
    parts = urlsplit(db_url)
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

    assert await _recorded(fresh_url) == ALL_VERSIONS
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


def _target(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.hostname}:{parts.port or 5432}{parts.path}"


async def test_setting_up_names_the_database(fresh_url):
    parts = urlsplit(fresh_url)
    seen: list[str] = []
    with capture_logs() as logs:
        await migrations.ensure_schema(fresh_url, on_setup=seen.append)

    setup = [entry for entry in logs if entry["event"] == "setting_up_database"]
    assert [entry["target"] for entry in setup] == [_target(fresh_url)]
    assert seen == [_target(fresh_url)]
    logged = repr(logs)
    assert parts.password not in logged
    assert f"{parts.username}:" not in logged
    assert f"{parts.username}@" not in logged

    migrations._checked.clear()
    with capture_logs() as logs:
        await migrations.ensure_schema(fresh_url, on_setup=seen.append)
    assert not [entry for entry in logs if entry["event"] == "setting_up_database"]
    assert len(seen) == 1


async def test_refuses_to_set_up_a_template_database(reserved_url):
    seen: list[str] = []
    with pytest.raises(migrations.MigrationError) as excinfo:
        await migrations.ensure_schema(reserved_url, on_setup=seen.append)

    message = str(excinfo.value)
    assert _target(reserved_url) in message
    assert "CREATE DATABASE" in message
    assert urlsplit(reserved_url).password not in message
    assert seen == []
    assert reserved_url not in migrations._checked
    conn = await asyncpg.connect(reserved_url)
    try:
        for schema in ("strata_core", "strata_vectors", "strata_learning"):
            assert await conn.fetchval("SELECT to_regnamespace($1)", schema) is None, schema
    finally:
        await conn.close()


async def test_updates_an_existing_schema_in_a_reserved_database(reserved_url):
    await _owner_execute(reserved_url, "CREATE SCHEMA strata_core")
    with capture_logs() as logs:
        await migrations.ensure_schema(reserved_url)

    assert await _recorded(reserved_url) == ALL_VERSIONS
    assert not [entry for entry in logs if entry["event"] == "setting_up_database"]


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("postgres", "maintenance database"),
        ("template0", "template database"),
        ("template1", "template database"),
        ("template_x", "template database"),
        ("strata", None),
        ("postgres2", None),
        ("my_template", None),
        ("Postgres", None),
    ],
)
def test_reserved_database_names(name, kind):
    assert migrations._reserved_database(name) == kind


async def test_records_01_and_02_on_an_existing_database(db_url, restore_schema):
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


async def test_records_existing_objects_without_running(db_url, restore_schema):
    await _owner_execute(db_url, "DROP TABLE strata_core.schema_migrations")
    with patch.object(migrations, "_load_sql", side_effect=AssertionError("file was run")):
        await migrations.ensure_schema(db_url)
    assert await _recorded(db_url) == ALL_VERSIONS
    assert db_url in migrations._checked


async def test_second_call_is_noop(db_url):
    await migrations.ensure_schema(db_url)
    with patch("stratakb.core.migrations.asyncpg.connect") as connect:
        await migrations.ensure_schema(db_url)
    connect.assert_not_called()


async def test_applies_missing_05(db_url, restore_schema):
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
    assert await _recorded(db_url) == ALL_VERSIONS


async def test_applies_missing_11_existing_layers_have_no_patterns(db_url, restore_schema):
    await _owner_execute(
        db_url,
        """
        DELETE FROM strata_core.knowledge_layers WHERE name = 'test_mig_structure';
        ALTER TABLE strata_core.knowledge_layers DROP COLUMN structure_patterns;
        DELETE FROM strata_core.schema_migrations WHERE version = 11;
        INSERT INTO strata_core.knowledge_layers (name, display_name)
        VALUES ('test_mig_structure', 'Older layer');
        """,
    )
    try:
        await migrations.ensure_schema(db_url)

        conn = await asyncpg.connect(db_url)
        try:
            patterns = await conn.fetchval(
                "SELECT structure_patterns::text FROM strata_core.knowledge_layers "
                "WHERE name = 'test_mig_structure'"
            )
        finally:
            await conn.close()
        assert patterns == "[]"
        assert await _recorded(db_url) == ALL_VERSIONS
    finally:
        await _owner_execute(
            db_url,
            "DELETE FROM strata_core.knowledge_layers WHERE name = 'test_mig_structure'",
        )


async def test_applies_missing_12_existing_layers_read_grounded(db_url, restore_schema):
    await _owner_execute(
        db_url,
        """
        DELETE FROM strata_core.knowledge_layers WHERE name = 'test_mig_answer_mode';
        ALTER TABLE strata_core.knowledge_layers DROP COLUMN answer_mode;
        DELETE FROM strata_core.schema_migrations WHERE version = 12;
        INSERT INTO strata_core.knowledge_layers (name, display_name)
        VALUES ('test_mig_answer_mode', 'Older layer');
        """,
    )
    try:
        await migrations.ensure_schema(db_url)

        conn = await asyncpg.connect(db_url)
        try:
            mode = await conn.fetchval(
                "SELECT answer_mode FROM strata_core.knowledge_layers "
                "WHERE name = 'test_mig_answer_mode'"
            )
        finally:
            await conn.close()
        assert mode == "grounded"
        assert await _recorded(db_url) == ALL_VERSIONS
    finally:
        await _owner_execute(
            db_url,
            "DELETE FROM strata_core.knowledge_layers WHERE name = 'test_mig_answer_mode'",
        )


async def test_applies_missing_07_older_chunks_keep_null_offsets(db_url, restore_schema):
    await _owner_execute(
        db_url,
        """
        DELETE FROM strata_vectors.chunks WHERE layer_name = 'test_mig_documents';
        ALTER TABLE strata_vectors.chunks
            DROP COLUMN document_id, DROP COLUMN char_start, DROP COLUMN char_end;
        DROP TABLE strata_core.receipt_documents;
        DROP TABLE strata_core.receipts;
        DROP TABLE strata_vectors.document_sections;
        DROP TABLE strata_vectors.documents;
        DELETE FROM strata_core.schema_migrations WHERE version IN (7, 9, 10);
        INSERT INTO strata_vectors.chunks (layer_name, content, source_file)
        VALUES ('test_mig_documents', 'Older chunk', 'old.md');
        """,
    )
    try:
        await migrations.ensure_schema(db_url)

        conn = await asyncpg.connect(db_url)
        try:
            assert await conn.fetchval(
                "SELECT to_regclass('strata_vectors.documents') IS NOT NULL"
            )
            row = await conn.fetchrow(
                "SELECT document_id, char_start, char_end FROM strata_vectors.chunks "
                "WHERE layer_name = 'test_mig_documents'"
            )
        finally:
            await conn.close()
        assert tuple(row) == (None, None, None)
        assert await _recorded(db_url) == ALL_VERSIONS
    finally:
        await _owner_execute(
            db_url,
            "DELETE FROM strata_vectors.chunks WHERE layer_name = 'test_mig_documents'",
        )


async def test_applies_missing_09_older_chunks_keep_null_clause_path(db_url, restore_schema):
    await _owner_execute(
        db_url,
        """
        DELETE FROM strata_vectors.chunks WHERE layer_name = 'test_mig_structure';
        ALTER TABLE strata_vectors.chunks DROP COLUMN clause_path;
        DROP TABLE strata_vectors.document_sections;
        DELETE FROM strata_core.schema_migrations WHERE version = 9;
        INSERT INTO strata_vectors.chunks (layer_name, content, source_file)
        VALUES ('test_mig_structure', 'Older chunk', 'old.md');
        """,
    )
    try:
        await migrations.ensure_schema(db_url)

        conn = await asyncpg.connect(db_url)
        try:
            assert await conn.fetchval(
                "SELECT to_regclass('strata_vectors.document_sections') IS NOT NULL"
            )
            clause_path = await conn.fetchval(
                "SELECT clause_path FROM strata_vectors.chunks "
                "WHERE layer_name = 'test_mig_structure'"
            )
        finally:
            await conn.close()
        assert clause_path is None
        assert await _recorded(db_url) == ALL_VERSIONS
    finally:
        await _owner_execute(
            db_url,
            "DELETE FROM strata_vectors.chunks WHERE layer_name = 'test_mig_structure'",
        )


async def test_applies_missing_10_receipts(db_url, restore_schema):
    await _owner_execute(
        db_url,
        """
        DROP TABLE strata_core.receipt_documents;
        DROP TABLE strata_core.receipts;
        DELETE FROM strata_core.schema_migrations WHERE version = 10;
        """,
    )
    await migrations.ensure_schema(db_url)

    conn = await asyncpg.connect(db_url)
    try:
        assert await conn.fetchval(
            "SELECT to_regclass('strata_core.receipts') IS NOT NULL "
            "AND to_regclass('strata_core.receipt_documents') IS NOT NULL"
        )
    finally:
        await conn.close()
    assert await _recorded(db_url) == ALL_VERSIONS


async def test_concurrent_calls(db_url, restore_schema):
    await _owner_execute(db_url, "DROP TABLE strata_core.schema_migrations")
    await asyncio.gather(migrations.ensure_schema(db_url), migrations.ensure_schema(db_url))
    assert await _recorded(db_url) == ALL_VERSIONS


async def test_broken_file_raises_naming_it(db_url):
    with (
        _with_extra(9001, "90_broken.sql", "SELEC 1"),
        pytest.raises(migrations.MigrationError, match="90_broken.sql") as exc,
    ):
        await migrations.ensure_schema(db_url)
    assert "psql -f" in str(exc.value)
    assert db_url not in migrations._checked
    assert 9001 not in await _recorded(db_url)


async def test_still_missing_file_has_generic_message(db_url):
    with (
        _with_extra(9003, "93_noop.sql", "SELECT 1"),
        pytest.raises(migrations.MigrationError, match="93_noop.sql") as exc,
    ):
        await migrations.ensure_schema(db_url)
    message = str(exc.value)
    assert "ran without an error" in message
    assert "only the owner of the tables" not in message.lower()
    assert db_url not in migrations._checked
    assert 9003 not in await _recorded(db_url)


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
        assert events[0]["versions"] == sorted(ALL_VERSIONS)
        assert readonly_url in migrations._checked
    finally:
        await migrations.ensure_schema(db_url)
    assert await _recorded(db_url) == ALL_VERSIONS


async def test_non_owner_schema_missing(db_url, readonly_url):
    await migrations.ensure_schema(db_url)
    migrations._checked.clear()
    with (
        _with_extra(9002, "92_probe.sql", "CREATE TABLE strata_core.migration_probe (id INT)"),
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
    "strata_vectors.documents",
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
    return next(m.presence for m in migrations.MIGRATIONS if m.version == version)


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
    for t in (
        "strata_core.audit_log",
        "strata_core.users",
        "strata_core.schema_migrations",
        "strata_core.receipts",
        "strata_core.receipt_documents",
    ):
        assert tables[t] is False
    learning = [t for t in tables if t.startswith("strata_learning.")]
    assert learning
    assert not any(tables[t] for t in learning)
    assert privileges["learning_usage"] is False
    assert privileges["default_acls"] == 0


async def test_readonly_grants_migrate_old_install(db_url, readonly_role, restore_schema):
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
        for table in (
            "strata_vectors.chunks",
            "strata_vectors.documents",
            "strata_core.knowledge_layers",
        ):
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


@pytest.fixture
async def probe_url(db_url):
    """URL for a new login role with no privileges on the StrataKB schemas."""
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
        await _owner_execute(db_url, drop)
        await _owner_execute(db_url, f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}'")
    except asyncpg.InsufficientPrivilegeError:
        pytest.skip("the owner may not create roles")
    try:
        netloc = f"{role}:{password}@{parts.hostname}:{parts.port}"
        yield urlunsplit(parts._replace(netloc=netloc))
    finally:
        await _owner_execute(db_url, drop)


async def test_repairs_01_without_02(fresh_url, readonly_role):
    # 01 and 04 with 06's grants in place, so presence 6 reaches the
    # strata_learning check instead of stopping at a missing table grant
    await _owner_execute(fresh_url, migrations._load_sql("01_init.sql"))
    await _owner_execute(fresh_url, migrations._load_sql("04_ingested_files.sql"))
    await _owner_execute(
        fresh_url,
        """
        GRANT USAGE ON SCHEMA strata_core, strata_vectors TO strata_readonly;
        GRANT SELECT ON strata_core.knowledge_layers, strata_core.ingestion_runs,
            strata_core.ingested_files, strata_vectors.chunks TO strata_readonly;
        """,
    )

    await migrations.ensure_schema(fresh_url)

    assert await _recorded(fresh_url) == ALL_VERSIONS
    conn = await asyncpg.connect(fresh_url)
    try:
        assert await conn.fetchval(
            "SELECT to_regclass('strata_learning.query_sessions') IS NOT NULL"
        )
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
    finally:
        await conn.close()
    assert readable == set(_READABLE)


async def test_held_advisory_lock_times_out(db_url):
    await migrations.ensure_schema(db_url)
    migrations._checked.clear()
    holder = await asyncpg.connect(db_url)
    try:
        await holder.execute("SELECT pg_advisory_lock($1)", migrations._MIGRATION_LOCK_KEY)
        started = time.monotonic()
        with (
            patch.object(migrations, "_TIMEOUT_S", 1),
            pytest.raises(migrations.MigrationError, match="schema update lock"),
        ):
            await migrations.ensure_schema(db_url)
        assert time.monotonic() - started < 10
        assert db_url not in migrations._checked
    finally:
        await holder.execute("SELECT pg_advisory_unlock($1)", migrations._MIGRATION_LOCK_KEY)
        await holder.close()


async def test_table_lock_names_the_file(db_url):
    await migrations.ensure_schema(db_url)
    migrations._checked.clear()
    await _owner_execute(
        db_url,
        """
        ALTER TABLE strata_core.knowledge_layers DROP COLUMN answer_mode;
        DELETE FROM strata_core.schema_migrations WHERE version = 12;
        """,
    )
    holder = await asyncpg.connect(db_url)
    tx = holder.transaction()
    try:
        await tx.start()
        await holder.execute("LOCK TABLE strata_core.knowledge_layers IN ACCESS SHARE MODE")
        with (
            patch.object(migrations, "_TIMEOUT_S", 1),
            pytest.raises(migrations.MigrationError, match="12_answer_mode.sql") as exc,
        ):
            await migrations.ensure_schema(db_url)
        assert "table lock" in str(exc.value)
    finally:
        await tx.rollback()
        await holder.close()
        migrations._checked.clear()
        await migrations.ensure_schema(db_url)
    assert 12 in await _recorded(db_url)


async def test_missing_usage_names_the_privilege(db_url, probe_url):
    await migrations.ensure_schema(db_url)
    migrations._checked.clear()
    with pytest.raises(migrations.MigrationError) as exc:
        await migrations.ensure_schema(probe_url)
    message = str(exc.value)
    assert "USAGE" in message
    assert "strata_core" in message
    assert "04_ingested_files" not in message
    assert probe_url not in migrations._checked


async def test_unrevokable_06_is_refused(db_url, readonly_role, probe_url):
    await migrations.ensure_schema(db_url)
    role = urlsplit(probe_url).username
    await _owner_execute(
        db_url,
        f"""
        GRANT USAGE ON SCHEMA strata_core, strata_vectors, strata_learning TO "{role}";
        GRANT SELECT ON ALL TABLES IN SCHEMA strata_core, strata_vectors, strata_learning
            TO "{role}";
        GRANT INSERT ON strata_core.schema_migrations TO "{role}";
        GRANT SELECT ON strata_core.audit_log TO strata_readonly;
        DELETE FROM strata_core.schema_migrations WHERE version = 6;
        """,
    )
    try:
        migrations._checked.clear()
        with pytest.raises(migrations.MigrationError, match="06_readonly_grants.sql") as exc:
            await migrations.ensure_schema(probe_url)
        # The re-check after the file, not the older out-of-date branch
        assert "ran without an error" in str(exc.value)
        assert "may not update it" not in str(exc.value)
        # The probe role owns nothing, so the owner advice applies
        assert "only the owner of the tables" in str(exc.value).lower()
        assert "psql -f" in str(exc.value)
        assert probe_url not in migrations._checked
        assert 6 not in await _recorded(db_url)
        conn = await asyncpg.connect(db_url)
        try:
            assert await conn.fetchval(
                "SELECT has_table_privilege('strata_readonly', 'strata_core.audit_log', 'SELECT')"
            )
        finally:
            await conn.close()
    finally:
        migrations._checked.clear()
        await migrations.ensure_schema(db_url)
    assert 6 in await _recorded(db_url)


async def test_readonly_presence_without_ingested_files(db_url, readonly_role):
    conn = await asyncpg.connect(db_url)
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            # Without its grant the renamed table is not readable, so the
            # missing strata_core.ingested_files is the only reason for false
            await conn.execute(
                "ALTER TABLE strata_core.ingested_files RENAME TO ingested_files_probe;"
                "REVOKE ALL ON strata_core.ingested_files_probe FROM strata_readonly"
            )
            assert await conn.fetchval(_presence_query(6)) is False
        finally:
            await tx.rollback()
    finally:
        await conn.close()


async def _documents_readable(db_url: str) -> bool:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval(
            "SELECT has_table_privilege('strata_readonly', 'strata_vectors.documents', 'SELECT')"
        )
    finally:
        await conn.close()


async def test_readonly_presence_06_does_not_require_documents(db_url, readonly_role):
    await migrations.ensure_schema(db_url)
    conn = await asyncpg.connect(db_url)
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            # As on a new database when 06 runs: 07 has not created the table yet
            await conn.execute(
                "ALTER TABLE strata_vectors.documents RENAME TO documents_probe;"
                "REVOKE ALL ON strata_vectors.documents_probe FROM strata_readonly"
            )
            assert await conn.fetchval(_presence_query(6)) is True
            assert await conn.fetchval(_presence_query(8)) is False
        finally:
            await tx.rollback()
    finally:
        await conn.close()


async def test_applies_missing_08_grants_documents(db_url, readonly_role, restore_schema):
    await migrations.ensure_schema(db_url)
    await _owner_execute(
        db_url,
        """
        REVOKE SELECT ON strata_vectors.documents FROM strata_readonly;
        DELETE FROM strata_core.schema_migrations WHERE version = 8;
        """,
    )
    conn = await asyncpg.connect(db_url)
    try:
        assert await conn.fetchval(_presence_query(8)) is False
    finally:
        await conn.close()

    migrations._checked.clear()
    await migrations.ensure_schema(db_url)

    assert await _documents_readable(db_url)
    assert 8 in await _recorded(db_url)
    _assert_closed(await _readonly_privileges(db_url))


async def test_rerun_06_keeps_documents_grant(db_url, readonly_role):
    await migrations.ensure_schema(db_url)
    await _owner_execute(db_url, migrations._load_sql("06_readonly_grants.sql"))

    assert await _documents_readable(db_url)
    conn = await asyncpg.connect(db_url)
    try:
        assert await conn.fetchval(_presence_query(6)) is True
        assert await conn.fetchval(_presence_query(8)) is True
    finally:
        await conn.close()


async def test_public_grant_is_named(db_url, readonly_role):
    await migrations.ensure_schema(db_url)
    await _owner_execute(
        db_url,
        """
        GRANT SELECT ON strata_core.audit_log TO PUBLIC;
        DELETE FROM strata_core.schema_migrations WHERE version = 6;
        """,
    )
    try:
        migrations._checked.clear()
        with pytest.raises(migrations.MigrationError, match="06_readonly_grants.sql") as exc:
            await migrations.ensure_schema(db_url)
        message = str(exc.value)
        # The docker user owns the tables, so the owner advice would mislead
        assert "PUBLIC" in message
        assert "pg_read_all_data" in message
        assert "only the owner of the tables" not in message.lower()
        assert "psql -f" not in message
    finally:
        await _owner_execute(db_url, "REVOKE SELECT ON strata_core.audit_log FROM PUBLIC")
        migrations._checked.clear()
        await migrations.ensure_schema(db_url)
    assert 6 in await _recorded(db_url)


async def test_readonly_presence_ignores_sequence_default_acl(db_url, readonly_role):
    await migrations.ensure_schema(db_url)
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA strata_core "
            "GRANT SELECT ON SEQUENCES TO strata_readonly"
        )
        # 06 revokes default privileges on tables only, so only those count
        assert await conn.fetchval(_presence_query(6)) is True
        await conn.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA strata_core "
            "GRANT SELECT ON TABLES TO strata_readonly"
        )
        assert await conn.fetchval(_presence_query(6)) is False
    finally:
        await conn.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA strata_core "
            "REVOKE SELECT ON SEQUENCES FROM strata_readonly;"
            "ALTER DEFAULT PRIVILEGES IN SCHEMA strata_core "
            "REVOKE SELECT ON TABLES FROM strata_readonly"
        )
        await conn.close()


async def test_schema_migrations_lock_names_the_table(db_url):
    await migrations.ensure_schema(db_url)
    # The table stage reads schema_migrations on every run
    migrations._checked.clear()
    holder = await asyncpg.connect(db_url)
    tx = holder.transaction()
    try:
        await tx.start()
        await holder.execute(
            "LOCK TABLE strata_core.schema_migrations IN ACCESS EXCLUSIVE MODE"
        )
        with (
            patch.object(migrations, "_TIMEOUT_S", 1),
            pytest.raises(
                migrations.MigrationError, match="strata_core.schema_migrations"
            ) as exc,
        ):
            await migrations.ensure_schema(db_url)
        assert "while reading" not in str(exc.value)
        assert db_url not in migrations._checked
    finally:
        await tx.rollback()
        await holder.close()
        migrations._checked.clear()


async def test_repairs_01_only(fresh_url):
    await _owner_execute(fresh_url, migrations._load_sql("01_init.sql"))

    await migrations.ensure_schema(fresh_url)

    assert await _recorded(fresh_url) == ALL_VERSIONS
    conn = await asyncpg.connect(fresh_url)
    try:
        assert await conn.fetchval(
            "SELECT to_regclass('strata_learning.query_sessions') IS NOT NULL"
        )
    finally:
        await conn.close()
