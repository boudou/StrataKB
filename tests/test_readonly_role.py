"""Tests for the strata_readonly role: its grants in the migrations and db/03.

The tests that use the role skip when it does not exist in the cluster.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import asyncpg
import pytest

from stratakb.core import migrations
from tests.helpers import (
    READONLY_READABLE,
    owner_execute,
    readonly_privileges,
    recorded_versions,
)

pytestmark = pytest.mark.usefixtures("clear_checked")

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
    assert set(READONLY_READABLE) == in_module
    assert in_readme == in_module
    assert _granted(migrations._load_sql("08_readonly_documents.sql"), "SELECT") == {
        f"{schema}.{table}" for schema, table in migrations._READONLY_TABLES_08
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
    await owner_execute(db_url, migrations._load_sql("06_readonly_grants.sql"))


def _readable(privileges: dict) -> set[str]:
    return {t for t, readable in privileges["tables"].items() if readable}


def _assert_closed(privileges: dict) -> None:
    tables = privileges["tables"]
    assert _readable(privileges) == set(READONLY_READABLE)
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
    await owner_execute(db_url, _OLD_GRANTS)
    conn = await asyncpg.connect(db_url)
    try:
        assert await conn.fetchval(_presence_query(6)) is False
    finally:
        await conn.close()

    await migrations.ensure_schema(db_url)

    _assert_closed(await readonly_privileges(db_url))
    assert 6 in await recorded_versions(db_url)


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
    await owner_execute(db_url, _OLD_GRANTS)
    try:
        await migrations.ensure_schema(db_url)
        await owner_execute(db_url, "CREATE TABLE strata_core.grant_probe (id INT)")
        conn = await asyncpg.connect(db_url)
        try:
            assert not await conn.fetchval(
                "SELECT has_table_privilege('strata_readonly', 'strata_core.grant_probe', 'SELECT')"
            )
        finally:
            await conn.close()
    finally:
        await owner_execute(db_url, "DROP TABLE IF EXISTS strata_core.grant_probe")


async def test_repairs_01_without_02(fresh_url, readonly_role):
    # 01 and 04 with 06's grants in place, so presence 6 reaches the
    # strata_learning check instead of stopping at a missing table grant
    await owner_execute(fresh_url, migrations._load_sql("01_init.sql"))
    await owner_execute(fresh_url, migrations._load_sql("04_ingested_files.sql"))
    await owner_execute(
        fresh_url,
        """
        GRANT USAGE ON SCHEMA strata_core, strata_vectors TO strata_readonly;
        GRANT SELECT ON strata_core.knowledge_layers, strata_core.ingestion_runs,
            strata_core.ingested_files, strata_vectors.chunks TO strata_readonly;
        """,
    )

    await migrations.ensure_schema(fresh_url)

    assert await recorded_versions(fresh_url) == {m.version for m in migrations.MIGRATIONS}
    conn = await asyncpg.connect(fresh_url)
    try:
        assert await conn.fetchval(
            "SELECT to_regclass('strata_learning.query_sessions') IS NOT NULL"
        )
    finally:
        await conn.close()
    assert _readable(await readonly_privileges(fresh_url)) == set(READONLY_READABLE)


async def test_unrevokable_06_is_refused(db_url, readonly_role, probe_url):
    await migrations.ensure_schema(db_url)
    role = urlsplit(probe_url).username
    await owner_execute(
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
        assert 6 not in await recorded_versions(db_url)
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
    assert 6 in await recorded_versions(db_url)


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
    await owner_execute(
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

    assert "strata_vectors.documents" in _readable(await readonly_privileges(db_url))
    assert 8 in await recorded_versions(db_url)
    _assert_closed(await readonly_privileges(db_url))


async def test_rerun_06_keeps_documents_grant(db_url, readonly_role):
    await migrations.ensure_schema(db_url)
    await owner_execute(db_url, migrations._load_sql("06_readonly_grants.sql"))

    assert "strata_vectors.documents" in _readable(await readonly_privileges(db_url))
    conn = await asyncpg.connect(db_url)
    try:
        assert await conn.fetchval(_presence_query(6)) is True
        assert await conn.fetchval(_presence_query(8)) is True
    finally:
        await conn.close()


async def test_public_grant_is_named(db_url, readonly_role):
    await migrations.ensure_schema(db_url)
    await owner_execute(
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
        await owner_execute(db_url, "REVOKE SELECT ON strata_core.audit_log FROM PUBLIC")
        migrations._checked.clear()
        await migrations.ensure_schema(db_url)
    assert 6 in await recorded_versions(db_url)


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


def _role_script_grants() -> str:
    """The grants of db/03_readonly_role.sql, without its psql line and role setup.

    The role is cluster-wide and already exists, so only the grants can run here.
    """
    script = (_REPO / "db" / "03_readonly_role.sql").read_text(encoding="utf-8")
    return script[script.index("GRANT USAGE ON SCHEMA"):]


async def test_role_script_grants_in_either_order(fresh_url, readonly_role):
    # On a fresh Docker volume the script runs before 04 and 07: its guarded
    # grants do nothing, and the migrations grant those tables afterwards
    await owner_execute(fresh_url, migrations._load_sql("01_init.sql"))
    await owner_execute(fresh_url, migrations._load_sql("02_learning.sql"))
    await owner_execute(fresh_url, _role_script_grants())
    assert _readable(await readonly_privileges(fresh_url)) == {
        "strata_core.knowledge_layers",
        "strata_core.ingestion_runs",
        "strata_vectors.chunks",
    }

    await migrations.ensure_schema(fresh_url)
    _assert_closed(await readonly_privileges(fresh_url))

    # In CI and after `stratakb db up` the migrations run first, while the role
    # does not exist, so they grant nothing and the guarded grants must
    await owner_execute(
        fresh_url,
        """
        REVOKE ALL ON ALL TABLES IN SCHEMA strata_core, strata_vectors FROM strata_readonly;
        REVOKE ALL ON SCHEMA strata_core, strata_vectors FROM strata_readonly;
        """,
    )
    await owner_execute(fresh_url, _role_script_grants())
    _assert_closed(await readonly_privileges(fresh_url))
    conn = await asyncpg.connect(fresh_url)
    try:
        assert await conn.fetchval(_presence_query(6)) is True
        assert await conn.fetchval(_presence_query(8)) is True
    finally:
        await conn.close()
