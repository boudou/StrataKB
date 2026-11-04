"""Numbered schema migrations applied by the server itself.

An empty database gets every file here, in version order, at startup or on
first use; a database created before a schema change gets only the missing
ones, so neither needs a manual SQL step. Fresh Docker volumes also get the
files from docker-compose. Applied versions are recorded in
strata_core.schema_migrations.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
import structlog

logger = structlog.get_logger()

SQL_DIR = Path(__file__).parent / "sql"

# Serializes concurrent runners across processes ("STRATAKB" in ASCII)
_MIGRATION_LOCK_KEY = 0x5354524154414B42

# (version, file name in SQL_DIR, query that is true when the file's objects
# already exist). Keep sorted by version: 05 records 04 as well. Presence
# queries run before any file, so they must not fail on an older or empty
# schema. 01 and 02 read pg_catalog only: to_regclass raises when the user
# lacks USAGE on the schema (strata_readonly on strata_learning).
MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "01_init.sql",
        """
        SELECT count(*) = 5 FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind IN ('r', 'p')
          AND (n.nspname, c.relname) IN (
              ('strata_core', 'knowledge_layers'),
              ('strata_core', 'ingestion_runs'),
              ('strata_core', 'users'),
              ('strata_core', 'audit_log'),
              ('strata_vectors', 'chunks')
          )
        """,
    ),
    (
        2,
        "02_learning.sql",
        """
        SELECT count(*) = 6 FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind IN ('r', 'p')
          AND n.nspname = 'strata_learning'
          AND c.relname IN (
              'query_sessions',
              'interaction_events',
              'feedback_signals',
              'learned_rewrites',
              'query_synonyms',
              'layer_affinity_scores'
          )
        """,
    ),
    (
        4,
        "04_ingested_files.sql",
        """
        SELECT to_regclass('strata_core.ingested_files') IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'strata_vectors'
                  AND table_name = 'chunks'
                  AND column_name = 'file_id'
            )
        """,
    ),
    (
        5,
        "05_add_folder.sql",
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'strata_core'
              AND table_name = 'knowledge_layers'
              AND column_name = 'root_path'
        )
        """,
    ),
    # Uses the role's OID, so a database without strata_readonly reads as
    # present and records 06 without running it. The role is cluster-wide, so
    # it can exist where the schemas don't yet: has_schema_privilege would raise.
    (
        6,
        "06_readonly_grants.sql",
        """
        SELECT CASE
        WHEN to_regnamespace('strata_core') IS NULL
            OR to_regnamespace('strata_vectors') IS NULL THEN false
        ELSE COALESCE((
            SELECT COALESCE(has_table_privilege(
                    r.oid, to_regclass('strata_core.knowledge_layers'), 'SELECT'), false)
                AND COALESCE(has_table_privilege(
                    r.oid, to_regclass('strata_core.ingestion_runs'), 'SELECT'), false)
                AND COALESCE(has_table_privilege(
                    r.oid, to_regclass('strata_core.ingested_files'), 'SELECT'), false)
                AND COALESCE(has_table_privilege(
                    r.oid, to_regclass('strata_vectors.chunks'), 'SELECT'), false)
                AND has_schema_privilege(r.oid, 'strata_core', 'USAGE')
                AND has_schema_privilege(r.oid, 'strata_vectors', 'USAGE')
                AND NOT has_schema_privilege(r.oid, 'strata_learning', 'USAGE')
                AND NOT EXISTS (
                    SELECT 1 FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname IN ('strata_core', 'strata_vectors', 'strata_learning')
                      AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                      AND (n.nspname, c.relname) NOT IN (
                          ('strata_core', 'knowledge_layers'),
                          ('strata_core', 'ingestion_runs'),
                          ('strata_core', 'ingested_files'),
                          ('strata_vectors', 'chunks')
                      )
                      AND has_table_privilege(r.oid, c.oid, 'SELECT')
                )
                AND NOT EXISTS (
                    SELECT 1 FROM pg_default_acl d
                    JOIN pg_namespace n ON n.oid = d.defaclnamespace
                    CROSS JOIN LATERAL aclexplode(d.defaclacl) a
                    WHERE n.nspname IN ('strata_core', 'strata_vectors', 'strata_learning')
                      AND a.grantee = r.oid
                )
            FROM pg_roles r WHERE r.rolname = 'strata_readonly'
        ), true)
        END
        """,
    ),
]

# Same definition as in 05_add_folder.sql
_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS strata_core.schema_migrations (
    version     INT PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

# Errors that mean "this user may not change the schema". A read-only role
# fails with ReadOnlySQLTransactionError before any privilege check.
_REFUSED = (asyncpg.InsufficientPrivilegeError, asyncpg.ReadOnlySQLTransactionError)

# Database URLs whose schema is known to be current in this process
_checked: set[str] = set()


class MigrationError(Exception):
    """The schema is out of date and could not be updated."""


def _load_sql(file_name: str) -> str:
    return (SQL_DIR / file_name).read_text(encoding="utf-8")


def _owner_fix(file_name: str) -> str:
    return f"Apply {SQL_DIR / file_name} as the database owner (psql -f)."


async def ensure_schema(database_url: str) -> None:
    """Apply and record every numbered migration the database is missing.

    Runs once per process and database URL, in one transaction under an
    advisory lock. A migration whose objects already exist is recorded
    without running its file. Connection errors propagate unchanged.

    Raises:
        MigrationError: If a migration fails, or if the schema is out of date
            and this user may not update it.
    """
    if database_url in _checked:
        return

    migrations = list(MIGRATIONS)
    conn = await asyncpg.connect(database_url)
    try:
        present: dict[int, bool] = {}
        recorded: set[int] = set()
        current = migrations[-1][1]
        try:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK_KEY)
                for version, _, query in migrations:
                    present[version] = bool(await conn.fetchval(query))

                if await conn.fetchval(
                    "SELECT to_regclass('strata_core.schema_migrations') IS NULL"
                ):
                    # Only on an empty database: CREATE SCHEMA IF NOT EXISTS checks
                    # the database's CREATE privilege even when the schema exists
                    if await conn.fetchval("SELECT to_regnamespace('strata_core') IS NULL"):
                        await conn.execute("CREATE SCHEMA IF NOT EXISTS strata_core")
                    await conn.execute(_SCHEMA_MIGRATIONS_DDL)
                recorded = {
                    row["version"]
                    for row in await conn.fetch(
                        "SELECT version FROM strata_core.schema_migrations"
                    )
                }

                for version, name, _ in migrations:
                    if version in recorded:
                        continue
                    current = name
                    if present[version]:
                        logger.info("migration_recorded", version=version, file=name)
                    else:
                        logger.info("applying_migration", version=version, file=name)
                        await conn.execute(_load_sql(name))
                        logger.info("migration_applied", version=version, file=name)
                    await conn.execute(
                        "INSERT INTO strata_core.schema_migrations (version, name) "
                        "VALUES ($1, $2) ON CONFLICT (version) DO NOTHING",
                        version,
                        name,
                    )
        except _REFUSED as e:
            # The transaction has rolled back; presence was read before any write
            if len(present) == len(migrations) and all(present.values()):
                logger.warning(
                    "migration_not_recorded",
                    versions=[v for v, _, _ in migrations if v not in recorded],
                )
                _checked.add(database_url)
                return
            missing = next((n for v, n, _ in migrations if not present.get(v, False)), current)
            raise MigrationError(
                "The database schema is out of date and this user may not update it. "
                + _owner_fix(missing)
            ) from e
        except asyncpg.PostgresError as e:
            raise MigrationError(
                f"Schema migration {current} failed: {e}. {_owner_fix(current)}"
            ) from e
    finally:
        await conn.close()

    _checked.add(database_url)
