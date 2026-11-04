"""Numbered schema migrations applied by the server itself.

Fresh Docker volumes get the schema files from docker-compose. A database
created before a schema change gets the missing files here, at startup or on
first use, so an upgrade needs no manual SQL step. Applied versions are
recorded in strata_core.schema_migrations.
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
# already exist). Keep sorted by version: 05 records 04 as well.
MIGRATIONS: list[tuple[int, str, str]] = [
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
