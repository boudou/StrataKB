"""Schema migrations applied by the server itself.

Fresh Docker volumes get every schema file from docker-compose. A database
created before a schema change gets the missing parts here, at startup or on
first use, so an upgrade needs no manual SQL step.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
import structlog

from stratakb.core.database import get_pool

logger = structlog.get_logger()

INGESTED_FILES_SQL = Path(__file__).parent / "sql" / "04_ingested_files.sql"

# Database URLs whose schema is known to be current in this process
_checked: set[str] = set()


class MigrationError(Exception):
    """The schema is out of date and could not be updated."""


async def ensure_ingestion_schema(database_url: str) -> None:
    """Apply 04_ingested_files.sql if its table or the chunks.file_id column is missing.

    Checks once per process and database URL.

    Raises:
        MigrationError: If the database user may not change the schema.
    """
    if database_url in _checked:
        return

    pool = await get_pool(database_url)
    missing = await pool.fetchval(
        """
        SELECT to_regclass('strata_core.ingested_files') IS NULL
            OR NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'strata_vectors'
                  AND table_name = 'chunks'
                  AND column_name = 'file_id'
            )
        """
    )
    if missing:
        logger.info("applying_migration", file=INGESTED_FILES_SQL.name)
        try:
            await pool.execute(INGESTED_FILES_SQL.read_text(encoding="utf-8"))
        except asyncpg.InsufficientPrivilegeError as e:
            raise MigrationError(
                f"The database schema is out of date and this user may not update it. "
                f"Run {INGESTED_FILES_SQL} as the database owner."
            ) from e
        logger.info("migration_applied", file=INGESTED_FILES_SQL.name)

    _checked.add(database_url)
