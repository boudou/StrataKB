"""Numbered schema migrations applied by the server itself.

An empty database gets every file here, in version order, at startup or on
first use; a database created before a schema change gets only the missing
ones, so neither needs a manual SQL step. Fresh Docker volumes also get the
files from docker-compose. Applied versions are recorded in
strata_core.schema_migrations.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import structlog

logger = structlog.get_logger()

SQL_DIR = Path(__file__).parent / "sql"

# Serializes concurrent runners across processes ("STRATAKB" in ASCII)
_MIGRATION_LOCK_KEY = 0x5354524154414B42

# Seconds to wait for the connection and for each lock (the advisory lock or
# a table lock a file takes); as the pool's command_timeout in core/database.py.
# A lock timeout, not a statement timeout: 07 may build an index for longer.
_TIMEOUT_S = 30

# The StrataKB schemas, and the tables strata_readonly may read (as granted
# by 06_readonly_grants.sql and db/03_readonly_role.sql)
_STRATA_SCHEMAS = ("strata_core", "strata_vectors", "strata_learning")
_READONLY_TABLES = (
    ("strata_core", "knowledge_layers"),
    ("strata_core", "ingestion_runs"),
    ("strata_core", "ingested_files"),
    ("strata_vectors", "chunks"),
)


@dataclass(frozen=True)
class Migration:
    """A numbered SQL file and the query that is true once its objects exist.

    not_in_place is added to the error when the file runs but leaves its
    presence query false.
    """

    version: int
    file: str
    presence: str
    not_in_place: str = ""


# The presence queries below are built once at import from the constants in
# this module, never from input. A pair must not repeat: count(*) would never
# reach len(tables).


def _pair_list(tables: tuple[tuple[str, str], ...]) -> str:
    return ", ".join(f"('{schema}', '{table}')" for schema, table in tables)


def _tables_present(tables: tuple[tuple[str, str], ...]) -> str:
    """Query that is true when every (schema, table) exists, read from pg_catalog."""
    return f"""
        SELECT count(*) = {len(tables)} FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind IN ('r', 'p')
          AND (n.nspname, c.relname) IN ({_pair_list(tables)})
        """


def _readonly_grants_present() -> str:
    """Query that is true when strata_readonly holds exactly 06's privileges.

    Uses the role's OID, so a database without strata_readonly reads as
    present and records 06 without running it. The role is cluster-wide, so it
    can exist where the schemas don't yet: has_schema_privilege would raise,
    hence the CASE guard over every schema the query names. has_table_privilege
    returns NULL for a missing table, so each check keeps its own COALESCE.
    Default privileges count on tables only, the kind 06 revokes.
    """
    schemas = ", ".join(f"'{schema}'" for schema in _STRATA_SCHEMAS)
    guard = "\n            OR ".join(
        f"to_regnamespace('{schema}') IS NULL" for schema in _STRATA_SCHEMAS
    )
    readable = "\n                AND ".join(
        f"COALESCE(has_table_privilege(\n"
        f"                    r.oid, to_regclass('{schema}.{table}'), 'SELECT'), false)"
        for schema, table in _READONLY_TABLES
    )
    return f"""
        SELECT CASE
        WHEN {guard} THEN false
        ELSE COALESCE((
            SELECT {readable}
                AND has_schema_privilege(r.oid, 'strata_core', 'USAGE')
                AND has_schema_privilege(r.oid, 'strata_vectors', 'USAGE')
                AND NOT has_schema_privilege(r.oid, 'strata_learning', 'USAGE')
                AND NOT EXISTS (
                    SELECT 1 FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname IN ({schemas})
                      AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                      AND (n.nspname, c.relname) NOT IN ({_pair_list(_READONLY_TABLES)})
                      AND has_table_privilege(r.oid, c.oid, 'SELECT')
                )
                AND NOT EXISTS (
                    SELECT 1 FROM pg_default_acl d
                    JOIN pg_namespace n ON n.oid = d.defaclnamespace
                    CROSS JOIN LATERAL aclexplode(d.defaclacl) a
                    WHERE d.defaclobjtype = 'r'
                      AND n.nspname IN ({schemas})
                      AND a.grantee = r.oid
                )
            FROM pg_roles r WHERE r.rolname = 'strata_readonly'
        ), true)
        END
        """


# Keep sorted by version: 05 records 04 as well. Presence queries run before
# any file, so they must not fail on an older or empty schema. 01 and 02 read
# pg_catalog only: to_regclass raises when the user lacks USAGE on the schema
# (strata_readonly on strata_learning).
MIGRATIONS: list[Migration] = [
    Migration(
        1,
        "01_init.sql",
        _tables_present(
            (
                ("strata_core", "knowledge_layers"),
                ("strata_core", "ingestion_runs"),
                ("strata_core", "users"),
                ("strata_core", "audit_log"),
                ("strata_vectors", "chunks"),
            )
        ),
    ),
    Migration(
        2,
        "02_learning.sql",
        _tables_present(
            (
                ("strata_learning", "query_sessions"),
                ("strata_learning", "interaction_events"),
                ("strata_learning", "feedback_signals"),
                ("strata_learning", "learned_rewrites"),
                ("strata_learning", "query_synonyms"),
                ("strata_learning", "layer_affinity_scores"),
            )
        ),
    ),
    Migration(
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
    Migration(
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
    # REVOKE and GRANT only warn when this user may not change a privilege,
    # and 06 revokes direct grants only
    Migration(
        6,
        "06_readonly_grants.sql",
        _readonly_grants_present(),
        not_in_place=(
            "Only the owner of the tables or a superuser can change these privileges. "
            "06 does not remove access that strata_readonly gets from a PUBLIC grant "
            "or from a role it belongs to (such as pg_read_all_data): revoke that by hand."
        ),
    ),
    Migration(
        7,
        "07_documents.sql",
        """
        SELECT to_regclass('strata_vectors.documents') IS NOT NULL
            AND (
                SELECT count(*) = 3 FROM information_schema.columns
                WHERE table_schema = 'strata_vectors'
                  AND table_name = 'chunks'
                  AND column_name IN ('document_id', 'char_start', 'char_end')
            )
        """,
    ),
    Migration(
        12,
        "12_answer_mode.sql",
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'strata_core'
              AND table_name = 'knowledge_layers'
              AND column_name = 'answer_mode'
        )
        """,
    ),
]

# Same definition as in 05_add_folder.sql (tests compare the two)
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


class _Stage(enum.Enum):
    """Where ensure_schema is, for its error messages."""

    LOCK = "lock"  # taking the advisory lock
    READ = "read"  # running the presence queries
    TABLE = "table"  # creating or reading strata_core.schema_migrations
    FILE = "file"  # applying or recording a migration


def _load_sql(file_name: str) -> str:
    return (SQL_DIR / file_name).read_text(encoding="utf-8")


def _owner_fix(file_name: str) -> str:
    return f"Apply {SQL_DIR / file_name} as the database owner (psql -f)."


async def ensure_schema(database_url: str) -> None:
    """Apply and record every numbered migration the database is missing.

    Runs once per process and database URL, in one transaction under an
    advisory lock. A migration whose objects already exist is recorded
    without running its file; a file that runs must leave its presence query
    true. Connection errors (a connect timeout included) propagate unchanged.

    Raises:
        MigrationError: If a migration fails, if a lock is not granted within
            _TIMEOUT_S seconds, if the schema cannot be read, or if the schema
            is out of date and this user may not update it.
    """
    if database_url in _checked:
        return

    migrations = list(MIGRATIONS)
    conn = await asyncpg.connect(database_url, timeout=_TIMEOUT_S)
    try:
        present: dict[int, bool] = {}
        recorded: set[int] = set()
        stage = _Stage.LOCK
        current = ""
        try:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL lock_timeout = '{_TIMEOUT_S}s'")
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK_KEY)
                stage = _Stage.READ
                for m in migrations:
                    present[m.version] = bool(await conn.fetchval(m.presence))

                stage = _Stage.TABLE
                # DDL only for what is missing: CREATE ... IF NOT EXISTS checks the
                # CREATE privilege first, and strata_readonly may run no DDL at all.
                # The advisory lock keeps other StrataKB processes from racing.
                missing = await conn.fetchrow(
                    "SELECT to_regnamespace('strata_core') IS NULL AS no_schema, "
                    "to_regclass('strata_core.schema_migrations') IS NULL AS no_table"
                )
                if missing["no_schema"]:
                    await conn.execute("CREATE SCHEMA strata_core")
                if missing["no_table"]:
                    await conn.execute(_SCHEMA_MIGRATIONS_DDL)
                recorded = {
                    row["version"]
                    for row in await conn.fetch(
                        "SELECT version FROM strata_core.schema_migrations"
                    )
                }

                stage = _Stage.FILE
                for m in migrations:
                    if m.version in recorded:
                        continue
                    current = m.file
                    if present[m.version]:
                        logger.info("migration_recorded", version=m.version, file=m.file)
                    else:
                        logger.info("applying_migration", version=m.version, file=m.file)
                        await conn.execute(_load_sql(m.file))
                        # A file can run without effect (REVOKE and GRANT only
                        # warn when this user may not change a privilege)
                        if not await conn.fetchval(m.presence):
                            detail = f" {m.not_in_place}" if m.not_in_place else ""
                            raise MigrationError(
                                f"{m.file} ran without an error, but its changes are not "
                                f"in place.{detail} {_owner_fix(m.file)}"
                            )
                        logger.info("migration_applied", version=m.version, file=m.file)
                    await conn.execute(
                        "INSERT INTO strata_core.schema_migrations (version, name) "
                        "VALUES ($1, $2) ON CONFLICT (version) DO NOTHING",
                        m.version,
                        m.file,
                    )
        except asyncpg.LockNotAvailableError as e:
            if stage is _Stage.LOCK:
                raise MigrationError(
                    f"Timed out after {_TIMEOUT_S} s waiting for the schema update lock: "
                    "another StrataKB process is updating this database's schema. "
                    "Try again when it has finished."
                ) from e
            if stage is _Stage.TABLE:
                raise MigrationError(
                    f"Timed out after {_TIMEOUT_S} s waiting for a lock on "
                    "strata_core.schema_migrations: another session is using it (a long "
                    "query or an open transaction). Try again when it has finished."
                ) from e
            if stage is _Stage.FILE:
                raise MigrationError(
                    f"Timed out after {_TIMEOUT_S} s waiting for a table lock while "
                    f"applying {current}: another session is using a table this file "
                    "changes (a long query or an open transaction). Try again when it "
                    f"has finished, or {_owner_fix(current)}"
                ) from e
            raise MigrationError(
                f"Timed out after {_TIMEOUT_S} s waiting for a lock while reading the "
                "database schema. Try again when other sessions have finished."
            ) from e
        except _REFUSED as e:
            # The transaction has rolled back
            if stage in (_Stage.LOCK, _Stage.READ):
                # Presence queries only read, so this is a missing USAGE
                raise MigrationError(
                    f"Could not read the database schema: {e}. The user in DATABASE_URL "
                    "needs USAGE on the strata_core and strata_vectors schemas; connect "
                    "as the database owner or grant it."
                ) from e
            # Every presence query has run, before any write
            if all(present.values()):
                logger.warning(
                    "migration_not_recorded",
                    versions=[m.version for m in migrations if m.version not in recorded],
                )
                _checked.add(database_url)
                return
            first_missing = next(m.file for m in migrations if not present[m.version])
            raise MigrationError(
                "The database schema is out of date and this user may not update it. "
                + _owner_fix(first_missing)
            ) from e
        except asyncpg.PostgresError as e:
            if stage in (_Stage.LOCK, _Stage.READ):
                raise MigrationError(f"Could not read the database schema: {e}.") from e
            if stage is _Stage.TABLE:
                raise MigrationError(
                    f"Creating strata_core.schema_migrations failed: {e}. "
                    + _owner_fix("01_init.sql")
                ) from e
            raise MigrationError(
                f"Schema migration {current} failed: {e}. {_owner_fix(current)}"
            ) from e
    finally:
        await conn.close()

    _checked.add(database_url)
