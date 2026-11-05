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
# by 06_readonly_grants.sql, 08_readonly_documents.sql and db/03_readonly_role.sql)
_STRATA_SCHEMAS = ("strata_core", "strata_vectors", "strata_learning")
_READONLY_TABLES = (
    ("strata_core", "knowledge_layers"),
    ("strata_core", "ingestion_runs"),
    ("strata_core", "ingested_files"),
    ("strata_vectors", "chunks"),
    ("strata_vectors", "documents"),
)
# The readable tables a migration after 06 creates; 08 grants them
_READONLY_TABLES_AFTER_06 = (("strata_vectors", "documents"),)
_STRATA_SCHEMA_LIST = ", ".join(f"'{schema}'" for schema in _STRATA_SCHEMAS)
# The schemas strata_readonly gets USAGE on, which are also the ones the
# presence queries resolve with to_regclass
_READABLE_SCHEMAS = tuple(dict.fromkeys(schema for schema, _ in _READONLY_TABLES))


@dataclass(frozen=True)
class Migration:
    """A numbered SQL file and the query that is true once its objects exist.

    not_in_place is added to the error when the file runs but leaves its
    presence query false. not_in_place_as_owner, when set, replaces it (and
    the advice to apply the file as the owner) when the connecting user
    already has the owner's privileges.
    """

    version: int
    file: str
    presence: str
    not_in_place: str = ""
    not_in_place_as_owner: str = ""


# The presence queries below are built once at import from the constants in
# this module, never from input.


def _pair_list(tables: tuple[tuple[str, str], ...]) -> str:
    return ", ".join(f"('{schema}', '{table}')" for schema, table in tables)


def _tables_present(tables: tuple[tuple[str, str], ...]) -> str:
    """Query that is true when every (schema, table) exists, read from pg_catalog.

    A pair must not repeat: count(*) would never reach len(tables).
    """
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
    Default privileges count on tables only, the kind 06 revokes. The tables
    a later migration creates are allowed but not required: 06 runs before
    they exist, and 08's query checks them.
    """
    guard = "\n            OR ".join(
        f"to_regnamespace('{schema}') IS NULL" for schema in _STRATA_SCHEMAS
    )
    readable = "\n                AND ".join(
        f"COALESCE(has_table_privilege(\n"
        f"                    r.oid, to_regclass('{schema}.{table}'), 'SELECT'), false)"
        for schema, table in _READONLY_TABLES
        if (schema, table) not in _READONLY_TABLES_AFTER_06
    )
    usage = "\n                AND ".join(
        ("" if schema in _READABLE_SCHEMAS else "NOT ")
        + f"has_schema_privilege(r.oid, '{schema}', 'USAGE')"
        for schema in _STRATA_SCHEMAS
    )
    return f"""
        SELECT CASE
        WHEN {guard} THEN false
        ELSE COALESCE((
            SELECT {readable}
                AND {usage}
                AND NOT EXISTS (
                    SELECT 1 FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname IN ({_STRATA_SCHEMA_LIST})
                      AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                      AND (n.nspname, c.relname) NOT IN ({_pair_list(_READONLY_TABLES)})
                      AND has_table_privilege(r.oid, c.oid, 'SELECT')
                )
                AND NOT EXISTS (
                    SELECT 1 FROM pg_default_acl d
                    JOIN pg_namespace n ON n.oid = d.defaclnamespace
                    CROSS JOIN LATERAL aclexplode(d.defaclacl) a
                    WHERE d.defaclobjtype = 'r'
                      AND n.nspname IN ({_STRATA_SCHEMA_LIST})
                      AND a.grantee = r.oid
                )
            FROM pg_roles r WHERE r.rolname = 'strata_readonly'
        ), true)
        END
        """


def _readonly_later_grants_present() -> str:
    """Query that is true when strata_readonly can read the tables 08 grants.

    True when the role does not exist. False while a table is missing, so 08
    is not recorded before 07 has created it: presence is read before any file
    runs. Stray grants are 06's check, not this one.
    """
    readable = "\n            AND ".join(
        f"COALESCE(has_table_privilege(\n"
        f"                r.oid, to_regclass('{schema}.{table}'), 'SELECT'), false)"
        for schema, table in _READONLY_TABLES_AFTER_06
    )
    return f"""
        SELECT COALESCE((
            SELECT {readable}
            FROM pg_roles r WHERE r.rolname = 'strata_readonly'
        ), true)
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
        not_in_place_as_owner=(
            "The user in DATABASE_URL owns the StrataKB tables, so the cause is access "
            "06 does not remove: strata_readonly still gets it from a PUBLIC grant or "
            "from a role it belongs to (such as pg_read_all_data). Revoke that by hand, "
            "then start StrataKB again."
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
    Migration(8, "08_readonly_documents.sql", _readonly_later_grants_present()),
    Migration(
        9,
        "09_structure.sql",
        """
        SELECT to_regclass('strata_vectors.document_sections') IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'strata_vectors'
                  AND table_name = 'chunks'
                  AND column_name = 'clause_path'
            )
        """,
    ),
    Migration(
        10,
        "10_receipts.sql",
        _tables_present(
            (
                ("strata_core", "receipts"),
                ("strata_core", "receipt_documents"),
            )
        ),
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

# True when current_user has the privileges of the owner of every StrataKB
# schema, every table-like relation in them and every role that set default
# privileges there (a superuser always has). Reads pg_catalog only.
_OWNS_STRATA_OBJECTS = f"""
    SELECT COALESCE(bool_and(pg_has_role(current_user, o.owner, 'USAGE')), false) FROM (
        SELECT n.nspowner AS owner FROM pg_catalog.pg_namespace n
        WHERE n.nspname IN ({_STRATA_SCHEMA_LIST})
        UNION ALL
        SELECT c.relowner FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname IN ({_STRATA_SCHEMA_LIST}) AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
        UNION ALL
        SELECT d.defaclrole FROM pg_catalog.pg_default_acl d
        JOIN pg_catalog.pg_namespace n ON n.oid = d.defaclnamespace
        WHERE n.nspname IN ({_STRATA_SCHEMA_LIST})
    ) o
    """

# Database URLs whose schema is known to be current in this process
_checked: set[str] = set()


class MigrationError(Exception):
    """The schema is out of date and could not be updated."""


class _Stage(enum.Enum):
    """Where ensure_schema is, for its error messages."""

    LOCK = enum.auto()  # taking the advisory lock
    READ = enum.auto()  # running the presence queries
    TABLE = enum.auto()  # creating or reading strata_core.schema_migrations
    FILE = enum.auto()  # applying or recording a migration


@dataclass(frozen=True)
class _StageErrors:
    """The messages for one stage of ensure_schema, by what went wrong.

    Placeholders: {timeout}, {error} (the exception), {file} and {fix}
    (_owner_fix(file)). str.format does not re-read substituted values,
    so braces in an error text or a path are safe.
    """

    lock: str  # asyncpg.LockNotAvailableError
    refused: str  # _REFUSED
    failed: str  # any other asyncpg.PostgresError


# Presence queries only read, so a refusal there is a missing USAGE
_USAGE_REFUSED = (
    "Could not read the database schema: {error}. The user in DATABASE_URL "
    "needs USAGE on the "
    + " and ".join(_READABLE_SCHEMAS)
    + " schemas; connect as the database owner or grant it."
)
_MAY_NOT_UPDATE = "The database schema is out of date and this user may not update it. {fix}"
_READ_FAILED = "Could not read the database schema: {error}."

_STAGE_ERRORS: dict[_Stage, _StageErrors] = {
    _Stage.LOCK: _StageErrors(
        lock=(
            "Timed out after {timeout} s waiting for the schema update lock: "
            "another StrataKB process is updating this database's schema. "
            "Try again when it has finished."
        ),
        refused=_USAGE_REFUSED,
        failed=_READ_FAILED,
    ),
    _Stage.READ: _StageErrors(
        lock=(
            "Timed out after {timeout} s waiting for a lock while reading the "
            "database schema. Try again when other sessions have finished."
        ),
        refused=_USAGE_REFUSED,
        failed=_READ_FAILED,
    ),
    _Stage.TABLE: _StageErrors(
        lock=(
            "Timed out after {timeout} s waiting for a lock on "
            "strata_core.schema_migrations: another session is using it (a long "
            "query or an open transaction). Try again when it has finished."
        ),
        refused=_MAY_NOT_UPDATE,
        failed="Creating strata_core.schema_migrations failed: {error}. {fix}",
    ),
    _Stage.FILE: _StageErrors(
        lock=(
            "Timed out after {timeout} s waiting for a table lock while "
            "applying {file}: another session is using a table this file "
            "changes (a long query or an open transaction). Try again when it "
            "has finished, or {fix}"
        ),
        refused=_MAY_NOT_UPDATE,
        failed="Schema migration {file} failed: {error}. {fix}",
    ),
}


def _load_sql(file_name: str) -> str:
    return (SQL_DIR / file_name).read_text(encoding="utf-8")


def _owner_fix(file_name: str) -> str:
    return f"Apply {SQL_DIR / file_name} as the database owner (psql -f)."


def _error_for(stage: _Stage, exc: asyncpg.PostgresError, current: str) -> MigrationError:
    """The MigrationError for exc raised at stage, with current as the file to fix."""
    errors = _STAGE_ERRORS[stage]
    if isinstance(exc, asyncpg.LockNotAvailableError):
        template = errors.lock
    elif isinstance(exc, _REFUSED):
        template = errors.refused
    else:
        template = errors.failed
    return MigrationError(
        template.format(timeout=_TIMEOUT_S, error=exc, file=current, fix=_owner_fix(current))
    )


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
                current = "01_init.sql"  # creates strata_core
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
                            not_in_place = (
                                f"{m.file} ran without an error, but its changes are "
                                "not in place."
                            )
                            if m.not_in_place_as_owner and await conn.fetchval(
                                _OWNS_STRATA_OBJECTS
                            ):
                                raise MigrationError(
                                    f"{not_in_place} {m.not_in_place_as_owner}"
                                )
                            detail = f" {m.not_in_place}" if m.not_in_place else ""
                            raise MigrationError(
                                f"{not_in_place}{detail} {_owner_fix(m.file)}"
                            )
                        logger.info("migration_applied", version=m.version, file=m.file)
                    await conn.execute(
                        "INSERT INTO strata_core.schema_migrations (version, name) "
                        "VALUES ($1, $2) ON CONFLICT (version) DO NOTHING",
                        m.version,
                        m.file,
                    )
        except _REFUSED as e:
            # The transaction has rolled back. In TABLE and FILE every presence
            # query has run, before any write.
            if stage in (_Stage.TABLE, _Stage.FILE):
                if all(present.values()):
                    logger.warning(
                        "migration_not_recorded",
                        versions=[m.version for m in migrations if m.version not in recorded],
                    )
                    _checked.add(database_url)
                    return
                current = next(m.file for m in migrations if not present[m.version])
            raise _error_for(stage, e, current) from e
        except asyncpg.PostgresError as e:
            raise _error_for(stage, e, current) from e
    finally:
        await conn.close()

    _checked.add(database_url)
