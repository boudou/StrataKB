"""Database access MCP tools for development and debugging.

Provides read-only SQL execution, schema introspection, and vector stats.
"""

from __future__ import annotations

import asyncio
import contextlib
import re

import asyncpg
import structlog
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import require_role
from stratakb.core.logging import log_tool_call

logger = structlog.get_logger()

# Built-in roles that reach the server's files, programs or other backends.
# A read-only role that is a member of any of them is refused.
_SERVER_ACCESS_ROLES = (
    "pg_execute_server_program",
    "pg_read_server_files",
    "pg_signal_backend",
    "pg_write_server_files",
)

# One row: whether session_user or current_user is, or is a member of, a
# superuser role, and which server-access roles either of them is a member
# of. 'MEMBER' covers direct and indirect membership, including INHERIT
# FALSE; a superuser is a member of every role, so it is caught too.
_ROLE_CHECK_QUERY = (
    "SELECT bool_or(rolsuper) AS superuser, "
    "array_agg(rolname ORDER BY rolname) "
    "FILTER (WHERE rolname = ANY($1::text[])) AS members "
    "FROM pg_roles "
    "WHERE pg_has_role(session_user, oid, 'MEMBER') "
    "OR pg_has_role(current_user, oid, 'MEMBER')"
)

# The free-form SQL tools. They run only as the role behind
# readonly_database_url, never as the owner role behind database_url.
_SQL_TOOLS = ("strata_db_query", "strata_db_explain")

# Seconds the whole role check at startup may take
_CHECK_TIMEOUT = 5

# Whether the SQL tools are on, and why not (read by strata_admin_health)
_sql_tools: dict[str, str] = {"status": "disabled", "reason": "not registered"}

# SQL statements that modify data or schema. This only gives an early, friendly
# error; the guard is the read-only transaction in _fetch_read_only and the
# grants of the role behind readonly_database_url, which secure_sql_tools
# checks at startup.
_FORBIDDEN_PATTERNS = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE|GRANT|REVOKE|"
    r"COPY|VACUUM|REINDEX|CLUSTER|COMMENT|LOCK|DISCARD|REASSIGN|"
    r"SET\s+ROLE|RESET\s+ROLE)\b",
    re.IGNORECASE,
)

# Max rows returned
_MAX_ROWS = 1000
# Query timeout in seconds
_QUERY_TIMEOUT = 30


def _validate_read_only(sql: str) -> str | None:
    """Check if SQL is read-only. Returns error message if not."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return "SQL query cannot be empty."

    match = _FORBIDDEN_PATTERNS.search(stripped)
    if match:
        return f"Write operation not allowed: {match.group(0).upper()} statements are blocked. Use read-only queries (SELECT, EXPLAIN, SHOW, WITH)."

    return None


async def _fetch_read_only(
    pool: asyncpg.Pool, sql: str, args: list[str], limit: int
) -> list[asyncpg.Record]:
    """Run one statement in a read-only transaction and return at most `limit` rows.

    The transaction is always rolled back, which also undoes any SET or
    set_config() the statement made, so nothing sticks to the pooled
    connection. Rows come from a cursor, so a large result is never fetched
    in full.
    """
    async with pool.acquire() as conn:
        tr = conn.transaction(readonly=True)
        await tr.start()
        try:
            await conn.execute(f"SET LOCAL statement_timeout = '{_QUERY_TIMEOUT}s'")
            cursor = await conn.cursor(sql, *args)
            return await cursor.fetch(limit)
        finally:
            await tr.rollback()


def _query_error(e: Exception) -> str:
    """Turn a database error into a message that says how to fix the query."""
    if isinstance(e, asyncpg.ReadOnlySQLTransactionError):
        return f"Write operation not allowed: the query runs in a read-only transaction ({e})."
    if isinstance(e, asyncpg.QueryCanceledError):
        return f"Query cancelled after {_QUERY_TIMEOUT}s. Add a WHERE clause or a LIMIT."
    return str(e)


def sql_tools_status() -> dict[str, str]:
    """Return whether the SQL tools are on and, when off, why."""
    return dict(_sql_tools)


def _set_sql_tools_status(status: str, reason: str | None = None) -> None:
    """Record whether the SQL tools are on and, when off, why."""
    _sql_tools.clear()
    _sql_tools["status"] = status
    if reason is not None:
        _sql_tools["reason"] = reason


def _disable_sql_tools(mcp: FastMCP, reason: str) -> None:
    """Remove the SQL tools that are still registered and record why."""
    for name in _SQL_TOOLS:
        # Not registered: no URL, or removed by an earlier check
        with contextlib.suppress(ToolError):
            mcp.remove_tool(name)
    _set_sql_tools_status("disabled", reason)
    logger.warning("sql_tools_disabled", reason=reason)


async def _unsafe_role_reason(query_database_url: str) -> str | None:
    """Say why the role behind `query_database_url` is unsafe, or None.

    Both session_user and current_user are checked, together with every role
    they are members of: a statement can switch to the login role or to any
    role it belongs to with set_config('role', ...), even in a read-only
    transaction, and rolsuper is not inherited.
    """
    conn = await asyncpg.connect(query_database_url)
    try:
        row = await conn.fetchrow(_ROLE_CHECK_QUERY, list(_SERVER_ACCESS_ROLES))
    except BaseException:
        # Drop the socket: a graceful close first cancels the stalled query
        # over a new connection, which has no deadline
        conn.terminate()
        raise
    await conn.close()
    if row["superuser"]:
        return "the READONLY_DATABASE_URL role is a superuser or a member of a superuser role"
    if row["members"]:
        return "the READONLY_DATABASE_URL role is a member of " + ", ".join(row["members"])
    return None


async def secure_sql_tools(mcp: FastMCP, query_database_url: str) -> None:
    """Remove the SQL tools unless the read-only role is safe to run SQL as.

    Refuses a superuser, a member of a server-access role, and a role that
    cannot be checked (fail closed), including one whose check does not
    finish within _CHECK_TIMEOUT seconds. Never raises; the URL is never
    logged.
    """
    if not query_database_url:
        return
    try:
        reason = await asyncio.wait_for(
            _unsafe_role_reason(query_database_url), _CHECK_TIMEOUT
        )
    except Exception as e:
        reason = f"could not check the READONLY_DATABASE_URL role: {type(e).__name__}"
    if reason:
        _disable_sql_tools(mcp, reason)
        return
    _set_sql_tools_status("enabled")
    logger.info("sql_tools_enabled")


def register_database_tools(mcp: FastMCP, database_url: str, query_database_url: str) -> None:
    """Register database access tools on the MCP server.

    `query_database_url` serves the free-form SQL tools (strata_db_query,
    strata_db_explain); the fixed introspection queries and the audit log use
    `database_url`. When `query_database_url` is empty the SQL tools are not
    registered at all, so they never run as the owner role. secure_sql_tools
    checks the role behind it at startup.
    """

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_db_schema(
        table_name: str | None = None,
    ) -> dict:
        """Explore the StrataKB database schema.

        Without arguments, lists all tables across all schemas.
        With a table_name, shows column details for that table.

        Args:
            table_name: Optional fully qualified table name (e.g., "strata_core.knowledge_layers").
                        If omitted, lists all tables.
        """
        from stratakb.core.database import get_pool

        pool = await get_pool(database_url)

        try:
            if table_name:
                # Parse schema.table
                parts = table_name.split(".")
                if len(parts) == 2:
                    schema, table = parts
                else:
                    schema, table = "public", parts[0]

                rows = await pool.fetch(
                    """
                    SELECT column_name, data_type, is_nullable, column_default,
                           character_maximum_length, numeric_precision
                    FROM information_schema.columns
                    WHERE table_schema = $1 AND table_name = $2
                    ORDER BY ordinal_position
                    """,
                    schema,
                    table,
                )

                columns = []
                for row in rows:
                    col = {
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                    }
                    if row["column_default"]:
                        col["default"] = row["column_default"]
                    if row["character_maximum_length"]:
                        col["max_length"] = row["character_maximum_length"]
                    columns.append(col)

                if not columns:
                    return {"status": "error", "error": f"Table '{table_name}' not found."}

                # Get constraints
                constraints = await pool.fetch(
                    """
                    SELECT tc.constraint_name, tc.constraint_type,
                           kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = $1 AND tc.table_name = $2
                    """,
                    schema,
                    table,
                )

                constraint_list = [
                    {
                        "name": row["constraint_name"],
                        "type": row["constraint_type"],
                        "column": row["column_name"],
                    }
                    for row in constraints
                ]

                return {
                    "status": "ok",
                    "table": table_name,
                    "columns": columns,
                    "constraints": constraint_list,
                }
            else:
                # List all tables
                rows = await pool.fetch(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_schema IN ('strata_core', 'strata_vectors', 'strata_learning')
                    ORDER BY table_schema, table_name
                    """
                )

                tables = [
                    {"schema": row["table_schema"], "table": row["table_name"]}
                    for row in rows
                ]

                return {
                    "status": "ok",
                    "tables": tables,
                    "count": len(tables),
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_db_vector_stats() -> dict:
        """Get vector store statistics.

        Returns layer names, chunk counts, embedding dimensions, and index info
        for the strata_vectors.chunks table.
        """
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import _vector_column_size

        pool = await get_pool(database_url)

        try:
            # Per-layer stats
            layer_stats = await pool.fetch(
                """
                SELECT layer_name,
                       count(*) as chunk_count,
                       count(DISTINCT source_file) as source_files,
                       count(DISTINCT doc_type) as doc_types,
                       min(created_at) as earliest,
                       max(created_at) as latest
                FROM strata_vectors.chunks
                GROUP BY layer_name
                ORDER BY layer_name
                """
            )

            layers = []
            for row in layer_stats:
                layers.append({
                    "layer_name": row["layer_name"],
                    "chunk_count": row["chunk_count"],
                    "source_files": row["source_files"],
                    "doc_types": row["doc_types"],
                    "earliest": row["earliest"].isoformat() if row["earliest"] else None,
                    "latest": row["latest"].isoformat() if row["latest"] else None,
                })

            # Total chunks
            total = await pool.fetchval("SELECT count(*) FROM strata_vectors.chunks")

            # Embedding dimensions
            dim = await _vector_column_size(pool)

            # Indexes
            indexes = await pool.fetch(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'strata_vectors' AND tablename = 'chunks'
                """
            )
            idx_list = [{"name": r["indexname"], "definition": r["indexdef"]} for r in indexes]

            return {
                "status": "ok",
                "total_chunks": total,
                "embedding_dimensions": dim,
                "layers": layers,
                "indexes": idx_list,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_db_sample_chunks(
        layer_name: str,
        limit: int = 5,
    ) -> dict:
        """Return sample chunks from a knowledge layer for inspection.

        Args:
            layer_name: Name of the knowledge layer (e.g., "iso20022", "runbooks").
            limit: Number of sample chunks to return (1-20, default 5).
        """
        from stratakb.core.database import get_pool

        limit = max(1, min(limit, 20))
        pool = await get_pool(database_url)

        try:
            rows = await pool.fetch(
                """
                SELECT id, layer_name, content, metadata, source_file,
                       source_section, doc_type, created_at
                FROM strata_vectors.chunks
                WHERE layer_name = $1
                ORDER BY random()
                LIMIT $2
                """,
                layer_name,
                limit,
            )

            chunks = []
            for row in rows:
                import json

                meta = row["metadata"]
                if isinstance(meta, str):
                    meta = json.loads(meta)

                chunks.append({
                    "id": str(row["id"]),
                    "content": row["content"][:500] + ("..." if len(row["content"]) > 500 else ""),
                    "content_length": len(row["content"]),
                    "metadata": meta if isinstance(meta, dict) else {},
                    "source_file": row["source_file"] or "",
                    "source_section": row["source_section"] or "",
                    "doc_type": row["doc_type"] or "",
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                })

            return {
                "status": "ok",
                "layer": layer_name,
                "samples": chunks,
                "count": len(chunks),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    if query_database_url:
        _register_sql_tools(mcp, database_url, query_database_url)
    else:
        reason = "READONLY_DATABASE_URL is not set"
        _set_sql_tools_status("disabled", reason)
        logger.info("sql_tools_disabled", reason=reason)


def _register_sql_tools(mcp: FastMCP, database_url: str, query_database_url: str) -> None:
    """Register the free-form SQL tools, which run as the role behind `query_database_url`.

    `database_url` serves only the audit log.
    """
    _set_sql_tools_status("enabled")

    @mcp.tool()
    @require_role("operator")
    @log_tool_call
    async def strata_db_query(
        sql: str,
        params: list[str] | None = None,
    ) -> dict:
        """Execute a read-only SQL query against the StrataKB database.

        Runs SELECT queries against strata_core.knowledge_layers,
        strata_core.ingestion_runs, strata_core.ingested_files,
        strata_vectors.chunks and strata_vectors.documents (each document's
        extracted text; the audit log, users and learning tables are not
        readable), in a read-only transaction that is rolled back afterwards.
        Write operations (INSERT, UPDATE, DELETE, DROP, etc.) are blocked.
        Returns at most 1000 rows. Use fully qualified table names
        (e.g., strata_core.knowledge_layers).

        Args:
            sql: SQL query to execute. Must be read-only (SELECT, EXPLAIN, SHOW, WITH).
                 Use $1, $2, etc. for parameterized values.
                 Examples:
                   "SELECT * FROM strata_core.knowledge_layers"
                   "SELECT count(*) FROM strata_vectors.chunks WHERE layer_name = $1"
            params: Optional list of parameter values for $1, $2, etc. placeholders.
        """
        # Validate read-only
        error = _validate_read_only(sql)
        if error:
            return {"status": "error", "error": error}

        try:
            from stratakb.core.database import get_pool

            pool = await get_pool(query_database_url)

            # One row past the cap tells whether the result was truncated
            rows = await _fetch_read_only(pool, sql, params or [], _MAX_ROWS + 1)
            results = [dict(row) for row in rows[:_MAX_ROWS]]

            # Serialize non-JSON-safe types
            for row_dict in results:
                for key, value in row_dict.items():
                    if hasattr(value, "isoformat"):
                        row_dict[key] = value.isoformat()
                    elif isinstance(value, (bytes, bytearray)):
                        row_dict[key] = f"<binary {len(value)} bytes>"
                    elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        row_dict[key] = str(value)

            truncated = len(rows) > _MAX_ROWS

            # Audit log
            await write_audit_log(
                database_url,
                action="query",
                tool_name="strata_db_query",
                parameters={"sql": sql[:500]},
                result_summary=f"{len(results)} rows returned" + (" (truncated)" if truncated else ""),
            )

            return {
                "status": "ok",
                "rows": results,
                "row_count": len(results),
                "truncated": truncated,
            }
        except Exception as e:
            return {"status": "error", "error": _query_error(e)}

    @mcp.tool()
    @require_role("operator")
    @log_tool_call
    async def strata_db_explain(
        sql: str,
    ) -> dict:
        """Run EXPLAIN ANALYZE on a SQL query to show the execution plan.

        Useful for understanding query performance and optimizing searches.
        EXPLAIN ANALYZE executes the query, so it runs in a read-only
        transaction that is rolled back afterwards.

        Args:
            sql: The SQL query to analyze. Must be a SELECT query.
        """
        error = _validate_read_only(sql)
        if error:
            return {"status": "error", "error": error}

        from stratakb.core.database import get_pool

        pool = await get_pool(query_database_url)

        try:
            rows = await _fetch_read_only(pool, f"EXPLAIN ANALYZE {sql}", [], _MAX_ROWS)

            plan_lines = [row[0] for row in rows]

            return {
                "status": "ok",
                "plan": "\n".join(plan_lines),
            }
        except Exception as e:
            return {"status": "error", "error": _query_error(e)}
