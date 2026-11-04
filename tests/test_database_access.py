"""Tests for database access tools.

Tests cover:
- SQL validation (read-only enforcement)
- strata_db_query execution
- strata_db_schema introspection
- strata_db_vector_stats
- strata_db_sample_chunks
- strata_db_explain
- MCP tool registration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ==========================================
# SQL Validation tests
# ==========================================


class TestSQLValidation:
    def test_select_allowed(self):
        from stratakb.database_access.tools import _validate_read_only

        assert _validate_read_only("SELECT * FROM strata_core.users") is None

    def test_with_cte_allowed(self):
        from stratakb.database_access.tools import _validate_read_only

        assert _validate_read_only("WITH cte AS (SELECT 1) SELECT * FROM cte") is None

    def test_explain_allowed(self):
        from stratakb.database_access.tools import _validate_read_only

        assert _validate_read_only("EXPLAIN SELECT * FROM strata_core.users") is None

    def test_drop_blocked(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("DROP TABLE strata_core.users")
        assert result is not None
        assert "DROP" in result

    def test_delete_blocked(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("DELETE FROM strata_core.users")
        assert result is not None
        assert "DELETE" in result

    def test_insert_blocked(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("INSERT INTO strata_core.users VALUES (1)")
        assert result is not None
        assert "INSERT" in result

    def test_update_blocked(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("UPDATE strata_core.users SET role = 'admin'")
        assert result is not None
        assert "UPDATE" in result

    def test_truncate_blocked(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("TRUNCATE strata_core.users")
        assert result is not None
        assert "TRUNCATE" in result

    def test_alter_blocked(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("ALTER TABLE strata_core.users ADD COLUMN x TEXT")
        assert result is not None
        assert "ALTER" in result

    def test_create_blocked(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("CREATE TABLE test (id int)")
        assert result is not None
        assert "CREATE" in result

    def test_empty_query(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("")
        assert result is not None
        assert "empty" in result.lower()

    def test_whitespace_only(self):
        from stratakb.database_access.tools import _validate_read_only

        result = _validate_read_only("   ;  ")
        assert result is not None


# ==========================================
# Read-only execution (mocked connection)
# ==========================================


def _mock_pool(rows=None, fail=None):
    """A pool whose connection records the calls _fetch_read_only makes."""
    tr = MagicMock()
    tr.start = AsyncMock()
    tr.rollback = AsyncMock()

    cursor = MagicMock()
    cursor.fetch = AsyncMock(return_value=rows or [], side_effect=fail)

    conn = MagicMock()
    conn.transaction.return_value = tr
    conn.execute = AsyncMock()
    conn.cursor = AsyncMock(return_value=cursor)

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = ctx
    return pool, conn, tr, cursor


class TestFetchReadOnly:
    @pytest.mark.asyncio
    async def test_runs_in_read_only_transaction_with_local_timeout(self):
        from stratakb.database_access.tools import _fetch_read_only

        pool, conn, tr, cursor = _mock_pool(rows=[{"x": 1}])
        rows = await _fetch_read_only(pool, "SELECT $1::int AS x", ["1"], 1001)

        assert rows == [{"x": 1}]
        conn.transaction.assert_called_once_with(readonly=True)
        tr.start.assert_awaited_once()
        assert conn.execute.await_args.args[0].startswith("SET LOCAL statement_timeout")
        conn.cursor.assert_awaited_once_with("SELECT $1::int AS x", "1")
        cursor.fetch.assert_awaited_once_with(1001)

    @pytest.mark.asyncio
    async def test_always_rolls_back(self):
        from stratakb.database_access.tools import _fetch_read_only

        pool, _, tr, _ = _mock_pool()
        await _fetch_read_only(pool, "SELECT 1", [], 10)
        tr.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rolls_back_on_error(self):
        from stratakb.database_access.tools import _fetch_read_only

        pool, _, tr, _ = _mock_pool(fail=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            await _fetch_read_only(pool, "SELECT 1", [], 10)
        tr.rollback.assert_awaited_once()


class TestQueryError:
    def test_read_only_violation_explained(self):
        import asyncpg

        from stratakb.database_access.tools import _query_error

        msg = _query_error(asyncpg.ReadOnlySQLTransactionError("cannot execute SELECT INTO"))
        assert "read-only" in msg

    def test_timeout_suggests_fix(self):
        import asyncpg

        from stratakb.database_access.tools import _query_error

        msg = _query_error(asyncpg.QueryCanceledError("canceling statement"))
        assert "LIMIT" in msg

    def test_other_errors_passed_through(self):
        from stratakb.database_access.tools import _query_error

        assert _query_error(ValueError("bad column")) == "bad column"


# ==========================================
# Integration tests (require DB)
# ==========================================


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
def query_db_url():
    from stratakb.config import get_settings
    settings = get_settings()
    return settings.readonly_database_url or settings.database_url


class TestReadOnlyEnforcement:
    @pytest.mark.asyncio
    async def test_select_into_rejected(self, query_db_url):
        import asyncpg

        from stratakb.core.database import get_pool
        from stratakb.database_access.tools import _fetch_read_only

        pool = await get_pool(query_db_url)
        with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
            await _fetch_read_only(pool, "SELECT 1 AS x INTO strata_core.sql_tool_probe", [], 10)

    @pytest.mark.asyncio
    async def test_set_config_does_not_stick_to_connection(self, query_db_url):
        import asyncpg

        from stratakb.database_access.tools import _fetch_read_only

        # A single connection, so the second query sees what the first left behind
        pool = await asyncpg.create_pool(query_db_url, min_size=1, max_size=1)
        try:
            await _fetch_read_only(
                pool, "SELECT set_config('application_name', 'sql_tool_probe', false)", [], 10
            )
            async with pool.acquire() as conn:
                assert await conn.fetchval("SHOW application_name") != "sql_tool_probe"
        finally:
            await pool.close()

    @pytest.mark.asyncio
    async def test_row_cap_applied_by_cursor(self, query_db_url):
        from stratakb.core.database import get_pool
        from stratakb.database_access.tools import _fetch_read_only

        pool = await get_pool(query_db_url)
        rows = await _fetch_read_only(pool, "SELECT generate_series(1, 5000) AS n", [], 1001)
        assert len(rows) == 1001

    @pytest.mark.asyncio
    async def test_readonly_role_cannot_read_server_files(self):
        import asyncpg

        from stratakb.config import get_settings
        from stratakb.core.database import get_pool
        from stratakb.database_access.tools import _fetch_read_only

        url = get_settings().readonly_database_url
        if not url:
            pytest.skip("READONLY_DATABASE_URL is not set")
        pool = await get_pool(url)
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await _fetch_read_only(pool, "SELECT pg_read_file('postgresql.conf')", [], 10)


class TestDBQuery:
    @pytest.mark.asyncio
    async def test_select_query(self, db_url):
        from stratakb.core.database import get_pool

        pool = await get_pool(db_url)
        row = await pool.fetchval("SELECT 1")
        assert row == 1

    @pytest.mark.asyncio
    async def test_schema_list_tables(self, db_url):
        from stratakb.core.database import get_pool

        pool = await get_pool(db_url)
        rows = await pool.fetch(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ('strata_core', 'strata_vectors', 'strata_learning')
            """
        )
        table_names = [r["table_name"] for r in rows]
        assert "knowledge_layers" in table_names
        assert "chunks" in table_names
        assert "learned_rewrites" in table_names


# ==========================================
# MCP tool registration
# ==========================================


class TestDBToolRegistration:
    def test_db_tools_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_db_query" in tool_names
        assert "strata_db_schema" in tool_names
        assert "strata_db_vector_stats" in tool_names
        assert "strata_db_sample_chunks" in tool_names
        assert "strata_db_explain" in tool_names
