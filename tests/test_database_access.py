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


_FIXED_TOOLS = {"strata_db_schema", "strata_db_vector_stats", "strata_db_sample_chunks"}
_SQL_TOOL_NAMES = {"strata_db_query", "strata_db_explain"}
_FAKE_URL = "postgresql://ro:secret-pw@db.example:5432/strata"


def _server(query_database_url: str = _FAKE_URL):
    from mcp.server.fastmcp import FastMCP

    from stratakb.database_access.tools import register_database_tools

    server = FastMCP("test")
    register_database_tools(server, "postgresql://owner@db.example/strata", query_database_url)
    return server


def _tool_names(server) -> set[str]:
    return {t.name for t in server._tool_manager.list_tools()}


_SAFE_ROW = {"superuser": False, "members": None}


async def _hang(*args, **kwargs):
    """Never return, like a server that stopped answering."""
    import asyncio

    await asyncio.Event().wait()


def _fake_connect(row=None, error=None):
    """Patch asyncpg.connect in the tools module; return (patcher, conn).

    `row` is the role check's result, an exception it raises, or a function
    standing in for it; `error` is what connect raises, or a function.
    """
    from unittest.mock import patch

    conn = MagicMock()
    if isinstance(row, BaseException) or callable(row):
        conn.fetchrow = AsyncMock(side_effect=row)
    else:
        conn.fetchrow = AsyncMock(return_value=row or _SAFE_ROW)
    conn.close = AsyncMock()
    connect = AsyncMock(side_effect=error) if error else AsyncMock(return_value=conn)
    return patch("stratakb.database_access.tools.asyncpg.connect", connect), conn


@pytest.fixture(autouse=True)
def _keep_sql_tools_status():
    """Restore the module's SQL tools status after each test."""
    from stratakb.database_access import tools

    saved = dict(tools._sql_tools)
    yield
    tools._sql_tools.clear()
    tools._sql_tools.update(saved)


class TestDBToolRegistration:
    def test_db_tools_registered(self):
        assert _tool_names(_server()) == _FIXED_TOOLS | _SQL_TOOL_NAMES


class TestSqlToolsGate:
    def test_sql_tools_not_registered_without_url(self):
        from structlog.testing import capture_logs

        from stratakb.database_access.tools import sql_tools_status

        with capture_logs() as logs:
            server = _server("")

        assert _tool_names(server) == _FIXED_TOOLS
        disabled = [e for e in logs if e["event"] == "sql_tools_disabled"]
        assert disabled and disabled[0]["reason"] == "READONLY_DATABASE_URL is not set"
        assert sql_tools_status() == {
            "status": "disabled",
            "reason": "READONLY_DATABASE_URL is not set",
        }

    def test_sql_tools_registered_with_url(self):
        from stratakb.database_access.tools import sql_tools_status

        assert _tool_names(_server()) >= _SQL_TOOL_NAMES
        assert sql_tools_status() == {"status": "enabled"}

    async def _secure(self, server, row=None, error=None):
        from structlog.testing import capture_logs

        from stratakb.database_access.tools import secure_sql_tools

        patcher, conn = _fake_connect(row, error)
        with patcher, capture_logs() as logs:
            await secure_sql_tools(server, _FAKE_URL)
        return logs, conn

    def _assert_disabled(self, server, logs, reason_part):
        from stratakb.database_access.tools import sql_tools_status

        assert _tool_names(server) == _FIXED_TOOLS
        disabled = [e for e in logs if e["event"] == "sql_tools_disabled"]
        assert len(disabled) == 1
        assert reason_part in disabled[0]["reason"]
        status = sql_tools_status()
        assert status["status"] == "disabled"
        assert reason_part in status["reason"]
        # The reason never carries the URL or its password
        assert "secret-pw" not in status["reason"]
        assert "db.example" not in status["reason"]

    @pytest.mark.asyncio
    async def test_superuser_role_removes_sql_tools(self):
        server = _server()
        logs, conn = await self._secure(server, {"superuser": True, "members": None})
        self._assert_disabled(server, logs, "is a superuser or a member of a superuser role")
        conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_superuser_check_covers_role_membership(self):
        """rolsuper is not inherited: the check must look at every role reachable."""
        server = _server()
        from stratakb.database_access.tools import _SERVER_ACCESS_ROLES

        _, conn = await self._secure(server)
        # One query answers both questions
        assert len(conn.fetchrow.await_args_list) == 1
        conn.fetchval.assert_not_called()
        args = conn.fetchrow.await_args_list[0].args
        assert "pg_has_role(session_user, oid, 'MEMBER')" in args[0]
        assert "pg_has_role(current_user, oid, 'MEMBER')" in args[0]
        assert args[1] == list(_SERVER_ACCESS_ROLES)

    @pytest.mark.asyncio
    async def test_server_access_member_removes_sql_tools(self):
        server = _server()
        logs, conn = await self._secure(
            server, {"superuser": False, "members": ["pg_read_server_files"]}
        )
        self._assert_disabled(server, logs, "is a member of pg_read_server_files")
        conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unreachable_role_removes_sql_tools(self):
        server = _server()
        logs, _ = await self._secure(server, error=OSError("connection refused"))
        self._assert_disabled(
            server, logs, "could not check the READONLY_DATABASE_URL role: OSError"
        )

    @pytest.mark.asyncio
    async def test_failed_check_query_terminates_connection(self):
        server = _server()
        logs, conn = await self._secure(server, RuntimeError("boom"))
        self._assert_disabled(server, logs, "could not check")
        conn.terminate.assert_called_once()
        conn.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stalled_query_fails_closed(self):
        """A query that never returns ends the check within the timeout."""
        import time
        from unittest.mock import patch

        server = _server()
        started = time.monotonic()
        with patch("stratakb.database_access.tools._CHECK_TIMEOUT", 0.05):
            logs, conn = await self._secure(server, _hang)
        assert time.monotonic() - started < 1
        self._assert_disabled(
            server, logs, "could not check the READONLY_DATABASE_URL role: TimeoutError"
        )
        # Dropped, not closed: a graceful close would wait on the stalled server
        conn.terminate.assert_called_once()
        conn.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stalled_connect_fails_closed(self):
        """A connection that never opens ends the check within the timeout."""
        import time
        from unittest.mock import patch

        server = _server()
        started = time.monotonic()
        with patch("stratakb.database_access.tools._CHECK_TIMEOUT", 0.05):
            logs, _ = await self._secure(server, error=_hang)
        assert time.monotonic() - started < 1
        self._assert_disabled(
            server, logs, "could not check the READONLY_DATABASE_URL role: TimeoutError"
        )

    @pytest.mark.asyncio
    async def test_safe_role_keeps_sql_tools(self):
        from stratakb.database_access.tools import sql_tools_status

        server = _server()
        logs, conn = await self._secure(server)
        assert _tool_names(server) == _FIXED_TOOLS | _SQL_TOOL_NAMES
        assert any(e["event"] == "sql_tools_enabled" for e in logs)
        assert not any(e["event"] == "sql_tools_disabled" for e in logs)
        assert sql_tools_status() == {"status": "enabled"}
        conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_secure_twice_is_safe(self):
        server = _server()
        await self._secure(server, {"superuser": True, "members": None})
        logs, _ = await self._secure(server, {"superuser": True, "members": None})
        self._assert_disabled(server, logs, "is a superuser")

    @pytest.mark.asyncio
    async def test_secure_without_url_does_nothing(self):
        from unittest.mock import patch

        from stratakb.database_access.tools import secure_sql_tools

        server = _server("")
        connect = AsyncMock()
        with patch("stratakb.database_access.tools.asyncpg.connect", connect):
            await secure_sql_tools(server, "")
        connect.assert_not_called()
        assert _tool_names(server) == _FIXED_TOOLS

    @pytest.mark.asyncio
    async def test_health_reports_sql_tools(self):
        from unittest.mock import patch

        from stratakb import mcp_server
        from stratakb.config import get_settings
        from stratakb.database_access import tools

        # Building a server resets the SQL tools status, so build first
        server = mcp_server.build_server(get_settings())
        tools._sql_tools.clear()
        tools._sql_tools.update(status="disabled", reason="READONLY_DATABASE_URL is not set")
        with patch(
            "stratakb.mcp_server.check_connection",
            AsyncMock(return_value={"status": "ok"}),
        ):
            result = await server._tool_manager.get_tool("strata_admin_health").fn()

        assert result["sql_tools"] == {
            "status": "disabled",
            "reason": "READONLY_DATABASE_URL is not set",
        }


class TestSqlToolsGateDatabase:
    """The role check against the real database."""

    @pytest.mark.asyncio
    async def test_real_superuser_url_refused(self, db_url):
        import asyncpg

        from stratakb.database_access.tools import secure_sql_tools

        try:
            conn = await asyncpg.connect(db_url, timeout=5)
        except Exception:
            pytest.skip("database not reachable")
        try:
            superuser = await conn.fetchval(
                "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
            )
        finally:
            await conn.close()
        if not superuser:
            pytest.skip("DATABASE_URL role is not a superuser")

        server = _server(db_url)
        await secure_sql_tools(server, db_url)
        assert _tool_names(server) == _FIXED_TOOLS

    @pytest.mark.asyncio
    async def test_member_of_superuser_role_detected(self, db_url):
        """A role granted a superuser role can reach it with set_config('role', ...)."""
        import asyncpg

        from stratakb.database_access.tools import _ROLE_CHECK_QUERY, _SERVER_ACCESS_ROLES

        try:
            conn = await asyncpg.connect(db_url, timeout=5)
        except Exception:
            pytest.skip("database not reachable")
        try:
            superuser = await conn.fetchval(
                "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
            )
            if not superuser:
                pytest.skip("DATABASE_URL role is not a superuser")
            owner = await conn.fetchval("SELECT current_user")
            tr = conn.transaction()
            await tr.start()
            try:
                # Roles and grants vanish with the rollback
                await conn.execute("CREATE ROLE strata_probe_member NOLOGIN NOSUPERUSER")
                await conn.execute("CREATE ROLE strata_probe_plain NOLOGIN NOSUPERUSER")
                await conn.execute(f'GRANT "{owner}" TO strata_probe_member')

                await conn.execute("SET LOCAL SESSION AUTHORIZATION strata_probe_member")
                row = await conn.fetchrow(_ROLE_CHECK_QUERY, list(_SERVER_ACCESS_ROLES))
                assert row["superuser"] is True

                await conn.execute("SET LOCAL SESSION AUTHORIZATION strata_probe_plain")
                row = await conn.fetchrow(_ROLE_CHECK_QUERY, list(_SERVER_ACCESS_ROLES))
                assert row["superuser"] is False
                assert row["members"] is None
            finally:
                await tr.rollback()
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_server_access_member_detected(self, db_url):
        """A role granted a server-access role is named by the same query."""
        import asyncpg

        from stratakb.database_access.tools import _ROLE_CHECK_QUERY, _SERVER_ACCESS_ROLES

        try:
            conn = await asyncpg.connect(db_url, timeout=5)
        except Exception:
            pytest.skip("database not reachable")
        try:
            superuser = await conn.fetchval(
                "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
            )
            if not superuser:
                pytest.skip("DATABASE_URL role is not a superuser")
            tr = conn.transaction()
            await tr.start()
            try:
                # Roles and grants vanish with the rollback
                await conn.execute("CREATE ROLE strata_probe_reader NOLOGIN NOSUPERUSER")
                await conn.execute("GRANT pg_read_server_files TO strata_probe_reader")

                await conn.execute("SET LOCAL SESSION AUTHORIZATION strata_probe_reader")
                row = await conn.fetchrow(_ROLE_CHECK_QUERY, list(_SERVER_ACCESS_ROLES))
                assert row["superuser"] is False
                assert row["members"] == ["pg_read_server_files"]
            finally:
                await tr.rollback()
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_real_readonly_role_accepted(self):
        from stratakb.config import get_settings
        from stratakb.database_access.tools import secure_sql_tools, sql_tools_status

        url = get_settings().readonly_database_url
        if not url:
            pytest.skip("READONLY_DATABASE_URL is not set")
        server = _server(url)
        await secure_sql_tools(server, url)
        assert _tool_names(server) == _FIXED_TOOLS | _SQL_TOOL_NAMES
        assert sql_tools_status() == {"status": "enabled"}
