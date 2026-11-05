"""Tests for the learning MCP tools: registration and audit entries."""

from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


def test_learning_tools_import():
    """Verify learning tools module imports without error."""
    from stratakb.learning.tools import register_learning_tools
    assert callable(register_learning_tools)


def test_mcp_server_registers_learning_tools():
    """Verify MCP server can import and register all tools including learning."""
    from stratakb.mcp_server import mcp
    assert mcp.name == "StrataKB"


# --- audit entries ------------------------------------------------------------

DB = "postgresql://unused"


def _tool(name: str, database_url: str = DB):
    from mcp.server.fastmcp import FastMCP

    from stratakb.learning.tools import register_learning_tools

    server = FastMCP("test")
    register_learning_tools(server, database_url)
    return server._tool_manager.get_tool(name).fn


@contextmanager
def _admin():
    with (
        patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
        patch("stratakb.learning.tools.get_current_user", return_value=("tester", "admin")),
    ):
        yield


@contextmanager
def _mocked(stored: bool = True, reset_ok: bool = True):
    """A fake LearningStore and a recorded write_audit_log."""
    store = MagicMock()
    store.store_feedback = AsyncMock(return_value=stored)
    store.reset = AsyncMock(side_effect=lambda layer=None: (
        None if not reset_ok
        else {"reset": "layer", "layer": layer} if layer is not None
        else {"reset": "all"}
    ))
    audit = AsyncMock()
    with (
        _admin(),
        patch("stratakb.learning.store.LearningStore", return_value=store),
        patch("stratakb.learning.tools.write_audit_log", audit),
    ):
        yield store, audit


async def test_feedback_writes_one_audit_entry():
    with _mocked() as (_, audit):
        result = await _tool("strata_feedback")(interaction_id="abc", helpful=True)
    assert result["status"] == "ok"
    audit.assert_awaited_once()
    kwargs = audit.await_args.kwargs
    assert kwargs["user_id"] == "tester"
    assert kwargs["action"] == "feedback"
    assert kwargs["tool_name"] == "strata_feedback"
    assert kwargs["parameters"] == {"interaction_id": "abc", "helpful": True}
    assert kwargs["result_summary"] == "Rated helpful"


async def test_feedback_audit_leaves_out_free_text():
    with _mocked() as (_, audit):
        await _tool("strata_feedback")(
            interaction_id="abc",
            helpful=False,
            correct_layer="x-layer",
            better_query="sensitive text",
            chunk_feedback={"c1": "wrong", "c2": "useful"},
        )
    audit.assert_awaited_once()
    call = str(audit.await_args)
    assert "x-layer" not in call
    assert "sensitive text" not in call
    assert "c1" not in call
    summary = audit.await_args.kwargs["result_summary"]
    assert summary == "Rated not helpful, with correct_layer, better_query, 2 chunk ratings"
    assert audit.await_args.kwargs["parameters"] == {"interaction_id": "abc", "helpful": False}


async def test_feedback_unknown_interaction_not_audited():
    with _mocked(stored=False) as (_, audit):
        result = await _tool("strata_feedback")(interaction_id="nope", helpful=True)
    assert result["status"] == "error"
    audit.assert_not_awaited()


async def test_reset_layer_audited():
    with _mocked() as (store, audit):
        result = await _tool("strata_reset_learning")(layer="L", confirm=True)
    assert result["status"] == "ok"
    store.reset.assert_awaited_once_with(layer="L")
    audit.assert_awaited_once()
    kwargs = audit.await_args.kwargs
    assert kwargs["action"] == "reset_learning"
    assert kwargs["tool_name"] == "strata_reset_learning"
    assert kwargs["parameters"] == {"scope": "layer", "layer": "L"}
    assert kwargs["result_summary"] == "Reset learned data for layer 'L'"


async def test_reset_all_audited():
    with _mocked() as (_, audit):
        await _tool("strata_reset_learning")(confirm=True)
    audit.assert_awaited_once()
    assert audit.await_args.kwargs["parameters"] == {"scope": "all"}
    assert "interaction history" in audit.await_args.kwargs["result_summary"]


async def test_reset_without_confirm_not_audited():
    with _mocked() as (store, audit):
        result = await _tool("strata_reset_learning")(layer="L")
    assert result["status"] == "error"
    store.reset.assert_not_awaited()
    audit.assert_not_awaited()


async def test_feedback_audit_names_only_stored_extras():
    """An extra signal that fails to store is left out of the audit summary."""
    async def store_feedback(**kwargs):
        # The rating and one chunk rating are stored; better_query is not
        name = kwargs["signal_name"]
        if name == "better_query":
            return False
        if name.startswith("chunk_"):
            return kwargs["metadata"]["chunk_id"] == "c1"
        return True

    with _mocked() as (store, audit):
        store.store_feedback = AsyncMock(side_effect=store_feedback)
        result = await _tool("strata_feedback")(
            interaction_id="abc",
            helpful=False,
            correct_layer="x-layer",
            better_query="sensitive text",
            chunk_feedback={"c1": "wrong", "c2": "useful"},
        )
    assert result["status"] == "ok"
    audit.assert_awaited_once()
    summary = audit.await_args.kwargs["result_summary"]
    assert summary == "Rated not helpful, with correct_layer, 1 chunk ratings"


async def test_feedback_audit_no_extras_when_none_stored():
    async def store_feedback(**kwargs):
        return kwargs["signal_name"] in ("helpful", "wrong")

    with _mocked() as (store, audit):
        store.store_feedback = AsyncMock(side_effect=store_feedback)
        await _tool("strata_feedback")(
            interaction_id="abc",
            helpful=True,
            correct_layer="x-layer",
            chunk_feedback={"c1": "useful"},
        )
    assert audit.await_args.kwargs["result_summary"] == "Rated helpful"


async def test_reset_failed_returns_error_not_audited():
    with _mocked(reset_ok=False) as (store, audit):
        result = await _tool("strata_reset_learning")(layer="L", confirm=True)
    assert result["status"] == "error"
    assert "nothing was deleted" in result["error"]
    store.reset.assert_awaited_once_with(layer="L")
    audit.assert_not_awaited()


async def test_reset_all_failed_returns_error_not_audited():
    with _mocked(reset_ok=False) as (_, audit):
        result = await _tool("strata_reset_learning")(confirm=True)
    assert result["status"] == "error"
    audit.assert_not_awaited()


async def test_reset_empty_layer_refused():
    for layer in ("", "   "):
        with _mocked() as (store, audit):
            result = await _tool("strata_reset_learning")(layer=layer, confirm=True)
        assert result["status"] == "error"
        assert "empty" in result["error"]
        store.reset.assert_not_awaited()
        audit.assert_not_awaited()


class _FailingPool:
    """Wraps a real pool; any statement containing `fail_on` raises."""

    def __init__(self, pool, fail_on: str):
        self._pool = pool
        self._fail_on = fail_on

    @asynccontextmanager
    async def acquire(self):
        async with self._pool.acquire() as conn:
            yield _FailingConn(conn, self._fail_on)


class _FailingConn:
    def __init__(self, conn, fail_on: str):
        self._conn = conn
        self._fail_on = fail_on

    def transaction(self):
        return self._conn.transaction()

    async def execute(self, sql, *args):
        if self._fail_on in sql:
            raise RuntimeError("simulated database failure")
        return await self._conn.execute(sql, *args)


async def test_store_reset_failure_deletes_nothing():
    """A layer reset whose last delete fails rolls back the earlier ones."""
    from stratakb.config import get_settings
    from stratakb.core.database import get_pool
    from stratakb.learning.store import LearningStore

    db_url = get_settings().database_url
    layer = f"probe{uuid4().hex[:8]}"
    pool = await get_pool(db_url)
    await pool.execute(
        "INSERT INTO strata_learning.query_synonyms (term, synonym, layer, source) "
        "VALUES ('t', 's', $1, 'manual')",
        layer,
    )
    count = "SELECT count(*) FROM strata_learning.query_synonyms WHERE layer = $1"
    store = LearningStore(db_url)
    try:
        failing = _FailingPool(pool, "layer_affinity_scores")
        with patch("stratakb.learning.store.get_pool", AsyncMock(return_value=failing)):
            assert await store.reset(layer=layer) is None
        assert await pool.fetchval(count, layer) == 1

        assert await store.reset(layer=layer) == {"reset": "layer", "layer": layer}
        assert await pool.fetchval(count, layer) == 0
    finally:
        await pool.execute(
            "DELETE FROM strata_learning.query_synonyms WHERE layer = $1", layer
        )


def _recording_pool():
    """A fake pool whose connection records each executed statement."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.transaction = MagicMock(return_value=AsyncMock())
    pool = MagicMock()

    @asynccontextmanager
    async def acquire():
        yield conn

    pool.acquire = acquire
    return pool, conn


async def test_store_reset_all_is_one_truncate():
    """A global reset clears every learning table in one statement."""
    from stratakb.learning.store import LearningStore

    pool, conn = _recording_pool()
    with patch("stratakb.learning.store.get_pool", AsyncMock(return_value=pool)):
        assert await LearningStore(DB).reset() == {"reset": "all"}
    conn.execute.assert_awaited_once()
    sql = conn.execute.await_args.args[0]
    assert sql.startswith("TRUNCATE") and sql.endswith("CASCADE")
    for table in (
        "feedback_signals", "interaction_events", "query_sessions",
        "learned_rewrites", "query_synonyms", "layer_affinity_scores",
    ):
        assert f"strata_learning.{table}" in sql


async def test_store_reset_empty_layer_is_not_global():
    """An empty layer never reaches the global TRUNCATE path."""
    from stratakb.learning.store import LearningStore

    pool, conn = _recording_pool()
    with patch("stratakb.learning.store.get_pool", AsyncMock(return_value=pool)):
        result = await LearningStore(DB).reset(layer="")
    assert result == {"reset": "layer", "layer": ""}
    assert not any("TRUNCATE" in c.args[0] for c in conn.execute.await_args_list)


async def test_reset_audit_entry_in_db():
    """A real layer reset leaves one audit row; the unique layer name touches no other data."""
    from stratakb.config import get_settings
    from stratakb.core.database import get_pool

    db_url = get_settings().database_url
    layer = f"probe{uuid4().hex[:8]}"
    with _admin():
        result = await _tool("strata_reset_learning", db_url)(layer=layer, confirm=True)
    assert result == {"status": "ok", "reset": "layer", "layer": layer}

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        "SELECT * FROM strata_core.audit_log WHERE tool_name = 'strata_reset_learning' "
        "AND parameters->>'layer' = $1",
        layer,
    )
    assert len(rows) == 1
    assert rows[0]["user_id"] == "tester"
    assert rows[0]["action"] == "reset_learning"
