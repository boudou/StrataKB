"""Tests for the learning MCP tools: registration and audit entries."""

from contextlib import contextmanager
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
def _mocked(stored: bool = True):
    """A fake LearningStore and a recorded write_audit_log."""
    store = MagicMock()
    store.store_feedback = AsyncMock(return_value=stored)
    store.reset = AsyncMock(side_effect=lambda layer=None: (
        {"reset": "layer", "layer": layer} if layer else {"reset": "all"}
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
