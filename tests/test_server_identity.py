"""A running server keeps the user and role it started with; other processes do not pin."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from stratakb.config import Settings
from stratakb.core import auth
from stratakb.core.auth import get_current_user
from tests.helpers import call_tool


def _settings(**overrides) -> Settings:
    values = {
        "readonly_database_url": "",
        "strata_local_only": False,
        "strata_user": "alice",
        "strata_role": "admin",
    }
    return Settings(_env_file=None, **{**values, **overrides})


@pytest.fixture
def main_server(monkeypatch):
    """stratakb.mcp_server with FastMCP.run patched, so main() returns at once."""
    from mcp.server.fastmcp import FastMCP

    from stratakb import mcp_server

    monkeypatch.delenv("STRATAKB_CONFIG", raising=False)
    monkeypatch.setattr(FastMCP, "run", MagicMock())
    return mcp_server


def _edit_identity(monkeypatch):
    monkeypatch.setenv("STRATA_USER", "mallory")
    monkeypatch.setenv("STRATA_ROLE", "viewer")


def test_outside_a_server_identity_follows_the_environment(monkeypatch):
    monkeypatch.delenv("STRATAKB_CONFIG", raising=False)
    monkeypatch.setenv("STRATA_USER", "bob")
    monkeypatch.setenv("STRATA_ROLE", "operator")
    assert get_current_user() == ("bob", "operator")

    _edit_identity(monkeypatch)
    assert get_current_user() == ("mallory", "viewer")


def test_main_pins_the_identity_it_started_with(main_server, monkeypatch):
    main_server.main(_settings())
    _edit_identity(monkeypatch)

    with patch("stratakb.config.get_settings") as get_settings:
        assert get_current_user() == ("alice", "admin")
    get_settings.assert_not_called()


async def test_health_and_role_check_agree_after_an_edit(main_server, monkeypatch):
    settings = _settings()
    main_server.main(settings)
    _edit_identity(monkeypatch)

    server = main_server.build_server(settings)
    with (
        patch(
            "stratakb.mcp_server.check_connection",
            AsyncMock(return_value={"status": "ok"}),
        ),
        patch("stratakb.mcp_server.query_audit_log", AsyncMock(return_value=[])),
    ):
        health = await call_tool(server, "strata_admin_health")
        # Health only needs a viewer; this one needs the admin the server started as
        audit = await call_tool(server, "strata_admin_audit_log")

    assert health["user"] == {"user_id": "alice", "role": "admin"}
    assert audit["status"] == "ok"


async def test_a_started_viewer_stays_refused_after_an_edit(main_server, monkeypatch):
    settings = _settings(strata_user="vera", strata_role="viewer")
    main_server.main(settings)
    monkeypatch.setenv("STRATA_USER", "mallory")
    monkeypatch.setenv("STRATA_ROLE", "admin")

    with pytest.raises(auth.PermissionDeniedError, match="user 'vera' has role 'viewer'"):
        await call_tool(main_server.build_server(settings), "strata_admin_audit_log")


async def test_audit_uses_the_pinned_user(main_server, monkeypatch, db_url):
    from stratakb.core.audit import query_audit_log, write_audit_log

    main_server.main(_settings(database_url=db_url))
    _edit_identity(monkeypatch)
    tool_name = f"test_tool_{uuid4().hex[:8]}"

    await write_audit_log(db_url, action="test_action", tool_name=tool_name)

    entries = await query_audit_log(db_url, tool_name=tool_name)
    assert len(entries) == 1
    assert entries[0]["user_id"] == "alice"


def test_refused_start_pins_nothing(main_server):
    settings = _settings(
        strata_local_only=True,
        anthropic_api_key="sk-ant-test-123",
        embedding_backend="local",
        embedding_base_url="",
        query_rewrite_provider="anthropic",
        query_rewrite_base_url="",
    )

    with pytest.raises(SystemExit):
        main_server.main(settings)

    assert auth._server_identity is None
