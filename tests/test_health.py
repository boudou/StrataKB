"""Test the health check tool."""

import subprocess
import sys
from unittest.mock import AsyncMock, call, patch

import pytest


def test_settings_load():
    """Settings should load from defaults without .env file."""
    from stratakb.config import get_settings

    settings = get_settings()
    assert settings.strata_user == "developer" or isinstance(settings.strata_user, str)
    assert settings.strata_role in ("viewer", "operator", "admin")


def test_mcp_server_imports():
    """MCP server module should import without errors."""
    from stratakb.mcp_server import mcp

    assert mcp is not None
    assert mcp.name == "StrataKB"


def test_import_reads_no_settings():
    """Importing the server module reads no settings; build_server and main do."""
    code = (
        "from unittest.mock import patch\n"
        "import stratakb.config\n"
        "with patch.object(stratakb.config, 'get_settings',"
        " side_effect=AssertionError('settings read at import')):\n"
        "    import stratakb.mcp_server\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_tracker_uses_the_live_settings():
    """The tracker binds get_settings at import and holds the real one, not a test's patch."""
    import stratakb.config
    import stratakb.learning.tracker

    assert stratakb.learning.tracker.get_settings is stratakb.config.get_settings


async def test_two_servers_use_their_own_database_url():
    """Two servers built in one process each check their own database."""
    from stratakb import mcp_server
    from stratakb.config import get_settings

    url_a = "postgresql://a:a@127.0.0.1:1/db_a"
    url_b = "postgresql://b:b@127.0.0.1:1/db_b"
    server_a = mcp_server.build_server(get_settings().model_copy(update={"database_url": url_a}))
    server_b = mcp_server.build_server(get_settings().model_copy(update={"database_url": url_b}))

    check = AsyncMock(return_value={"status": "ok"})
    with patch("stratakb.mcp_server.check_connection", check):
        await server_a._tool_manager.get_tool("strata_admin_health").fn()
        await server_b._tool_manager.get_tool("strata_admin_health").fn()

    assert check.await_args_list == [call(url_a), call(url_b)]
    admin_tools = {
        "strata_admin_health",
        "strata_admin_audit_log",
        "strata_whoami",
        "strata_admin_list_users",
        "strata_admin_update_role",
    }
    for server in (server_a, server_b):
        names = {tool.name for tool in await server.list_tools()}
        assert admin_tools <= names
