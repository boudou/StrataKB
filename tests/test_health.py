"""Test the health check tool."""

import subprocess
import sys
from unittest.mock import AsyncMock, call, patch

import pytest

from tests.conftest import call_tool


def test_settings_load():
    """Settings should load from defaults without .env file."""
    from stratakb.config import get_settings

    settings = get_settings()
    assert settings.strata_user == "developer" or isinstance(settings.strata_user, str)
    assert settings.strata_role in ("viewer", "operator", "admin")


def test_build_server_names_the_server(server):
    """A built server carries the StrataKB name."""
    assert server.name == "StrataKB"


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


def test_tracker_import_reads_no_settings():
    """Importing the tracker reads no settings and binds no get_settings."""
    code = (
        "from unittest.mock import patch\n"
        "import stratakb.config\n"
        "with patch.object(stratakb.config, 'get_settings',"
        " side_effect=AssertionError('settings read at import')):\n"
        "    import stratakb.learning.tracker\n"
        "assert not hasattr(stratakb.learning.tracker, 'get_settings')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_two_servers_track_with_their_own_settings():
    """Each built server hands its own settings to its tracker, not the environment's."""
    from stratakb import mcp_server
    from stratakb.config import get_settings
    from stratakb.learning.tracker import Tracker

    env = get_settings().model_copy(update={"learning_enabled": False})
    a = env.model_copy(update={"learning_enabled": True, "learning_store_raw_queries": True})
    b = env.model_copy(update={"learning_enabled": True, "learning_store_raw_queries": False})

    with (
        patch("stratakb.config.get_settings", return_value=env),
        patch("stratakb.mcp_server.get_settings", return_value=env),
        patch("stratakb.learning.tracker.Tracker", wraps=Tracker) as tracker_cls,
    ):
        mcp_server.build_server(a)
        mcp_server.build_server(b)

    assert len(tracker_cls.call_args_list) == 2
    assert tracker_cls.call_args_list[0].args[1] is a
    assert tracker_cls.call_args_list[1].args[1] is b


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
        await call_tool(server_a, "strata_admin_health")
        await call_tool(server_b, "strata_admin_health")

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
