"""Test the health check tool."""

import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import pytest

from tests.helpers import call_tool


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
        "assert not hasattr(stratakb.mcp_server, 'get_settings')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_no_module_binds_get_settings_at_import():
    """No src module imports get_settings at module level.

    A module first imported while a test patches stratakb.config.get_settings
    would keep the mock for the rest of the session (#170).
    """
    src = Path(__file__).resolve().parent.parent / "src" / "stratakb"
    bound = []
    for path in sorted(src.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "stratakb.config"
                and any(alias.name == "get_settings" for alias in node.names)
            ):
                bound.append(path.relative_to(src).as_posix())
    assert bound == []


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


def test_version_matches_pyproject():
    """The package version is the one pyproject.toml writes."""
    import tomllib

    import stratakb

    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text("utf-8"))
    assert stratakb.__version__ == data["project"]["version"]


def test_server_reports_strata_version(server):
    """A client sees the StrataKB version on connect, not the MCP SDK's."""
    import importlib.metadata

    import stratakb

    version = server._mcp_server.create_initialization_options().server_version
    assert version == stratakb.__version__
    assert version != importlib.metadata.version("mcp")


async def test_health_reports_version():
    """strata_admin_health reports the StrataKB version."""
    import stratakb
    from stratakb.config import get_settings
    from tests.helpers import call_health

    result = await call_health(get_settings())
    assert result["version"] == stratakb.__version__


def test_version_falls_back_when_not_installed():
    """A source tree that was never installed reports 0+unknown."""
    import importlib
    import importlib.metadata

    import stratakb

    try:
        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("stratakb"),
        ):
            importlib.reload(stratakb)
            assert stratakb.__version__ == "0+unknown"
    finally:
        importlib.reload(stratakb)
    assert stratakb.__version__ != "0+unknown"
