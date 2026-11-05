"""Test the health check tool."""

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


def test_settings_readers_use_the_live_settings():
    """Modules that bind get_settings at import hold the real one, not a test's patch."""
    import stratakb.config
    import stratakb.learning.tracker
    import stratakb.mcp_server

    assert stratakb.mcp_server.get_settings is stratakb.config.get_settings
    assert stratakb.learning.tracker.get_settings is stratakb.config.get_settings
    assert (
        stratakb.mcp_server.settings.database_url
        == stratakb.config.get_settings().database_url
    )
