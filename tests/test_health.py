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
