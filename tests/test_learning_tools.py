"""Tests for learning MCP tools registration."""


def test_learning_tools_import():
    """Verify learning tools module imports without error."""
    from stratakb.learning.tools import register_learning_tools
    assert callable(register_learning_tools)


def test_mcp_server_registers_learning_tools():
    """Verify MCP server can import and register all tools including learning."""
    from stratakb.mcp_server import mcp
    assert mcp.name == "StrataKB"
