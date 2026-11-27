"""Every tool declares its MCP annotations.

The rules behind the table:
- Read-only means the tool changes no layer, document, chunk, learned pattern,
  user role or setting. Bookkeeping writes (audit entries, interaction rows,
  receipts, ingestion-run history, the caller's own user row) do not count.
- openWorldHint is False everywhere: each tool works on this server's own
  knowledge base, database and local files. Outbound providers depend on
  settings and are reported by strata_admin_health, not by fixed annotations.
- Read-only tools set only readOnlyHint and openWorldHint; destructiveHint and
  idempotentHint mean something only for tools that write.
- Each tool is marked by its worst case (strata_apply_preset without dry_run).
- Idempotent means a second call with the same arguments has no further
  effect on the data.
"""

from __future__ import annotations

import pytest
from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def _writes(*, destructive: bool, idempotent: bool) -> ToolAnnotations:
    return ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=False,
    )


EXPECTED = {
    "strata_search": READ_ONLY,
    "strata_ask": READ_ONLY,
    "strata_verify": READ_ONLY,
    "strata_get_receipt": READ_ONLY,
    "strata_list_knowledge_layers": READ_ONLY,
    "strata_get_layer_status": READ_ONLY,
    "strata_create_knowledge_layer": _writes(destructive=False, idempotent=True),
    "strata_set_archived_patterns": _writes(destructive=True, idempotent=True),
    "strata_delete_knowledge_layer": _writes(destructive=True, idempotent=True),
    "strata_ingest_documents": _writes(destructive=True, idempotent=True),
    "strata_apply_preset": _writes(destructive=True, idempotent=True),
    "strata_feedback": _writes(destructive=False, idempotent=False),
    "strata_learning_stats": READ_ONLY,
    "strata_reset_learning": _writes(destructive=True, idempotent=True),
    "strata_admin_health": READ_ONLY,
    "strata_admin_audit_log": READ_ONLY,
    "strata_whoami": READ_ONLY,
    "strata_admin_list_users": READ_ONLY,
    "strata_admin_update_role": _writes(destructive=True, idempotent=True),
    "strata_db_schema": READ_ONLY,
    "strata_db_vector_stats": READ_ONLY,
    "strata_db_sample_chunks": READ_ONLY,
    "strata_db_query": READ_ONLY,
    "strata_db_explain": READ_ONLY,
}


@pytest.fixture
async def tools():
    """All tools of a server built with the SQL tools registered (no database needed)."""
    from stratakb import mcp_server
    from stratakb.config import get_settings

    settings = get_settings().model_copy(
        update={"readonly_database_url": "postgresql://ro:x@127.0.0.1:1/db"}
    )
    server = mcp_server.build_server(settings)
    return await server.list_tools()


async def test_every_tool_has_the_expected_annotations(tools):
    assert len(tools) == 24
    assert {tool.name: tool.annotations for tool in tools} == EXPECTED


async def test_read_only_tools_set_no_write_hints(tools):
    for tool in tools:
        assert tool.annotations is not None, tool.name
        if tool.annotations.readOnlyHint is True:
            assert tool.annotations.destructiveHint is None, tool.name
            assert tool.annotations.idempotentHint is None, tool.name


async def test_no_tool_is_open_world(tools):
    for tool in tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.openWorldHint is False, tool.name
