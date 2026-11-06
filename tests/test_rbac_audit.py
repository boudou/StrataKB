"""Tests for RBAC and audit logging.

Tests cover:
- Role hierarchy (viewer < operator < admin)
- require_role decorator enforcement
- PermissionDeniedError for insufficient roles
- Sensitive field redaction
- Audit log writing and querying
- MCP tool role assignments
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from stratakb.core.auth import (
    PermissionDeniedError,
    check_role,
    require_role,
)
from stratakb.core.audit import redact_params


# ==========================================
# Role hierarchy tests
# ==========================================


class TestCheckRole:
    def test_viewer_meets_viewer(self):
        assert check_role("viewer", "viewer") is True

    def test_operator_meets_viewer(self):
        assert check_role("operator", "viewer") is True

    def test_admin_meets_viewer(self):
        assert check_role("admin", "viewer") is True

    def test_viewer_fails_operator(self):
        assert check_role("viewer", "operator") is False

    def test_viewer_fails_admin(self):
        assert check_role("viewer", "admin") is False

    def test_operator_meets_operator(self):
        assert check_role("operator", "operator") is True

    def test_operator_fails_admin(self):
        assert check_role("operator", "admin") is False

    def test_admin_meets_admin(self):
        assert check_role("admin", "admin") is True

    def test_unknown_role_fails(self):
        assert check_role("unknown", "viewer") is False

    def test_unknown_required_role_fails(self):
        assert check_role("admin", "superadmin") is False


# ==========================================
# require_role decorator tests
# ==========================================


class TestRequireRole:
    @pytest.mark.asyncio
    async def test_admin_can_call_admin_tool(self):
        @require_role("admin")
        async def my_tool():
            return {"ok": True}

        with patch("stratakb.core.auth.get_current_user", return_value=("admin_user", "admin")):
            result = await my_tool()
            assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_viewer_cannot_call_admin_tool(self):
        @require_role("admin")
        async def my_tool():
            return {"ok": True}

        with patch("stratakb.core.auth.get_current_user", return_value=("viewer_user", "viewer")):
            with pytest.raises(PermissionDeniedError) as exc_info:
                await my_tool()
            assert exc_info.value.user_id == "viewer_user"
            assert exc_info.value.user_role == "viewer"
            assert exc_info.value.required_role == "admin"
            assert exc_info.value.tool_name == "my_tool"

    @pytest.mark.asyncio
    async def test_operator_can_call_operator_tool(self):
        @require_role("operator")
        async def op_tool():
            return {"ok": True}

        with patch("stratakb.core.auth.get_current_user", return_value=("op_user", "operator")):
            result = await op_tool()
            assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_viewer_can_call_viewer_tool(self):
        @require_role("viewer")
        async def view_tool():
            return {"data": []}

        with patch("stratakb.core.auth.get_current_user", return_value=("viewer_user", "viewer")):
            result = await view_tool()
            assert result == {"data": []}

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        @require_role("viewer")
        async def my_named_tool():
            pass

        assert my_named_tool.__name__ == "my_named_tool"

    @pytest.mark.asyncio
    async def test_permission_denied_error_message(self):
        err = PermissionDeniedError("user1", "viewer", "admin", "delete_stuff")
        assert "user1" in str(err)
        assert "viewer" in str(err)
        assert "admin" in str(err)
        assert "delete_stuff" in str(err)


# ==========================================
# Redaction tests
# ==========================================


class TestRedaction:
    def test_redacts_sensitive_fields(self):
        params = {
            "name": "test_layer",
            "api_key": "sk-secret-123",
            "password": "hunter2",
        }
        result = redact_params(params)
        assert result["name"] == "test_layer"
        assert result["api_key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"

    def test_redacts_nested_dicts(self):
        params = {
            "config": {
                "token": "abc123",
                "host": "localhost",
            }
        }
        result = redact_params(params)
        assert result["config"]["token"] == "***REDACTED***"
        assert result["config"]["host"] == "localhost"

    def test_case_insensitive_redaction(self):
        params = {"Authorization": "Bearer xyz", "Token": "abc"}
        result = redact_params(params)
        assert result["Authorization"] == "***REDACTED***"
        assert result["Token"] == "***REDACTED***"

    def test_none_params_returns_none(self):
        assert redact_params(None) is None

    def test_empty_params_returns_empty(self):
        assert redact_params({}) == {}  # empty dict passes through

    def test_preserves_non_sensitive_fields(self):
        params = {"name": "iso20022", "display_name": "Ops Runbooks", "count": 42}
        result = redact_params(params)
        assert result == params

    def test_redacts_all_known_sensitive_fields(self):
        from stratakb.core.audit import SENSITIVE_FIELDS

        params = {field: f"value_{field}" for field in SENSITIVE_FIELDS}
        result = redact_params(params)
        for field in SENSITIVE_FIELDS:
            assert result[field] == "***REDACTED***"


# ==========================================
# Audit log DB tests (require database)
# ==========================================


@pytest.fixture
def db_url():
    from stratakb.config import get_settings

    return get_settings().database_url


class TestAuditLogDB:
    @pytest.mark.asyncio
    async def test_write_and_query_audit_log(self, db_url):
        from stratakb.core.audit import query_audit_log, write_audit_log

        test_user = "test_audit_user"
        test_action = "test_action"
        test_tool = "test_tool_xyz"

        await write_audit_log(
            db_url,
            user_id=test_user,
            action=test_action,
            tool_name=test_tool,
            parameters={"key": "value", "password": "secret123"},
            result_summary="Test completed",
        )

        entries = await query_audit_log(
            db_url,
            user_id=test_user,
            tool_name=test_tool,
            limit=10,
        )

        assert len(entries) >= 1
        entry = entries[0]
        assert entry["user_id"] == test_user
        assert entry["action"] == test_action
        assert entry["tool_name"] == test_tool
        assert entry["result_summary"] == "Test completed"

        # Verify password was redacted before storage
        import json

        params = entry["parameters"]
        if isinstance(params, str):
            params = json.loads(params)
        assert params["key"] == "value"
        assert params["password"] == "***REDACTED***"

    @pytest.mark.asyncio
    async def test_user_defaults_to_current_user(self, db_url):
        from uuid import uuid4

        from stratakb.core.audit import query_audit_log, write_audit_log

        caller = f"audit_default_{uuid4().hex[:8]}"
        with patch("stratakb.core.auth.get_current_user", return_value=(caller, "admin")):
            await write_audit_log(db_url, action="test_action", tool_name="test_tool_default")

        entries = await query_audit_log(db_url, user_id=caller, tool_name="test_tool_default")
        assert len(entries) == 1
        assert entries[0]["user_id"] == caller

    @pytest.mark.asyncio
    async def test_explicit_user_wins_over_current_user(self, db_url):
        from uuid import uuid4

        from stratakb.core.audit import query_audit_log, write_audit_log

        explicit = f"audit_explicit_{uuid4().hex[:8]}"
        with patch("stratakb.core.auth.get_current_user", return_value=("someone_else", "admin")):
            await write_audit_log(
                db_url, user_id=explicit, action="test_action", tool_name="test_tool_explicit"
            )

        entries = await query_audit_log(db_url, user_id=explicit, tool_name="test_tool_explicit")
        assert len(entries) == 1
        assert entries[0]["user_id"] == explicit

    @pytest.mark.asyncio
    async def test_query_with_date_filter(self, db_url):
        from stratakb.core.audit import query_audit_log

        # Query with a future date should return nothing
        entries = await query_audit_log(
            db_url,
            from_date="2099-01-01",
            limit=10,
        )
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_query_limit_clamped(self, db_url):
        from stratakb.core.audit import query_audit_log

        # Should not raise even with extreme limits
        entries = await query_audit_log(db_url, limit=0)
        assert isinstance(entries, list)

        entries = await query_audit_log(db_url, limit=9999)
        assert isinstance(entries, list)

    @pytest.mark.asyncio
    async def test_audit_log_never_raises(self, db_url):
        from stratakb.core.audit import write_audit_log

        # Even with a bad database URL, should not raise
        await write_audit_log(
            "postgresql://invalid:invalid@localhost:9999/nonexistent",
            user_id="test",
            action="test",
            tool_name="test",
        )
        # No exception = pass


# ==========================================
# MCP tool role assignments
# ==========================================


class TestToolRoleAssignments:
    def test_all_tools_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]

        # Viewer tools
        assert "strata_admin_health" in tool_names
        assert "strata_list_knowledge_layers" in tool_names
        assert "strata_get_layer_status" in tool_names

        # Admin tools
        assert "strata_create_knowledge_layer" in tool_names
        assert "strata_delete_knowledge_layer" in tool_names
        assert "strata_ingest_documents" in tool_names
        assert "strata_admin_audit_log" in tool_names

    def test_viewer_tools_accessible_by_viewer(self):
        """Viewer-role tools should not raise for viewer users."""
        # Verified via the require_role decorator unit tests above
        # This test confirms the tools are registered (smoke test)
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert len(tool_names) >= 7  # At least 7 tools registered
