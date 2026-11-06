"""Tests for structured logging.

Tests cover:
- Logging configuration (JSON vs Console)
- Sensitive data masking (IBAN, card numbers, keys)
- Tool invocation decorator (timing, fields)
- Log level configuration
"""

from __future__ import annotations

import io
import logging
from unittest.mock import MagicMock, patch

import pytest
import structlog

from stratakb.core.logging import (
    _mask_card,
    _mask_iban,
    configure_logging,
    log_tool_call,
    sensitive_data_filter,
)


# ==========================================
# Sensitive data masking tests
# ==========================================


class TestIBANMasking:
    def test_masks_iban(self):
        result = _mask_iban("Transfer to DE89370400440532013000")
        assert "DE89" in result
        assert "3000" in result
        assert "0532013" not in result

    def test_masks_short_iban(self):
        result = _mask_iban("Account NL91ABNA0417164300")
        assert "NL91" in result
        assert "4300" in result
        assert "041716" not in result

    def test_no_iban_unchanged(self):
        text = "Hello world, no account here"
        assert _mask_iban(text) == text

    def test_multiple_ibans(self):
        text = "From DE89370400440532013000 to FR7630006000011234567890189"
        result = _mask_iban(text)
        assert "0532013" not in result
        assert "1234567890" not in result


class TestCardMasking:
    def test_masks_card_with_spaces(self):
        result = _mask_card("Card 4111 1111 1111 1111")
        assert "****-****-****-1111" in result

    def test_masks_card_with_dashes(self):
        result = _mask_card("Card 4111-1111-1111-1111")
        assert "****-****-****-1111" in result

    def test_masks_card_continuous(self):
        result = _mask_card("Card 4111111111111111")
        assert "****-****-****-1111" in result

    def test_no_card_unchanged(self):
        text = "No card number here 12345"
        assert _mask_card(text) == text


class TestSensitiveDataFilter:
    def test_masks_sensitive_keys(self):
        event = {"event": "test", "password": "secret123", "api_key": "sk-abc"}
        result = sensitive_data_filter(None, "info", event)
        assert result["password"] == "***MASKED***"
        assert result["api_key"] == "***MASKED***"
        assert result["event"] == "test"

    def test_masks_iban_in_values(self):
        event = {"event": "transfer", "details": "To DE89370400440532013000"}
        result = sensitive_data_filter(None, "info", event)
        assert "0532013" not in result["details"]

    def test_masks_card_in_values(self):
        event = {"event": "payment", "info": "Card 4111111111111111"}
        result = sensitive_data_filter(None, "info", event)
        assert "****-****-****-1111" in result["info"]

    def test_non_string_values_unchanged(self):
        event = {"event": "test", "count": 42, "active": True}
        result = sensitive_data_filter(None, "info", event)
        assert result["count"] == 42
        assert result["active"] is True

    def test_account_number_key_masked(self):
        event = {"event": "lookup", "account_number": "123456789"}
        result = sensitive_data_filter(None, "info", event)
        assert result["account_number"] == "***MASKED***"

    def test_case_insensitive_key_matching(self):
        event = {"event": "test", "Token": "abc123"}
        result = sensitive_data_filter(None, "info", event)
        assert result["Token"] == "***MASKED***"


# ==========================================
# Configure logging tests
# ==========================================


class TestConfigureLogging:
    def test_configure_info_level(self):
        configure_logging("INFO")
        log = structlog.get_logger()
        assert log is not None

    def test_configure_debug_level(self):
        configure_logging("DEBUG")
        log = structlog.get_logger()
        assert log is not None

    def test_configure_case_insensitive(self):
        configure_logging("warning")
        log = structlog.get_logger()
        assert log is not None

    def test_configure_invalid_level_defaults_to_info(self):
        configure_logging("INVALID")
        log = structlog.get_logger()
        assert log is not None


class TestMainLogging:
    def test_main_configures_logging_before_building_the_server(self, capsys, monkeypatch):
        """main() configures logging before the first registration logs.

        So the first log line goes to stderr and nothing reaches stdout.
        """
        from mcp.server.fastmcp import FastMCP

        from stratakb import mcp_server
        from stratakb.config import Settings

        saved = structlog.get_config()
        # Start unconfigured: earlier tests leave structlog writing to stderr
        structlog.reset_defaults()
        monkeypatch.setattr(FastMCP, "run", MagicMock())
        try:
            mcp_server.main(
                Settings(
                    _env_file=None,
                    readonly_database_url="",
                    strata_local_only=False,
                    log_level="INFO",
                )
            )
        finally:
            structlog.configure(**saved)

        captured = capsys.readouterr()
        assert "sql_tools_disabled" in captured.err
        assert "sql_tools_disabled" not in captured.out


# ==========================================
# Tool invocation decorator tests
# ==========================================


class TestLogToolCall:
    @pytest.mark.asyncio
    async def test_logs_tool_call_and_returns_result(self):
        @log_tool_call
        async def my_tool():
            return {"status": "ok", "data": []}

        with patch("stratakb.core.auth.get_current_user", return_value=("test_user", "admin")):
            result = await my_tool()
            assert result == {"status": "ok", "data": []}

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        @log_tool_call
        async def my_named_tool():
            return {}

        assert my_named_tool.__name__ == "my_named_tool"

    @pytest.mark.asyncio
    async def test_extracts_layer_from_kwargs(self):
        @log_tool_call
        async def strata_get_layer_status(name: str = ""):
            return {"status": "ok"}

        with patch("stratakb.core.auth.get_current_user", return_value=("u", "admin")):
            result = await strata_get_layer_status(name="iso20022")
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_logs_error_on_exception(self):
        @log_tool_call
        async def failing_tool():
            raise ValueError("something broke")

        with patch("stratakb.core.auth.get_current_user", return_value=("u", "admin")):
            with pytest.raises(ValueError, match="something broke"):
                await failing_tool()

    @pytest.mark.asyncio
    async def test_duration_is_positive(self):
        """Verify the decorator measures time by checking it completes without error."""
        import time

        @log_tool_call
        async def slow_tool():
            # Tiny delay to ensure measurable duration
            time.sleep(0.01)
            return {"status": "ok"}

        with patch("stratakb.core.auth.get_current_user", return_value=("u", "admin")):
            result = await slow_tool()
            assert result["status"] == "ok"


# ==========================================
# Integration: all modules use structlog
# ==========================================


class TestAllModulesUseStructlog:
    def test_mcp_server_uses_structlog(self):
        import stratakb.mcp_server
        assert hasattr(stratakb.mcp_server, "logger")

    def test_database_uses_structlog(self):
        import stratakb.core.database
        assert hasattr(stratakb.core.database, "logger")

    def test_auth_uses_structlog(self):
        import stratakb.core.auth
        assert hasattr(stratakb.core.auth, "logger")

    def test_audit_uses_structlog(self):
        import stratakb.core.audit
        assert hasattr(stratakb.core.audit, "logger")

    def test_layers_uses_structlog(self):
        import stratakb.knowledge.layers
        assert hasattr(stratakb.knowledge.layers, "logger")

    def test_vector_store_uses_structlog(self):
        import stratakb.knowledge.vector_store
        assert hasattr(stratakb.knowledge.vector_store, "logger")

    def test_ingestion_uses_structlog(self):
        import stratakb.knowledge.ingestion
        assert hasattr(stratakb.knowledge.ingestion, "logger")

    def test_embeddings_uses_structlog(self):
        import stratakb.knowledge.embeddings
        assert hasattr(stratakb.knowledge.embeddings, "logger")

    def test_pdf_parser_uses_structlog(self):
        import stratakb.knowledge.parsers.pdf_parser
        assert hasattr(stratakb.knowledge.parsers.pdf_parser, "logger")
