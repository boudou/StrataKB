"""Audit entries for strata_search and strata_ask keep questions scrubbed."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from stratakb.config import Settings
from stratakb.rag.tools import _audit_text, register_rag_tools
from tests.helpers import capture_tools

IBAN = "GR1601101250000000012300695"
QUESTION = f"why was {IBAN} rejected?"
SCRUBBED = "why was [IBAN_REDACTED] rejected?"

# (audit_store_raw_queries, learning_store_raw_queries)
SETTING_CASES = [(False, False), (False, True), (True, False)]


def _settings(audit_raw: bool, learning_raw: bool, user: str = "test-user") -> Settings:
    return Settings(
        _env_file=None,
        learning_enabled=False,
        learning_store_raw_queries=learning_raw,
        audit_store_raw_queries=audit_raw,
        embedding_backend="local",
        reranking_enabled=False,
        strata_user=user,
        strata_role="viewer",
    )


def _register(monkeypatch, settings: Settings, database_url: str) -> dict:
    monkeypatch.setattr("stratakb.config.get_settings", lambda: settings)
    monkeypatch.setattr("stratakb.rag.tools._detect_layers", AsyncMock(return_value=[]))
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[]))
    return capture_tools(register_rag_tools, database_url)


class TestAuditedQuestion:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("audit_raw", "learning_raw"), SETTING_CASES)
    async def test_search_entry(self, monkeypatch, audit_raw, learning_raw):
        audit = AsyncMock()
        monkeypatch.setattr("stratakb.rag.tools.write_audit_log", audit)
        tools = _register(monkeypatch, _settings(audit_raw, learning_raw), "postgresql://test")

        response = await tools["strata_search"](QUESTION, top_k=1)

        stored = audit.call_args.kwargs["parameters"]["query"]
        assert stored == (QUESTION if audit_raw else SCRUBBED)
        assert response["query"] == QUESTION

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("audit_raw", "learning_raw"), SETTING_CASES)
    async def test_ask_entry(self, monkeypatch, audit_raw, learning_raw):
        audit = AsyncMock()
        monkeypatch.setattr("stratakb.rag.tools.write_audit_log", audit)
        monkeypatch.setattr(
            "stratakb.knowledge.layers.get_answer_modes", AsyncMock(return_value=["grounded"])
        )
        tools = _register(monkeypatch, _settings(audit_raw, learning_raw), "postgresql://test")

        await tools["strata_ask"](question=QUESTION, layers=["sepa"])

        stored = audit.call_args.kwargs["parameters"]["question"]
        assert stored == (QUESTION if audit_raw else SCRUBBED)


class TestAuditText:
    def test_scrubs_bic(self):
        assert _audit_text("route via ETHNGRAA please", False) == "route via [BIC_REDACTED] please"

    def test_scrubs_account_number(self):
        assert _audit_text("account 12345678901234 blocked", False) == (
            "account [ACCT_REDACTED] blocked"
        )

    def test_keeps_short_codes(self):
        assert _audit_text("reason code 0401", False) == "reason code 0401"

    def test_raw_mode_only_cuts(self):
        text = IBAN + "z" * 300
        assert _audit_text(text, True) == text[:200]

    def test_scrubs_before_cutting(self):
        # The IBAN starts at character 190 and straddles the 200-character cut
        text = "x" * 189 + " " + IBAN + " " + "y" * 32
        assert len(text) == 250
        assert _audit_text(text, True).endswith("GR16011012")
        stored = _audit_text(text, False)
        # The placeholder itself is cut; no part of the IBAN is left
        assert stored == "x" * 189 + " [IBAN_REDA"
        assert "GR16" not in stored

    def test_whole_placeholder_when_it_fits(self):
        text = "x" * 184 + " " + IBAN + " " + "y" * 37
        stored = _audit_text(text, False)
        assert "[IBAN_REDACTED]" in stored
        assert "GR16" not in stored
        assert len(stored) <= 200


class TestStoredEntry:
    @pytest.mark.asyncio
    async def test_audit_log_row_holds_placeholder(self, monkeypatch, db_url):
        from stratakb.core.audit import query_audit_log

        user = f"test-audit-scrub-{uuid4().hex[:10]}"
        tools = _register(monkeypatch, _settings(False, False, user=user), db_url)

        await tools["strata_search"](QUESTION, top_k=1)

        entries = await query_audit_log(db_url, user_id=user, tool_name="strata_search")
        assert len(entries) == 1
        parameters = entries[0]["parameters"]
        if isinstance(parameters, str):
            import json
            parameters = json.loads(parameters)
        assert parameters["query"] == SCRUBBED
        assert IBAN not in str(entries[0])
