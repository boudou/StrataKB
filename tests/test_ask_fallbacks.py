"""strata_ask keeps its fallbacks but logs each failed step (#198)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from stratakb.config import Settings
from stratakb.rag.tools import _detect_layers, register_rag_tools
from tests.helpers import capture_tools

DATABASE_URL = "postgresql://test"
QUESTION = "how is a SEPA rejection handled?"


class _BrokenError(Exception):
    pass


def _raise(*_args, **_kwargs):
    raise _BrokenError("unreachable")


def _warnings(logs: list[dict], event: str) -> list[dict]:
    return [e for e in logs if e["event"] == event and e["log_level"] == "warning"]


def _assert_logged(logs: list[dict], event: str) -> None:
    entries = _warnings(logs, event)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["error_type"] == "_BrokenError"
    # The error type only: no question text, no error message
    assert set(entry) == {"event", "log_level", "error_type"}


class TestDetectLayers:
    @pytest.mark.asyncio
    async def test_routing_patterns_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "stratakb.knowledge.layers.get_layer_routing", AsyncMock(side_effect=_BrokenError("x"))
        )

        with capture_logs() as logs:
            layers = await _detect_layers(DATABASE_URL, QUESTION, learning_enabled=False)

        assert layers == []
        _assert_logged(logs, "routing_patterns_unavailable")

    @pytest.mark.asyncio
    async def test_layer_affinities_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "stratakb.knowledge.layers.get_layer_routing",
            AsyncMock(return_value={"patterns": {"sepa": [r"\bsepa\b"]}, "descriptions": {}}),
        )
        monkeypatch.setattr(
            "stratakb.learning.store.LearningStore.find_layer_affinities",
            AsyncMock(side_effect=_BrokenError("x")),
        )

        with capture_logs() as logs:
            layers = await _detect_layers(DATABASE_URL, QUESTION, learning_enabled=True)

        # Falls back to the routing patterns alone
        assert layers == ["sepa"]
        _assert_logged(logs, "layer_affinities_unavailable")


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "learning_enabled": False,
        "embedding_backend": "local",
        "reranking_enabled": False,
        "strata_user": "test-user",
        "strata_role": "viewer",
    }
    values.update(overrides)
    return Settings(**values)


def _register(monkeypatch, settings: Settings) -> tuple[dict, AsyncMock]:
    monkeypatch.setattr("stratakb.config.get_settings", lambda: settings)
    monkeypatch.setattr("stratakb.rag.tools.write_audit_log", AsyncMock())
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "stratakb.knowledge.layers.get_answer_modes", AsyncMock(return_value=["grounded"])
    )
    rewrite = AsyncMock(return_value=[QUESTION])
    monkeypatch.setattr("stratakb.rag.query_rewriter.rewrite_query", rewrite)
    return capture_tools(register_rag_tools, DATABASE_URL), rewrite


class TestAsk:
    @pytest.mark.asyncio
    async def test_learning_store_unavailable(self, monkeypatch):
        monkeypatch.setattr("stratakb.learning.store.LearningStore.__init__", _raise)
        tools, rewrite = _register(monkeypatch, _settings(learning_enabled=True))

        with capture_logs() as logs:
            response = await tools["strata_ask"](question=QUESTION, layers=["sepa"])

        assert "error" not in response
        # The question is still rewritten, without learned rewrites
        assert rewrite.call_args.kwargs["learning_store"] is None
        _assert_logged(logs, "learning_store_unavailable")

    @pytest.mark.asyncio
    async def test_domain_context_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "stratakb.knowledge.layers.get_layer_routing", AsyncMock(side_effect=_BrokenError("x"))
        )
        monkeypatch.setattr(
            "stratakb.rag.query_rewriter.rewrite_configured", lambda *_a, **_k: True
        )
        tools, rewrite = _register(monkeypatch, _settings())

        with capture_logs() as logs:
            response = await tools["strata_ask"](question=QUESTION, layers=["sepa"])

        assert "error" not in response
        assert rewrite.call_args.kwargs["domain_context"] == ""
        _assert_logged(logs, "domain_context_unavailable")
