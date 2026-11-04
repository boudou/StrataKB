"""Tests for RAG tools.

Tests cover:
- Layer detection (NLP)
- Metadata hint extraction
- Confidence scoring
- Query rewriting (mocked)
- strata_ask tool integration
- MCP tool registration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ==========================================
# Layer detection tests
# ==========================================

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def example_patterns():
    """Routing patterns from the example SEPA and ISO 20022 presets."""
    from stratakb.knowledge.presets import load_preset
    from stratakb.rag.layer_detect import compile_patterns

    raw = {}
    for name in ("sepa", "iso20022"):
        preset = load_preset(EXAMPLES_DIR / name / "preset.yaml")
        raw[preset.name] = preset.routing_patterns
    return compile_patterns(raw)


class TestLayerDetection:
    def test_detect_sepa(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "sepa" in detect_layers("SEPA Credit Transfer rejection codes", example_patterns)

    def test_detect_pain_message(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "iso20022" in detect_layers("pain.001 message structure", example_patterns)

    def test_detect_camt_message(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "iso20022" in detect_layers("camt.053 bank statement", example_patterns)

    def test_detect_iso20022(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "iso20022" in detect_layers("ISO 20022 external code sets", example_patterns)

    def test_detect_multiple_layers(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        layers = detect_layers("SEPA rulebook rules for pacs.008", example_patterns)
        assert "sepa" in layers
        assert "iso20022" in layers

    def test_no_layer_detected(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert detect_layers("general payments question", example_patterns) == []

    def test_empty_query(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert detect_layers("", example_patterns) == []

    def test_no_patterns_detects_nothing(self):
        from stratakb.rag.layer_detect import detect_layers

        assert detect_layers("SEPA Credit Transfer") == []
        assert detect_layers("SEPA Credit Transfer", {}) == []

    def test_detect_sct_inst(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "sepa" in detect_layers("SCT Inst instant payment", example_patterns)

    def test_case_insensitive(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "sepa" in detect_layers("sepa credit transfer", example_patterns)
        assert "iso20022" in detect_layers("Iso 20022 Messages", example_patterns)

    def test_word_boundaries(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        # "epc" must not match inside another word
        assert "sepa" not in detect_layers("receptacle sizing", example_patterns)

    def test_invalid_pattern_is_dropped(self):
        from stratakb.rag.layer_detect import compile_patterns, detect_layers

        patterns = compile_patterns({"runbooks": ["(unclosed", r"\brunbook\b"]})
        assert len(patterns["runbooks"]) == 1
        assert detect_layers("which runbook covers failover", patterns) == ["runbooks"]

    def test_layer_without_valid_patterns_is_omitted(self):
        from stratakb.rag.layer_detect import compile_patterns

        assert compile_patterns({"empty": [], "broken": ["("]}) == {}


# ==========================================
# Metadata hint extraction tests
# ==========================================


class TestMetadataHints:
    def test_message_identifiers_add_no_filter(self):
        from stratakb.rag.layer_detect import extract_metadata_hints

        # Chunks carry no message_type metadata, so a filter would match nothing
        assert extract_metadata_hints("pain.001 GroupHeader fields") == {}

    def test_yaml_doc_type(self):
        from stratakb.rag.layer_detect import extract_metadata_hints

        hints = extract_metadata_hints("show me the yaml API specs")
        assert hints.get("doc_type") == "yaml"

    def test_pdf_doc_type(self):
        from stratakb.rag.layer_detect import extract_metadata_hints

        hints = extract_metadata_hints("what does the guide say about accounting")
        assert hints.get("doc_type") == "pdf"

    def test_no_hints(self):
        from stratakb.rag.layer_detect import extract_metadata_hints

        hints = extract_metadata_hints("how does payment work")
        assert hints == {}


# ==========================================
# Confidence scoring tests
# ==========================================


class TestConfidenceScoring:
    def test_high_confidence(self):
        from stratakb.rag.confidence import score_confidence

        result = score_confidence([0.92, 0.85, 0.81, 0.75])
        assert result["level"] == "high"
        assert result["top_score"] == 0.92
        assert result["relevant_sources"] >= 3

    def test_medium_confidence(self):
        from stratakb.rag.confidence import score_confidence

        result = score_confidence([0.72, 0.65])
        assert result["level"] == "medium"

    def test_low_confidence(self):
        from stratakb.rag.confidence import score_confidence

        result = score_confidence([0.45, 0.35])
        assert result["level"] == "low"
        assert "warning" in result

    def test_no_results(self):
        from stratakb.rag.confidence import score_confidence

        result = score_confidence([])
        assert result["level"] == "low"
        assert "warning" in result
        assert result["relevant_sources"] == 0

    def test_custom_thresholds(self):
        from stratakb.rag.confidence import ConfidenceThresholds, score_confidence

        thresholds = ConfidenceThresholds(high_score=0.9, medium_score=0.7)
        result = score_confidence([0.85, 0.82, 0.80], thresholds=thresholds)
        assert result["level"] == "medium"  # top_score < 0.9

    def test_single_high_result(self):
        from stratakb.rag.confidence import score_confidence

        # Only 1 relevant source — not enough for "high" (needs 3+)
        result = score_confidence([0.95])
        assert result["level"] == "medium"


# ==========================================
# Query rewriter tests
# ==========================================


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_original(self):
        from stratakb.rag.query_rewriter import rewrite_query

        result = await rewrite_query("How does payment work?", anthropic_api_key="")
        assert result == ["How does payment work?"]

    @pytest.mark.asyncio
    async def test_rewrite_with_mocked_api(self):
        from stratakb.rag.query_rewriter import rewrite_query

        mock_response = AsyncMock()
        mock_response.content = [
            AsyncMock(text='["payment initiation pain.001 flow", "SEPA SCT payment flow"]')
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            result = await rewrite_query(
                "How does payment work?",
                anthropic_api_key="fake-key",
                model="claude-haiku-5-5",
            )

        assert len(result) == 2
        assert "pain.001" in result[0]

    @pytest.mark.asyncio
    async def test_rewrite_prompt_includes_domain_context(self):
        from stratakb.rag.query_rewriter import build_domain_context, rewrite_query

        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text='["a", "b"]')]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        context = build_domain_context({"sepa": "SEPA Payments: rulebooks"})
        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            await rewrite_query(
                "How does payment work?",
                anthropic_api_key="fake-key",
                model="claude-haiku-5-5",
                domain_context=context,
            )

        system = mock_client.messages.create.call_args.kwargs["system"]
        assert "**sepa**: SEPA Payments: rulebooks" in system
        assert "{domain_context}" not in system

    def test_domain_context_without_layers(self):
        from stratakb.rag.query_rewriter import build_domain_context

        assert "documentation" in build_domain_context({})

    @pytest.mark.asyncio
    async def test_rewrite_api_error_fallback(self):
        from stratakb.rag.query_rewriter import rewrite_query

        with patch("anthropic.AsyncAnthropic", side_effect=Exception("API error")):
            result = await rewrite_query(
                "How does payment work?",
                anthropic_api_key="fake-key",
                model="claude-haiku-5-5",
            )

        assert result == ["How does payment work?"]


# ==========================================
# MCP tool registration tests
# ==========================================


class TestRagToolRegistration:
    def test_ask_tool_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_ask" in tool_names

    def test_search_tool_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_search" in tool_names


# ==========================================
# User management tool registration
# ==========================================


class TestUserToolRegistration:
    def test_user_tools_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_whoami" in tool_names
        assert "strata_admin_list_users" in tool_names
        assert "strata_admin_update_role" in tool_names
