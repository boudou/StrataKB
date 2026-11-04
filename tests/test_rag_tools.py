"""Tests for RAG tools.

Tests cover:
- Layer detection (NLP)
- Metadata hint extraction
- Confidence scoring
- Query rewriting (mocked), through Anthropic or an OpenAI-compatible endpoint
- Lenient parsing of the rewrite reply
- strata_ask tool integration
- MCP tool registration
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
# Query rewriting through an OpenAI-compatible endpoint
# ==========================================

_LOCAL_URL = "http://127.0.0.1:11434/v1"


@pytest.fixture
def no_sdk_env(monkeypatch):
    """Keep SDK keys and base-URL overrides from the shell out of the tests."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def _rewrite_settings(**overrides):
    from stratakb.config import Settings

    values = {"strata_user": "test", "strata_role": "viewer"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _compatible_settings(**overrides):
    values = {
        "query_rewrite_provider": "openai_compatible",
        "query_rewrite_base_url": _LOCAL_URL,
        "query_rewrite_model": "llama3.2",
    }
    values.update(overrides)
    return _rewrite_settings(**values)


def _openai_client(content):
    """A mocked AsyncOpenAI client whose chat reply carries this content."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _anthropic_client(text):
    """A mocked AsyncAnthropic client whose reply carries this text."""
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text=text)])
    )
    return client


async def _rewrite_with_reply(provider: str, reply, question: str = "q") -> list[str]:
    """Run rewrite_query with the given provider answering with reply."""
    from stratakb.rag.query_rewriter import rewrite_query

    if provider == "openai_compatible":
        settings = _compatible_settings()
        target, client = "openai.AsyncOpenAI", _openai_client(reply)
    else:
        settings = _rewrite_settings()
        target, client = "anthropic.AsyncAnthropic", _anthropic_client(reply)
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch(target, return_value=client),
    ):
        return await rewrite_query(question, anthropic_api_key="fake-key")


class TestOpenAICompatibleRewriter:
    @pytest.mark.asyncio
    async def test_openai_compatible_rewrite(self, no_sdk_env):
        from stratakb.rag.query_rewriter import build_domain_context, rewrite_query

        client = _openai_client('["payment rejection codes", "pacs.002 reject"]')
        context = build_domain_context({"sepa": "SEPA Payments: rulebooks"})
        with (
            patch("stratakb.config.get_settings", return_value=_compatible_settings()),
            patch("openai.AsyncOpenAI", return_value=client) as openai_cls,
            patch("anthropic.AsyncAnthropic") as anthropic_cls,
        ):
            result = await rewrite_query("Why was it rejected?", domain_context=context)

        assert result == ["payment rejection codes", "pacs.002 reject"]
        openai_cls.assert_called_once()
        kwargs = openai_cls.call_args.kwargs
        assert kwargs["base_url"] == _LOCAL_URL
        assert kwargs["api_key"] == ""
        assert kwargs["max_retries"] == 0
        anthropic_cls.assert_not_called()

        sent = client.chat.completions.create.call_args.kwargs
        assert sent["model"] == "llama3.2"
        system, user = sent["messages"]
        assert system["role"] == "system"
        assert "**sepa**: SEPA Payments: rulebooks" in system["content"]
        assert user == {"role": "user", "content": "Why was it rejected?"}

    @pytest.mark.asyncio
    async def test_openai_compatible_ignores_openai_env(self, monkeypatch):
        from stratakb.rag.query_rewriter import rewrite_query

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        client = _openai_client('["a", "b"]')

        with (
            patch("stratakb.config.get_settings", return_value=_compatible_settings()),
            patch("openai.AsyncOpenAI", return_value=client) as openai_cls,
        ):
            await rewrite_query("q")
        kwargs = openai_cls.call_args.kwargs
        assert kwargs["api_key"] == ""
        assert kwargs["base_url"] == _LOCAL_URL

        settings = _compatible_settings(query_rewrite_api_key="endpoint-key")
        with (
            patch("stratakb.config.get_settings", return_value=settings),
            patch("openai.AsyncOpenAI", return_value=client) as openai_cls,
        ):
            await rewrite_query("q")
        assert openai_cls.call_args.kwargs["api_key"] == "endpoint-key"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "overrides",
        [
            {"query_rewrite_base_url": ""},
            {"query_rewrite_model": "claude-haiku-5-5"},
            {"query_rewrite_provider": "ollama"},
        ],
        ids=["no_base_url", "default_model", "unknown_provider"],
    )
    async def test_incomplete_settings_skip(self, no_sdk_env, overrides):
        from stratakb.rag.query_rewriter import rewrite_configured, rewrite_query

        settings = _compatible_settings(**overrides)
        with (
            patch("stratakb.config.get_settings", return_value=settings),
            patch("openai.AsyncOpenAI") as openai_cls,
            patch("anthropic.AsyncAnthropic") as anthropic_cls,
        ):
            result = await rewrite_query("q", anthropic_api_key="fake-key")
        assert result == ["q"]
        openai_cls.assert_not_called()
        anthropic_cls.assert_not_called()
        assert rewrite_configured(settings, anthropic_api_key="fake-key") is False

    def test_rewrite_configured(self):
        from stratakb.rag.query_rewriter import rewrite_configured

        assert rewrite_configured(_rewrite_settings()) is False
        assert rewrite_configured(_rewrite_settings(anthropic_api_key="k")) is True
        assert rewrite_configured(_compatible_settings()) is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
    @pytest.mark.parametrize(
        "reply",
        [
            'Sure! ["a", "b"] hope this helps',
            '```json\n["a", "b"]\n```',
            'Here:\n```\n["a","b"]\n```\nDone',
            '[1] then ["a", " b ", ""]',
        ],
    )
    async def test_lenient_json(self, no_sdk_env, provider, reply):
        assert await _rewrite_with_reply(provider, reply) == ["a", "b"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
    @pytest.mark.parametrize("reply", ["no array here", "[]", "[1, 2]", None])
    async def test_parse_failure_is_counted(self, no_sdk_env, provider, reply):
        from structlog.testing import capture_logs

        with capture_logs() as logs:
            first = await _rewrite_with_reply(provider, reply, "secret question")
            second = await _rewrite_with_reply(provider, reply, "secret question")

        assert first == second == ["secret question"]
        errors = [e for e in logs if e["event"] == "query_rewrite_json_error"]
        assert len(errors) == 2
        assert errors[1]["failures"] == errors[0]["failures"] + 1
        assert errors[0]["provider"] == provider
        if reply:
            assert all(reply not in str(e) for e in errors)

    @pytest.mark.asyncio
    async def test_openai_compatible_empty_choices(self, no_sdk_env):
        from stratakb.rag.query_rewriter import rewrite_query

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(choices=[]))
        with (
            patch("stratakb.config.get_settings", return_value=_compatible_settings()),
            patch("openai.AsyncOpenAI", return_value=client),
        ):
            assert await rewrite_query("q") == ["q"]

    @pytest.mark.asyncio
    async def test_openai_compatible_error_fallback(self, no_sdk_env):
        from stratakb.rag.query_rewriter import rewrite_query

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("Connection error."))
        with (
            patch("stratakb.config.get_settings", return_value=_compatible_settings()),
            patch("openai.AsyncOpenAI", return_value=client),
        ):
            assert await rewrite_query("q") == ["q"]


def _capture_ask(settings):
    """Register the RAG tools on a mock server and return strata_ask."""
    from stratakb.rag.tools import register_rag_tools

    captured = {}
    mock_mcp = MagicMock()

    def capture_tool(**kwargs):
        def decorator(fn):
            captured[getattr(fn, "__wrapped__", fn).__name__] = fn
            return fn
        return decorator

    mock_mcp.tool = capture_tool
    with patch("stratakb.config.get_settings", return_value=settings):
        register_rag_tools(mock_mcp, "postgresql://test", "")
    return captured["strata_ask"]


class TestAskDomainContext:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured", [True, False])
    async def test_ask_fetches_domain_context_only_when_rewriting(
        self, no_sdk_env, configured
    ):
        base = {"learning_enabled": False, "reranking_enabled": False}
        settings = (
            _compatible_settings(**base) if configured else _rewrite_settings(**base)
        )
        ask = _capture_ask(settings)
        routing = AsyncMock(
            return_value={"descriptions": {"sepa": "SEPA Payments: rulebooks"}, "patterns": {}}
        )
        client = _openai_client('["sepa rejection", "pacs.002"]')
        with (
            patch("stratakb.config.get_settings", return_value=settings),
            patch("stratakb.knowledge.layers.get_layer_routing", routing),
            patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[])),
            patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
            patch("openai.AsyncOpenAI", return_value=client) as openai_cls,
        ):
            result = await ask(question="Why was it rejected?", layers=["sepa"])

        assert result["status"] == "ok"
        if configured:
            routing.assert_awaited_once()
            assert result["queries_used"] == ["sepa rejection", "pacs.002"]
            system = client.chat.completions.create.call_args.kwargs["messages"][0]
            assert "**sepa**: SEPA Payments: rulebooks" in system["content"]
        else:
            routing.assert_not_awaited()
            openai_cls.assert_not_called()
            assert result["queries_used"] == ["Why was it rejected?"]


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
