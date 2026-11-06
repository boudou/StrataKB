"""Tests for the query rewriter.

Tests cover:
- Query rewriting (mocked), through Anthropic or an OpenAI-compatible endpoint
- Lenient parsing of the rewrite reply
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stratakb.core.egress import PLACEHOLDER_API_KEY
from tests.helpers import (
    LOCAL_REWRITE_URL,
    compatible_rewrite_settings,
    openai_reply_client,
    rewrite_settings,
)

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
        settings = compatible_rewrite_settings()
        target, client = "openai.AsyncOpenAI", openai_reply_client(reply)
    else:
        settings = rewrite_settings()
        target, client = "anthropic.AsyncAnthropic", _anthropic_client(reply)
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch(target, return_value=client),
    ):
        return await rewrite_query(question, anthropic_api_key="fake-key")


class TestOpenAICompatibleRewriter:
    def test_rewrite_provider_table(self):
        """Every provider name has one table entry, keyed by its own name."""
        import typing

        from stratakb.core.egress import RewriteProviderName
        from stratakb.rag.query_rewriter import REWRITE_PROVIDERS

        assert set(typing.get_args(RewriteProviderName)) == set(REWRITE_PROVIDERS)
        for name, p in REWRITE_PROVIDERS.items():
            assert name == p.name

    @pytest.mark.asyncio
    async def test_openai_compatible_rewrite(self, no_sdk_env):
        from stratakb.rag.query_rewriter import build_domain_context, rewrite_query

        client = openai_reply_client('["payment rejection codes", "pacs.002 reject"]')
        context = build_domain_context({"sepa": "SEPA Payments: rulebooks"})
        with (
            patch("stratakb.config.get_settings", return_value=compatible_rewrite_settings()),
            patch("openai.AsyncOpenAI", return_value=client) as openai_cls,
            patch("anthropic.AsyncAnthropic") as anthropic_cls,
        ):
            result = await rewrite_query("Why was it rejected?", domain_context=context)

        assert result == ["payment rejection codes", "pacs.002 reject"]
        openai_cls.assert_called_once()
        kwargs = openai_cls.call_args.kwargs
        assert kwargs["base_url"] == LOCAL_REWRITE_URL
        assert kwargs["api_key"] == PLACEHOLDER_API_KEY
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
        client = openai_reply_client('["a", "b"]')

        with (
            patch("stratakb.config.get_settings", return_value=compatible_rewrite_settings()),
            patch("openai.AsyncOpenAI", return_value=client) as openai_cls,
        ):
            await rewrite_query("q")
        kwargs = openai_cls.call_args.kwargs
        assert kwargs["api_key"] == PLACEHOLDER_API_KEY
        assert kwargs["base_url"] == LOCAL_REWRITE_URL

        settings = compatible_rewrite_settings(query_rewrite_api_key="endpoint-key")
        with (
            patch("stratakb.config.get_settings", return_value=settings),
            patch("openai.AsyncOpenAI", return_value=client) as openai_cls,
        ):
            await rewrite_query("q")
        assert openai_cls.call_args.kwargs["api_key"] == "endpoint-key"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_fails", [False, True], ids=["reply", "failed_request"])
    @pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
    async def test_rewrite_client_closed(self, no_sdk_env, provider, request_fails):
        """Each rewrite call closes the client it made, also when the request fails."""
        from stratakb.rag.query_rewriter import rewrite_query

        if provider == "openai_compatible":
            settings = compatible_rewrite_settings()
            client = openai_reply_client('["a", "b"]')
            create = client.chat.completions.create
        else:
            settings = rewrite_settings()
            client = _anthropic_client('["a", "b"]')
            create = client.messages.create
        if request_fails:
            create.side_effect = RuntimeError("connection refused")

        with (
            patch("stratakb.config.get_settings", return_value=settings),
            patch("openai.AsyncOpenAI", return_value=client),
            patch("anthropic.AsyncAnthropic", return_value=client),
        ):
            result = await rewrite_query("q", anthropic_api_key="fake-key")

        create.assert_awaited_once()
        client.__aexit__.assert_awaited_once()
        assert result == (["q"] if request_fails else ["a", "b"])

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

        settings = compatible_rewrite_settings(**overrides)
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

        assert rewrite_configured(rewrite_settings()) is False
        assert rewrite_configured(rewrite_settings(anthropic_api_key="k")) is True
        assert rewrite_configured(compatible_rewrite_settings()) is True

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
            patch("stratakb.config.get_settings", return_value=compatible_rewrite_settings()),
            patch("openai.AsyncOpenAI", return_value=client),
        ):
            assert await rewrite_query("q") == ["q"]

    @pytest.mark.asyncio
    async def test_openai_compatible_error_fallback(self, no_sdk_env):
        from stratakb.rag.query_rewriter import rewrite_query

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("Connection error."))
        with (
            patch("stratakb.config.get_settings", return_value=compatible_rewrite_settings()),
            patch("openai.AsyncOpenAI", return_value=client),
        ):
            assert await rewrite_query("q") == ["q"]
