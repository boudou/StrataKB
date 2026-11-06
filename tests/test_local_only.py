"""Tests for STRATA_LOCAL_ONLY and the outbound provider report.

Tests cover:
- Which settings make a provider outbound (loopback servers are local)
- main() refuses to start with an outbound provider and names the setting
- main() forces the Hugging Face libraries offline when the switch is on
- strata_admin_health lists the switch and the outbound providers
- Embedding, model loading and query rewriting refuse or skip at call time
- A rewrite endpoint (QUERY_REWRITE_BASE_URL) is outbound unless loopback

No database and no model are needed: the pool, the server run and the
model classes are patched.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stratakb.config import Settings
from stratakb.core.egress import outbound_providers
from tests.conftest import call_tool


@pytest.fixture(autouse=True)
def _no_base_url_env(monkeypatch):
    """Keep SDK base-URL overrides from the shell out of the tests."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("QUERY_REWRITE_MODEL", raising=False)


def _settings(**overrides) -> Settings:
    values = {"strata_user": "test", "strata_role": "viewer"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


# ==========================================
# Outbound providers
# ==========================================


def test_local_only_defaults_off(monkeypatch):
    assert Settings(_env_file=None).strata_local_only is False
    monkeypatch.setenv("STRATA_LOCAL_ONLY", "true")
    assert Settings(_env_file=None).strata_local_only is True


def test_default_settings_have_no_outbound():
    assert outbound_providers(_settings()) == []


def test_anthropic_key_is_outbound():
    providers = outbound_providers(_settings(anthropic_api_key="sk-ant-test"))
    assert len(providers) == 1
    p = providers[0]
    assert p.provider == "anthropic"
    assert p.setting == "ANTHROPIC_API_KEY"
    assert p.destination == "api.anthropic.com"


def test_openai_backend_is_outbound():
    providers = outbound_providers(_settings(embedding_backend="openai"))
    assert len(providers) == 1
    p = providers[0]
    assert p.provider == "embeddings"
    assert p.destination == "api.openai.com"
    assert p.setting == "EMBEDDING_BACKEND=openai"


def test_unknown_backend_is_outbound():
    providers = outbound_providers(_settings(embedding_backend="ollama"))
    assert [p.setting for p in providers] == ["EMBEDDING_BACKEND=ollama"]


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434/v1", "http://localhost:11434/v1", "http://[::1]:11434/v1"],
)
def test_loopback_base_url_is_local(url):
    settings = _settings(embedding_backend="openai", embedding_base_url=url)
    assert outbound_providers(settings) == []


def test_remote_base_url_names_host_only():
    providers = outbound_providers(
        _settings(embedding_backend="openai", embedding_base_url="http://gpu-box:11434/v1")
    )
    assert providers[0].destination == "gpu-box"
    assert providers[0].setting == "EMBEDDING_BASE_URL"

    providers = outbound_providers(
        _settings(embedding_backend="openai", embedding_base_url="http://user:pw@10.0.0.5/v1")
    )
    p = providers[0]
    assert p.destination == "10.0.0.5"
    assert not any("pw" in str(v) for v in vars(p).values())


def test_openai_base_url_env_is_followed(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    assert outbound_providers(_settings(embedding_backend="openai")) == []


def _rewrite_endpoint(url: str, **overrides) -> Settings:
    return _settings(
        query_rewrite_provider="openai_compatible",
        query_rewrite_base_url=url,
        query_rewrite_model="llama3.2",
        **overrides,
    )


def test_remote_rewrite_endpoint_is_outbound():
    providers = outbound_providers(_rewrite_endpoint("http://gpu-box:11434/v1"))
    assert len(providers) == 1
    p = providers[0]
    assert p.provider == "openai_compatible"
    assert p.purpose == "query rewriting"
    assert p.destination == "gpu-box"
    assert p.setting == "QUERY_REWRITE_BASE_URL"


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434/v1", "http://localhost:11434/v1", "http://[::1]:11434/v1"],
)
def test_loopback_rewrite_endpoint_is_local(url):
    assert outbound_providers(_rewrite_endpoint(url)) == []


def test_rewrite_endpoint_hides_credentials():
    providers = outbound_providers(_rewrite_endpoint("http://user:pw@10.0.0.5/v1"))
    p = providers[0]
    assert p.destination == "10.0.0.5"
    assert not any("pw" in str(v) for v in vars(p).values())


def test_rewrite_endpoint_without_model_is_not_outbound():
    # The default model is an Anthropic name: rewrite_query skips, nothing is sent
    settings = _settings(
        query_rewrite_provider="openai_compatible",
        query_rewrite_base_url="http://gpu-box:11434/v1",
    )
    assert outbound_providers(settings) == []


def test_anthropic_key_unused_with_openai_compatible():
    settings = _rewrite_endpoint("http://127.0.0.1:11434/v1", anthropic_api_key="sk-ant-test")
    assert outbound_providers(settings) == []
    settings = _settings(
        query_rewrite_provider="openai_compatible", anthropic_api_key="sk-ant-test"
    )
    assert outbound_providers(settings) == []


# ==========================================
# main()
# ==========================================


@pytest.fixture
def server(monkeypatch):
    """The server module, local-only settings for main(), FastMCP.run patched.

    main() builds its server inside the call, so run is patched on the class.
    The offline flags are rolled back after the test.
    """
    from mcp.server.fastmcp import FastMCP

    from stratakb import mcp_server

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    settings = _settings(
        strata_local_only=True,
        anthropic_api_key="",
        embedding_backend="local",
        embedding_base_url="",
        query_rewrite_provider="anthropic",
        query_rewrite_base_url="",
    )
    run = MagicMock()
    monkeypatch.setattr(FastMCP, "run", run)
    return mcp_server, settings, run


def test_main_refuses_with_named_settings(server, monkeypatch):
    mcp_server, settings, run = server
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test-123")
    with pytest.raises(SystemExit) as exc:
        mcp_server.main(settings)
    text = str(exc.value)
    assert "STRATA_LOCAL_ONLY" in text
    assert "ANTHROPIC_API_KEY" in text
    run.assert_not_called()

    monkeypatch.setattr(settings, "embedding_backend", "openai")
    with pytest.raises(SystemExit) as exc:
        mcp_server.main(settings)
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert "EMBEDDING_BACKEND" in str(exc.value)
    run.assert_not_called()


def test_main_refusal_hides_key_value(server, monkeypatch):
    mcp_server, settings, _ = server
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test-123")
    with pytest.raises(SystemExit) as exc:
        mcp_server.main(settings)
    assert "sk-ant-test-123" not in str(exc.value)


def test_main_refuses_remote_rewrite_endpoint(server, monkeypatch):
    mcp_server, settings, run = server
    monkeypatch.setattr(settings, "query_rewrite_provider", "openai_compatible")
    monkeypatch.setattr(
        settings, "query_rewrite_base_url", "http://user:pw@gpu-box:11434/v1"
    )
    monkeypatch.setattr(settings, "query_rewrite_model", "llama3.2")
    with pytest.raises(SystemExit) as exc:
        mcp_server.main(settings)
    text = str(exc.value)
    assert "QUERY_REWRITE_BASE_URL" in text
    assert "gpu-box" in text
    assert "pw" not in text
    run.assert_not_called()


def test_main_starts_with_remote_rewrite_endpoint_without_model(server, monkeypatch):
    mcp_server, settings, run = server
    monkeypatch.setattr(settings, "query_rewrite_provider", "openai_compatible")
    monkeypatch.setattr(settings, "query_rewrite_base_url", "http://gpu-box:11434/v1")
    mcp_server.main(settings)
    run.assert_called_once()


def test_main_starts_with_loopback_rewrite_endpoint(server, monkeypatch):
    mcp_server, settings, run = server
    monkeypatch.setattr(settings, "query_rewrite_provider", "openai_compatible")
    monkeypatch.setattr(settings, "query_rewrite_base_url", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test-123")
    mcp_server.main(settings)
    run.assert_called_once()


def test_main_forces_hf_offline(server, monkeypatch):
    mcp_server, settings, run = server
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    mcp_server.main(settings)
    run.assert_called_once()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_main_leaves_env_when_off(server, monkeypatch):
    mcp_server, settings, run = server
    monkeypatch.setattr(settings, "strata_local_only", False)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test-123")
    mcp_server.main(settings)
    run.assert_called_once()
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


# ==========================================
# strata_admin_health
# ==========================================


async def _health(mcp_server, settings: Settings) -> dict:
    """Call strata_admin_health on a server built from `settings`."""
    return await call_tool(mcp_server.build_server(settings), "strata_admin_health")


@pytest.mark.asyncio
async def test_health_lists_outbound():
    from stratakb import mcp_server

    settings = _settings(anthropic_api_key="sk-ant-test")
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch(
            "stratakb.mcp_server.check_connection",
            AsyncMock(return_value={"status": "ok"}),
        ),
    ):
        result = await _health(mcp_server, settings)

    for key in ("database", "anthropic_api", "openai_api", "overall", "user"):
        assert key in result
    assert result["local_only"] is False
    assert result["outbound"] == [
        {
            "provider": "anthropic",
            "purpose": "query rewriting",
            "sends": "questions and layer descriptions",
            "destination": "api.anthropic.com",
            "setting": "ANTHROPIC_API_KEY",
        }
    ]
    assert "sk-ant-test" not in str(result)


@pytest.mark.asyncio
async def test_health_lists_rewrite_endpoint():
    from stratakb import mcp_server

    settings = _rewrite_endpoint(
        "http://user:pw@gpu-box:11434/v1", query_rewrite_api_key="endpoint-key"
    )
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch(
            "stratakb.mcp_server.check_connection",
            AsyncMock(return_value={"status": "ok"}),
        ),
    ):
        result = await _health(mcp_server, settings)

    assert result["outbound"] == [
        {
            "provider": "openai_compatible",
            "purpose": "query rewriting",
            "sends": "questions and layer descriptions",
            "destination": "gpu-box",
            "setting": "QUERY_REWRITE_BASE_URL",
        }
    ]
    text = str(result)
    assert "pw" not in text
    assert "endpoint-key" not in text
    assert "11434" not in text


# ==========================================
# Call-time refusals
# ==========================================


def _capture_tools(register, *args) -> dict:
    captured = {}
    mock_mcp = MagicMock()

    def capture_tool(**kwargs):
        def decorator(fn):
            name = getattr(fn, "__wrapped__", fn).__name__
            captured[name] = fn
            return fn
        return decorator

    mock_mcp.tool = capture_tool
    register(mock_mcp, *args)
    return captured


@pytest.mark.asyncio
async def test_search_refused_under_local_only():
    from stratakb.rag.tools import register_rag_tools

    fn = _capture_tools(register_rag_tools, "postgresql://test", "sk-test")["strata_search"]
    search = AsyncMock(return_value=[])
    settings = _settings(
        strata_local_only=True, embedding_backend="openai", learning_enabled=False
    )
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.rag.retriever.search_chunks", search),
    ):
        result = await fn(query="test query")
    assert result["status"] == "error"
    assert "STRATA_LOCAL_ONLY" in result["error"]
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_texts_refused_under_local_only():
    from stratakb.knowledge.embeddings import embed_texts

    settings = _settings(strata_local_only=True, embedding_backend="openai")
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("openai.AsyncOpenAI") as client_cls,
        pytest.raises(ValueError, match="STRATA_LOCAL_ONLY"),
    ):
        await embed_texts(["x"], api_key="sk-test")
    client_cls.assert_not_called()

    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1] * 768)])
    )
    settings = _settings(
        strata_local_only=True,
        embedding_backend="openai",
        embedding_base_url="http://127.0.0.1:11434/v1",
        embedding_dimensions=768,
    )
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("openai.AsyncOpenAI", return_value=client) as client_cls,
    ):
        vectors = await embed_texts(["x"], api_key="sk-test")
    client_cls.assert_called_once()
    assert len(vectors) == 1


@pytest.fixture
def local_only_settings(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    settings = _settings(strata_local_only=True)
    with patch("stratakb.config.get_settings", return_value=settings):
        yield settings


def test_local_model_loads_local_files_only(local_only_settings, monkeypatch):
    import stratakb.knowledge.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_local_model", None)
    model_cls = MagicMock(side_effect=OSError("not cached"))
    fake = SimpleNamespace(SentenceTransformer=model_cls)
    with (
        patch.dict(sys.modules, {"sentence_transformers": fake}),
        pytest.raises(RuntimeError) as exc,
    ):
        embeddings._get_local_model("all-mpnet-base-v2")
    model_cls.assert_called_once_with("all-mpnet-base-v2", local_files_only=True)
    text = str(exc.value)
    assert "STRATA_LOCAL_ONLY" in text
    assert "all-mpnet-base-v2" in text
    assert "EMBEDDING_MODEL" in text
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_reranker_loads_local_files_only(local_only_settings, monkeypatch):
    import stratakb.rag.reranker as reranker

    monkeypatch.setattr(reranker, "_reranker", None)
    model_cls = MagicMock(side_effect=OSError("not cached"))
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fake = SimpleNamespace(CrossEncoder=model_cls)
    with (
        patch.dict(sys.modules, {"sentence_transformers": fake}),
        pytest.raises(RuntimeError) as exc,
    ):
        reranker._get_reranker(name)
    model_cls.assert_called_once_with(name, local_files_only=True)
    text = str(exc.value)
    assert "STRATA_LOCAL_ONLY" in text
    assert name in text
    assert "RERANKING_MODEL" in text
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


@pytest.mark.asyncio
async def test_rewrite_skipped_under_local_only():
    from stratakb.rag.query_rewriter import rewrite_query

    with (
        patch("stratakb.config.get_settings", return_value=_settings(strata_local_only=True)),
        patch("anthropic.AsyncAnthropic") as client_cls,
    ):
        result = await rewrite_query("q", anthropic_api_key="sk-ant-test")
    assert result == ["q"]
    client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_openai_compatible_rewrite_skipped_under_local_only():
    from stratakb.rag.query_rewriter import rewrite_query

    settings = _rewrite_endpoint("http://gpu-box:11434/v1", strata_local_only=True)
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("openai.AsyncOpenAI") as client_cls,
    ):
        result = await rewrite_query("q")
    assert result == ["q"]
    client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_model_rewrite_refused_under_local_only():
    from stratakb.rag.query_rewriter import rewrite_query

    # The settings model is the default, but the caller names one: it would be sent
    settings = _settings(
        query_rewrite_provider="openai_compatible",
        query_rewrite_base_url="http://gpu-box:11434/v1",
        strata_local_only=True,
    )
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("openai.AsyncOpenAI") as client_cls,
    ):
        result = await rewrite_query("q", model="llama3.2")
    assert result == ["q"]
    client_cls.assert_not_called()

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='["a", "b"]'))]
        )
    )
    settings = _rewrite_endpoint("http://127.0.0.1:11434/v1", strata_local_only=True)
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("openai.AsyncOpenAI", return_value=client) as client_cls,
    ):
        result = await rewrite_query("q")
    assert result == ["a", "b"]
    client_cls.assert_called_once()
