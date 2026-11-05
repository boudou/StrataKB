"""Tests for the headers the OpenAI SDK sends to a configured base URL.

Tests cover:
- Query rewriting, embeddings and the model check never send OPENAI_ORG_ID
  or OPENAI_PROJECT_ID to QUERY_REWRITE_BASE_URL or EMBEDDING_BASE_URL
- They do send the endpoint's own key, even with OPENAI_ADMIN_KEY set
- With no key configured, the client still builds, holds no OPENAI_ADMIN_KEY,
  and its requests carry no Authorization header: not OPENAI_API_KEY,
  OPENAI_ADMIN_KEY or an Authorization line in OPENAI_CUSTOM_HEADERS
- A 401 from a keyless rewrite endpoint logs a hint naming QUERY_REWRITE_API_KEY
- Embeddings sent to OpenAI itself keep the SDK's organization and project

The real SDK client runs over a mock transport; nothing leaves the machine.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx2
import openai
import pytest

from stratakb.config import Settings
from stratakb.core.egress import PLACEHOLDER_API_KEY

BASE_URL = "http://127.0.0.1:11434/v1"

_RealAsyncOpenAI = openai.AsyncOpenAI


@pytest.fixture(autouse=True)
def _sdk_env(monkeypatch):
    """Set the SDK's organization and project, and keep other overrides out."""
    monkeypatch.setenv("OPENAI_ORG_ID", "org-env")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-env")
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_ADMIN_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CUSTOM_HEADERS",
        "ANTHROPIC_BASE_URL",
        "QUERY_REWRITE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides) -> Settings:
    values = {"strata_user": "test", "strata_role": "admin"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _embedding_settings(**overrides) -> Settings:
    values = {
        "embedding_backend": "openai",
        "embedding_base_url": BASE_URL,
        "embedding_api_key": "endpoint-key",
        "embedding_model": "nomic-embed-text",
        "embedding_dimensions": 768,
    }
    values.update(overrides)
    return _settings(**values)


def _reply(request: httpx2.Request) -> httpx2.Response:
    path = request.url.path
    if path.endswith("/chat/completions"):
        body = {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "llama3.2",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '["a", "b"]'},
                }
            ],
        }
    elif path.endswith("/embeddings"):
        body = {
            "object": "list",
            "model": "nomic-embed-text",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 768}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    elif path.endswith("/models"):
        body = {
            "object": "list",
            "data": [{"id": "nomic-embed-text", "object": "model", "created": 0, "owned_by": "x"}],
        }
    else:
        return httpx2.Response(404)
    return httpx2.Response(
        200, content=json.dumps(body), headers={"content-type": "application/json"}
    )


class _Recorder:
    """Builds real SDK clients over a mock transport and keeps each request's headers."""

    def __init__(self, status: int = 200):
        self.headers: list[httpx2.Headers] = []
        self.status = status

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.headers.append(request.headers)
        if self.status != 200:
            return httpx2.Response(
                self.status,
                content=json.dumps({"error": {"message": "unauthorized"}}),
                headers={"content-type": "application/json"},
            )
        return _reply(request)

    def __call__(self, **kwargs):
        transport = httpx2.MockTransport(self._handle)
        return _RealAsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(transport=transport))


@pytest.fixture
def recorder():
    rec = _Recorder()
    with patch("openai.AsyncOpenAI", side_effect=rec):
        yield rec


def _assert_no_sdk_env_headers(rec: _Recorder) -> None:
    assert rec.headers, "no request was sent"
    for headers in rec.headers:
        assert "openai-organization" not in headers
        assert "openai-project" not in headers
        assert headers["authorization"] == "Bearer endpoint-key"


async def test_rewrite_endpoint_gets_no_org_or_project(recorder, monkeypatch):
    from stratakb.rag.query_rewriter import rewrite_query

    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-env")
    settings = _settings(
        query_rewrite_provider="openai_compatible",
        query_rewrite_base_url=BASE_URL,
        query_rewrite_model="llama3.2",
        query_rewrite_api_key="endpoint-key",
    )
    with patch("stratakb.config.get_settings", return_value=settings):
        result = await rewrite_query("q")
    assert result == ["a", "b"]
    _assert_no_sdk_env_headers(recorder)


async def test_embedding_endpoint_gets_no_org_or_project(recorder, monkeypatch):
    from stratakb.knowledge.embeddings import embed_texts

    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-env")
    with patch("stratakb.config.get_settings", return_value=_embedding_settings()):
        vectors = await embed_texts(["x"])
    assert len(vectors) == 1
    _assert_no_sdk_env_headers(recorder)


async def test_model_check_gets_no_org_or_project(recorder, monkeypatch):
    from stratakb.knowledge.embedding_checks import check_embedding_model

    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-env")
    result = await check_embedding_model(_embedding_settings())
    assert result["status"] == "ok"
    _assert_no_sdk_env_headers(recorder)


async def test_openai_itself_keeps_org_and_project(recorder):
    from stratakb.knowledge.embeddings import embed_texts

    settings = _embedding_settings(embedding_base_url="", embedding_api_key="")
    with patch("stratakb.config.get_settings", return_value=settings):
        await embed_texts(["x"], api_key="sk-test")
    assert recorder.headers
    for headers in recorder.headers:
        assert headers["openai-organization"] == "org-env"
        assert headers["openai-project"] == "proj-env"
        assert headers["authorization"] == "Bearer sk-test"


def test_keyless_client_builds_without_openai_api_key(monkeypatch):
    from stratakb.core.egress import openai_compatible_client

    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-env")
    client = openai_compatible_client("", BASE_URL)
    assert client.api_key == PLACEHOLDER_API_KEY
    assert client.admin_api_key == ""
    assert str(client.base_url).startswith(BASE_URL)


_ENV_SECRETS = ("sk-env", "sk-admin-env", "sk-custom-env")


@pytest.fixture
def env_credentials(monkeypatch):
    """Put a credential in every environment variable the SDK reads one from."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-env")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer sk-custom-env")


def _assert_keyless(rec: _Recorder) -> None:
    assert rec.headers, "no request was sent"
    for headers in rec.headers:
        assert "authorization" not in headers
        assert "openai-organization" not in headers
        assert "openai-project" not in headers
        for value in headers.values():
            assert not any(secret in value for secret in _ENV_SECRETS)


def _keyless_rewrite_settings() -> Settings:
    return _settings(
        query_rewrite_provider="openai_compatible",
        query_rewrite_base_url=BASE_URL,
        query_rewrite_model="llama3.2",
        query_rewrite_api_key="",
    )


async def test_keyless_rewrite_sends_no_authorization(recorder, env_credentials):
    from stratakb.rag.query_rewriter import rewrite_query

    with patch("stratakb.config.get_settings", return_value=_keyless_rewrite_settings()):
        result = await rewrite_query("q")
    assert result == ["a", "b"]
    _assert_keyless(recorder)


async def test_keyless_embedding_sends_no_authorization(recorder, env_credentials):
    from stratakb.knowledge.embeddings import embed_texts

    settings = _embedding_settings(embedding_api_key="")
    with patch("stratakb.config.get_settings", return_value=settings):
        vectors = await embed_texts(["x"])
    assert len(vectors) == 1
    _assert_keyless(recorder)


async def test_keyless_model_check_sends_no_authorization(recorder, env_credentials):
    from stratakb.knowledge.embedding_checks import check_embedding_model

    result = await check_embedding_model(_embedding_settings(embedding_api_key=""))
    assert result["status"] == "ok"
    _assert_keyless(recorder)


async def test_rewrite_401_without_key_names_the_setting():
    from structlog.testing import capture_logs

    from stratakb.rag.query_rewriter import rewrite_query

    rec = _Recorder(status=401)
    with (
        patch("openai.AsyncOpenAI", side_effect=rec),
        patch("stratakb.config.get_settings", return_value=_keyless_rewrite_settings()),
        capture_logs() as logs,
    ):
        result = await rewrite_query("q")
    assert result == ["q"]
    _assert_keyless(rec)
    failed = [e for e in logs if e["event"] == "query_rewrite_failed"]
    assert len(failed) == 1
    assert "QUERY_REWRITE_API_KEY" in failed[0]["hint"]
    assert "not set" in failed[0]["hint"]
