"""Tests for the startup embedding checks and their health report.

Tests cover:
- The model match rule: <model> or <model>:latest, exact
- A missing model degrades strata_admin_health; a listed one does not
- A server that cannot list its models is unchecked, not degraded
- A 401 adds a detail naming EMBEDDING_API_KEY (not set, or refused),
  never the key itself; other errors add none
- The check is skipped for the local backend, without EMBEDDING_BASE_URL,
  and for a remote URL with STRATA_LOCAL_ONLY on
- The client gets EMBEDDING_API_KEY only, never OPENAI_API_KEY
- A vector column of another size degrades health and names both numbers
- No column check without a database at startup
- A check that raises is stored as unchecked and logged
- Neither the URL's credentials, its port nor the key reach health or logs
- The real column matches the default EMBEDDING_DIMENSIONS, and
  strata_db_vector_stats reports the same size (database)

The HTTP client and the pool are patched, except in the database test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from stratakb.config import Settings
from stratakb.core.egress import PLACEHOLDER_API_KEY
from stratakb.knowledge import embedding_checks
from stratakb.knowledge.embedding_checks import (
    check_vector_column,
    embedding_checks_status,
    model_listed,
    reset_embedding_checks,
    run_embedding_checks,
)
from tests.helpers import call_health, call_tool

BASE_URL = "http://127.0.0.1:11434/v1"


def _settings(**overrides) -> Settings:
    values = {
        "embedding_backend": "openai",
        "embedding_base_url": BASE_URL,
        "embedding_model": "nomic-embed-text",
        "embedding_dimensions": 768,
        "strata_user": "test",
        "strata_role": "admin",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class _Listing:
    """An async iterable of model entries, like the SDK's AsyncPaginator."""

    def __init__(self, ids: list[str], error: Exception | None = None):
        self._ids = ids
        self._error = error

    async def __aiter__(self):
        if self._error is not None:
            raise self._error
        for model_id in self._ids:
            yield SimpleNamespace(id=model_id)


def _fake_client(ids: list[str], error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    client.models.list = MagicMock(return_value=_Listing(ids, error))
    client.close = AsyncMock()
    return client


def _fake_pool(size: int | None) -> MagicMock:
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=size)
    return pool


@pytest.fixture(autouse=True)
def fresh_checks():
    """Module state outlives a test: start and end each one pending."""
    reset_embedding_checks()
    yield
    reset_embedding_checks()


async def _run(settings: Settings, ids: list[str], *, column: int | None = 768, error=None):
    """Run the checks with a patched client and pool; return the client class mock."""
    client = _fake_client(ids, error)
    with (
        patch("openai.AsyncOpenAI", return_value=client) as client_cls,
        patch(
            "stratakb.knowledge.embedding_checks.get_pool",
            AsyncMock(return_value=_fake_pool(column)),
        ),
    ):
        await run_embedding_checks(settings, db_ready=True)
    return client_cls, client


# ==========================================
# Match rule
# ==========================================


def test_model_listed_matches_latest_tag():
    assert model_listed("nomic-embed-text", ["nomic-embed-text:latest"])
    assert model_listed("nomic-embed-text", ["nomic-embed-text"])


def test_model_listed_is_exact():
    assert not model_listed("nomic-embed-text", ["nomic-embed-text:v1.5"])
    assert not model_listed("nomic-embed-text", ["Nomic-Embed-Text:latest"])
    assert model_listed("nomic-embed-text:v1.5", ["nomic-embed-text:v1.5"])
    assert not model_listed("nomic-embed-text:v1.5", ["nomic-embed-text:latest"])


# ==========================================
# Model check
# ==========================================


async def test_missing_model_degrades_health():
    settings = _settings()
    await _run(settings, ["llama3.2:latest"])
    result = await call_health(settings)

    model = result["embedding"]["model"]
    assert model["status"] == "missing"
    assert model["model"] == "nomic-embed-text"
    assert model["available"] == ["llama3.2:latest"]
    assert "EMBEDDING_MODEL" in model["detail"]
    assert result["overall"] == "degraded"


async def test_latest_tag_counts_as_present():
    settings = _settings()
    client_cls, client = await _run(settings, ["nomic-embed-text:latest"])
    result = await call_health(settings)

    assert result["embedding"]["model"] == {"status": "ok", "model": "nomic-embed-text"}
    assert result["embedding"]["vector_column"]["status"] == "ok"
    assert result["overall"] == "ok"
    client.close.assert_awaited_once()


async def test_available_is_sorted_and_capped():
    ids = [f"model-{i:02d}" for i in range(30, 0, -1)]
    await _run(_settings(), ids)
    available = embedding_checks_status()["model"]["available"]
    assert available == sorted(ids)[:20]


async def test_long_id_is_cut():
    await _run(_settings(), ["x" * 10_000])
    available = embedding_checks_status()["model"]["available"]
    assert len(available) == 1
    assert len(available[0]) <= 100
    assert available[0].endswith("…")


async def test_unprintable_characters_are_dropped():
    await _run(_settings(), ["bad\nmo\x1bdel‮"])
    assert embedding_checks_status()["model"]["available"] == ["badmodel"]


async def test_non_string_ids_are_skipped():
    await _run(_settings(), [None, 5, "llama3.2"])
    model = embedding_checks_status()["model"]
    assert model["status"] == "missing"
    assert model["available"] == ["llama3.2"]


async def test_ids_that_clean_alike_appear_once():
    await _run(_settings(), ["llama3.2", "llama3.2\n"])
    assert embedding_checks_status()["model"]["available"] == ["llama3.2"]


async def test_listing_failure_is_unchecked():
    import httpx
    import openai

    error = openai.APIConnectionError(request=httpx.Request("GET", BASE_URL + "/models"))
    settings = _settings()
    with capture_logs() as logs:
        await _run(settings, [], error=error)
    result = await call_health(settings)

    model = result["embedding"]["model"]
    assert model["status"] == "unchecked"
    assert "APIConnectionError" in model["reason"]
    assert result["overall"] == "ok"
    assert any(e["event"] == "embedding_model_unchecked" for e in logs)


_REMOTE_URL = "http://models.example.com/v1"


def _unauthorized() -> Exception:
    import httpx2
    import openai

    response = httpx2.Response(401, request=httpx2.Request("GET", _REMOTE_URL + "/models"))
    return openai.AuthenticationError("Unauthorized", response=response, body=None)


async def test_401_without_key_names_embedding_api_key():
    settings = _settings(embedding_base_url=_REMOTE_URL, embedding_api_key="")
    await _run(settings, [], error=_unauthorized())
    result = await call_health(settings)

    model = result["embedding"]["model"]
    assert model["status"] == "unchecked"
    assert "EMBEDDING_API_KEY" in model["detail"]
    assert "not set" in model["detail"]
    assert result["overall"] == "ok"


async def test_401_with_key_says_the_key_was_refused():
    import json

    settings = _settings(embedding_base_url=_REMOTE_URL, embedding_api_key="sk-endpoint-secret")
    with capture_logs() as logs:
        await _run(settings, [], error=_unauthorized())
        result = await call_health(settings)

    model = result["embedding"]["model"]
    assert model["status"] == "unchecked"
    assert "EMBEDDING_API_KEY" in model["detail"]
    assert "not set" not in model["detail"]
    assert result["overall"] == "ok"
    assert "sk-endpoint-secret" not in json.dumps(result)
    assert "sk-endpoint-secret" not in repr(logs)


async def test_other_errors_carry_no_key_detail():
    import httpx
    import openai

    error = openai.APIConnectionError(request=httpx.Request("GET", _REMOTE_URL + "/models"))
    await _run(_settings(embedding_base_url=_REMOTE_URL, embedding_api_key=""), [], error=error)
    assert "detail" not in embedding_checks_status()["model"]


async def test_skipped_without_base_url():
    client_cls, _ = await _run(_settings(embedding_base_url=""), [])
    assert embedding_checks_status()["model"]["status"] == "skipped"
    client_cls.assert_not_called()


async def test_skipped_for_local_backend():
    client_cls, _ = await _run(_settings(embedding_backend="local"), [])
    status = embedding_checks_status()
    assert status["model"]["status"] == "skipped"
    # The column check still runs
    assert status["vector_column"]["status"] == "ok"
    client_cls.assert_not_called()


async def test_skipped_for_remote_url_with_local_only():
    settings = _settings(
        embedding_base_url="http://gpu-box:11434/v1", strata_local_only=True
    )
    client_cls, _ = await _run(settings, [])
    model = embedding_checks_status()["model"]
    assert model["status"] == "skipped"
    assert "STRATA_LOCAL_ONLY" in model["reason"]
    client_cls.assert_not_called()


async def test_loopback_url_with_local_only_is_checked():
    settings = _settings(strata_local_only=True)
    client_cls, _ = await _run(settings, ["nomic-embed-text"])
    assert embedding_checks_status()["model"]["status"] == "ok"
    client_cls.assert_called_once()


async def test_client_gets_embedding_key_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    client_cls, _ = await _run(
        _settings(embedding_api_key="endpoint-key", openai_api_key="sk-openai"),
        ["nomic-embed-text"],
    )
    kwargs = client_cls.call_args.kwargs
    assert kwargs["api_key"] == "endpoint-key"
    assert kwargs["base_url"] == BASE_URL
    assert kwargs["max_retries"] == 0


async def test_client_gets_placeholder_key_without_embedding_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    client_cls, _ = await _run(_settings(openai_api_key="sk-openai"), ["nomic-embed-text"])
    assert client_cls.call_args.kwargs["api_key"] == PLACEHOLDER_API_KEY


async def test_default_model_is_checked_by_resolved_name():
    # EMBEDDING_MODEL left at the local default: embed_texts sends text-embedding-3-large
    await _run(_settings(embedding_model="all-mpnet-base-v2"), ["nomic-embed-text:latest"])
    model = embedding_checks_status()["model"]
    assert model["status"] == "missing"
    assert model["model"] == "text-embedding-3-large"


# ==========================================
# Column check
# ==========================================


async def test_column_mismatch_degrades_health():
    settings = _settings()
    with capture_logs() as logs:
        await _run(settings, ["nomic-embed-text"], column=1024)
    result = await call_health(settings)

    column = result["embedding"]["vector_column"]
    assert column["status"] == "mismatch"
    assert column["column_dimensions"] == 1024
    assert column["setting_dimensions"] == 768
    for text in ("1024", "768", "EMBEDDING_DIMENSIONS"):
        assert text in column["detail"]
    assert "re_embed_all_layers" in column["detail"]
    assert "re_embed_layer()" not in column["detail"]
    assert result["overall"] == "degraded"
    assert any(e["event"] == "vector_column_mismatch" for e in logs)


async def test_column_without_fixed_size_is_unchecked():
    settings = _settings()
    await _run(settings, ["nomic-embed-text"], column=-1)
    column = embedding_checks_status()["vector_column"]
    assert column == {"status": "unchecked", "reason": "the vector column has no fixed size"}
    assert (await call_health(settings))["overall"] == "ok"


async def test_column_read_failure_is_unchecked():
    pool = MagicMock()
    pool.fetchval = AsyncMock(side_effect=RuntimeError("relation does not exist"))
    with patch(
        "stratakb.knowledge.embedding_checks.get_pool", AsyncMock(return_value=pool)
    ):
        column = await check_vector_column("postgresql://x", 768)
    assert column["status"] == "unchecked"
    assert column["reason"] == "could not read the vector column: RuntimeError"


async def test_column_unchecked_without_database():
    get_pool = AsyncMock(return_value=_fake_pool(1024))
    with (
        patch("openai.AsyncOpenAI", return_value=_fake_client(["nomic-embed-text"])),
        patch("stratakb.knowledge.embedding_checks.get_pool", get_pool),
    ):
        await run_embedding_checks(_settings(), db_ready=False)

    column = embedding_checks_status()["vector_column"]
    assert column == {"status": "unchecked", "reason": "the database was unreachable at startup"}
    get_pool.assert_not_called()


async def test_model_check_failure_is_logged():
    with (
        patch.object(
            embedding_checks, "check_embedding_model", AsyncMock(side_effect=RuntimeError("x"))
        ),
        patch.object(
            embedding_checks, "check_vector_column", AsyncMock(return_value={"status": "ok"})
        ),
        capture_logs() as logs,
    ):
        await run_embedding_checks(_settings(), db_ready=True)

    assert embedding_checks_status()["model"] == {
        "status": "unchecked",
        "reason": "could not list the server's models: RuntimeError",
    }
    assert any(e["event"] == "embedding_model_unchecked" for e in logs)


async def test_column_check_failure_is_logged():
    with (
        patch.object(
            embedding_checks, "check_embedding_model", AsyncMock(return_value={"status": "ok"})
        ),
        patch.object(
            embedding_checks, "check_vector_column", AsyncMock(side_effect=RuntimeError("x"))
        ),
        capture_logs() as logs,
    ):
        await run_embedding_checks(_settings(), db_ready=True)

    assert embedding_checks_status()["vector_column"] == {
        "status": "unchecked",
        "reason": "could not read the vector column: RuntimeError",
    }
    assert any(e["event"] == "vector_column_unchecked" for e in logs)


async def test_pending_before_the_checks_run():
    settings = _settings()
    result = await call_health(settings)
    assert result["embedding"] == {
        "model": {"status": "pending"},
        "vector_column": {"status": "pending"},
    }
    assert result["overall"] == "ok"


def test_status_is_a_copy():
    status = embedding_checks_status()
    status["model"]["status"] = "missing"
    assert embedding_checks.embedding_checks_status()["model"]["status"] == "pending"
    assert not embedding_checks.embedding_checks_degraded()


# ==========================================
# Nothing secret in the output
# ==========================================


async def test_health_hides_url_and_key():
    settings = _settings(
        embedding_base_url="http://user:pw@gpu-box:11434/v1",
        embedding_api_key="endpoint-key",
    )
    with capture_logs() as logs:
        await _run(settings, ["llama3.2:latest"], column=1024)
        result = await call_health(settings)

    assert result["embedding"]["model"]["status"] == "missing"
    for secret in ("pw", "11434", "endpoint-key"):
        assert secret not in str(result)
        assert secret not in repr(logs)


# ==========================================
# Real database
# ==========================================


async def test_real_column_matches_default():
    import asyncpg

    from stratakb.config import get_settings

    db_url = get_settings().database_url
    try:
        conn = await asyncpg.connect(db_url, timeout=5)
    except Exception:
        pytest.skip("database not reachable")
    await conn.close()

    ok = await check_vector_column(db_url, 768)
    mismatch = await check_vector_column(db_url, 1024)

    assert ok == {"status": "ok", "column_dimensions": 768, "setting_dimensions": 768}
    assert mismatch["status"] == "mismatch"
    assert mismatch["column_dimensions"] == 768


async def test_vector_stats_reports_the_column_size():
    import asyncpg
    from mcp.server.fastmcp import FastMCP

    from stratakb.config import get_settings
    from stratakb.database_access.tools import register_database_tools

    db_url = get_settings().database_url
    try:
        conn = await asyncpg.connect(db_url, timeout=5)
    except Exception:
        pytest.skip("database not reachable")
    await conn.close()

    server = FastMCP("test")
    register_database_tools(server, db_url, "")
    with patch("stratakb.config.get_settings", return_value=_settings()):
        stats = await call_tool(server, "strata_db_vector_stats")

    column = await check_vector_column(db_url, 768)
    assert stats["embedding_dimensions"] == column["column_dimensions"]
