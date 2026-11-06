"""Helpers shared by test modules: import them with `from tests.helpers import ...`.

Fixtures live in `tests/conftest.py`; plain functions and constants live here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from stratakb.config import Settings

FAKE_EMBEDDING = [0.01] * 768


async def call_tool(server: FastMCP, name: str, /, **arguments) -> dict:
    """Call tool `name` on a built server, through its role and logging decorators."""
    return await server._tool_manager.get_tool(name).fn(**arguments)


async def call_health(settings: Settings) -> dict:
    """Call strata_admin_health on a server built from `settings`, database reported up."""
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch(
            "stratakb.mcp_server.check_connection",
            AsyncMock(return_value={"status": "ok"}),
        ),
    ):
        return await call_tool(build_server(settings), "strata_admin_health")


def write_file(path: Path, text: str) -> Path:
    """Write text to path, creating its parent folders; return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def ingest_with_texts(db_url: str, layer: str, source: Path, **kwargs):
    """Run ingest_documents with mocked embeddings; return (result, texts embedded)."""
    from stratakb.knowledge.ingestion import ingest_documents

    embedded: list[str] = []

    def fake_embed(texts, **_):
        embedded.extend(texts)
        return [FAKE_EMBEDDING for _ in texts]

    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = fake_embed
        result = await ingest_documents(
            db_url, "fake-key", layer_name=layer, source_path=str(source), **kwargs
        )
    return result, embedded


async def ingest(db_url: str, layer: str, source: Path, **kwargs):
    """Run ingest_documents with mocked embeddings; return the result."""
    result, _ = await ingest_with_texts(db_url, layer, source, **kwargs)
    return result


def file_outcomes(result) -> dict[str, tuple[str, str | None]]:
    """Map each file's relative path to its (outcome, reason) in an ingest result."""
    return {f.relative_path: (f.outcome, f.reason) for f in result.files}


def openai_settings(**overrides) -> Settings:
    """Settings for the openai embedding backend, read from no .env file."""
    from stratakb.config import Settings

    values = {
        "embedding_backend": "openai",
        "embedding_dimensions": 768,
        "strata_user": "test",
        "strata_role": "admin",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def fake_openai_client(n: int = 768) -> MagicMock:
    """An OpenAI-compatible client whose embeddings.create returns one n-length vector."""
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1] * n)])
    )
    return client
