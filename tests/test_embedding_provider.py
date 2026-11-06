"""Tests for the openai embedding backend with an OpenAI-compatible server.

Tests cover:
- EMBEDDING_BASE_URL and EMBEDDING_API_KEY reach the client; OPENAI_API_KEY never does
- Request shape: model, dimensions and encoding_format per endpoint
- Default model resolution for each backend
- Vectors of the wrong length are refused on both backends
- strata_search and strata_ingest_documents need no OpenAI key with a base URL
- is_model_cached needs config.json and the weights, and never downloads
- The OpenAI path refuses an empty key without a base URL, whatever the backend name
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stratakb.config import Settings
from stratakb.core.egress import PLACEHOLDER_API_KEY
from tests.helpers import fake_openai_client, openai_settings

BASE_URL = "http://127.0.0.1:11434/v1"


async def _embed(settings: Settings, api_key: str = "", n: int = 768):
    """Run embed_texts(["x"]) and return (client_kwargs, create_kwargs, vectors)."""
    from stratakb.knowledge.embeddings import embed_texts

    client = fake_openai_client(n)
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("openai.AsyncOpenAI", return_value=client) as client_cls,
    ):
        vectors = await embed_texts(["x"], api_key=api_key)
    return client_cls.call_args.kwargs, client.embeddings.create.call_args.kwargs, vectors


class TestCustomEndpoint:
    @pytest.mark.asyncio
    async def test_custom_endpoint_uses_base_url_and_embedding_key(self):
        client_kwargs, create_kwargs, vectors = await _embed(
            openai_settings(embedding_base_url=BASE_URL, embedding_model="nomic-embed-text"),
            api_key="sk-openai",
        )
        import openai

        headers = client_kwargs.pop("default_headers")
        assert client_kwargs == {
            "api_key": PLACEHOLDER_API_KEY,
            "admin_api_key": "",
            "base_url": BASE_URL,
        }
        # Dropped: the SDK would fill them from OPENAI_ORG_ID and OPENAI_PROJECT_ID
        assert set(headers) == {"OpenAI-Organization", "OpenAI-Project"}
        # No key: the request drops the Authorization header (Omit has no __eq__)
        extra_headers = create_kwargs.pop("extra_headers")
        assert set(extra_headers) == {"Authorization"}
        assert isinstance(extra_headers["Authorization"], openai.Omit)
        assert create_kwargs == {
            "input": ["x"],
            "model": "nomic-embed-text",
            "encoding_format": "float",
        }
        assert len(vectors[0]) == 768

        client_kwargs, create_kwargs, _ = await _embed(
            openai_settings(
                embedding_base_url=BASE_URL,
                embedding_model="nomic-embed-text",
                embedding_api_key="k",
            ),
            api_key="sk-openai",
        )
        assert client_kwargs["api_key"] == "k"
        assert "extra_headers" not in create_kwargs

    @pytest.mark.asyncio
    async def test_custom_endpoint_never_receives_openai_key(self):
        client_kwargs, _, _ = await _embed(
            openai_settings(
                embedding_base_url=BASE_URL,
                embedding_model="nomic-embed-text",
                openai_api_key="sk-openai",
            ),
            api_key="sk-openai",
        )
        assert client_kwargs["api_key"] == PLACEHOLDER_API_KEY

    @pytest.mark.asyncio
    async def test_openai_default_model_and_dimensions(self):
        client_kwargs, create_kwargs, _ = await _embed(openai_settings(), api_key="sk-openai")
        assert client_kwargs == {"api_key": "sk-openai", "base_url": None}
        assert create_kwargs == {
            "input": ["x"],
            "model": "text-embedding-3-large",
            "dimensions": 768,
        }

    @pytest.mark.asyncio
    async def test_text_embedding_3_on_custom_endpoint_gets_dimensions(self):
        _, create_kwargs, _ = await _embed(
            openai_settings(embedding_base_url=BASE_URL, embedding_model="text-embedding-3-small")
        )
        assert create_kwargs["dimensions"] == 768
        assert create_kwargs["encoding_format"] == "float"

    @pytest.mark.asyncio
    async def test_no_key_and_no_base_url_still_raises(self):
        from stratakb.knowledge.embeddings import embed_texts

        with (
            patch("stratakb.config.get_settings", return_value=openai_settings()),
            pytest.raises(ValueError, match="API key"),
        ):
            await embed_texts(["x"], api_key="", backend="openai")

    @pytest.mark.asyncio
    async def test_embed_openai_refuses_empty_key_without_base_url(self):
        from stratakb.knowledge.embeddings import _embed_openai

        with (
            patch("openai.AsyncOpenAI") as client_cls,
            pytest.raises(ValueError, match="API key"),
        ):
            await _embed_openai(["x"], "", "m", 768)
        client_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_misspelt_backend_without_key_is_refused(self):
        from stratakb.knowledge.embeddings import embed_texts

        with (
            patch("stratakb.config.get_settings", return_value=openai_settings()),
            patch("openai.AsyncOpenAI") as client_cls,
            pytest.raises(ValueError, match="API key"),
        ):
            await embed_texts(["x"], api_key="", backend="opnai")
        client_cls.assert_not_called()


class TestDimensionCheck:
    @pytest.mark.asyncio
    async def test_dimension_mismatch_refused_openai(self):
        with pytest.raises(ValueError) as exc:
            await _embed(
                openai_settings(embedding_base_url=BASE_URL, embedding_model="mxbai-embed-large"),
                n=1024,
            )
        message = str(exc.value)
        assert "mxbai-embed-large" in message
        assert "1024" in message
        assert "EMBEDDING_DIMENSIONS" in message

    @pytest.mark.asyncio
    async def test_dimension_mismatch_refused_local(self):
        from stratakb.knowledge.embeddings import embed_texts

        with (
            patch(
                "stratakb.config.get_settings",
                return_value=openai_settings(embedding_backend="local"),
            ),
            patch(
                "stratakb.knowledge.embeddings._embed_local",
                return_value=[[0.1] * 384],
            ),
            pytest.raises(ValueError, match="384-dimensional.*EMBEDDING_DIMENSIONS"),
        ):
            await embed_texts(["x"])


class TestResolveEmbeddingModel:
    def test_resolve_embedding_model(self):
        from stratakb.knowledge.embeddings import resolve_embedding_model

        assert resolve_embedding_model("local", "") == "all-mpnet-base-v2"
        assert resolve_embedding_model("openai", "") == "text-embedding-3-large"
        assert resolve_embedding_model("openai", "all-mpnet-base-v2") == "text-embedding-3-large"
        assert resolve_embedding_model("openai", "nomic-embed-text") == "nomic-embed-text"


def _capture_tools(register, *args) -> dict:
    mock_mcp = MagicMock()
    captured = {}

    def capture_tool():
        def decorator(fn):
            name = getattr(fn, "__name__", None) or getattr(fn, "__wrapped__", fn).__name__
            captured[name] = fn
            return fn
        return decorator

    mock_mcp.tool = capture_tool
    register(mock_mcp, *args)
    return captured


class TestToolKeyChecks:
    @pytest.mark.asyncio
    async def test_search_needs_no_openai_key_with_base_url(self):
        from stratakb.rag.tools import register_rag_tools

        fn = _capture_tools(register_rag_tools, "postgresql://test", "")["strata_search"]
        search = AsyncMock(return_value=[])
        with (
            patch(
                "stratakb.config.get_settings",
                return_value=openai_settings(embedding_base_url=BASE_URL, learning_enabled=False),
            ),
            patch("stratakb.rag.retriever.search_chunks", search),
            patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
            patch("stratakb.rag.tools._detect_layers", AsyncMock(return_value=[])),
        ):
            result = await fn(query="test query")
        assert result["status"] == "ok"
        search.assert_awaited_once()

        with (
            patch(
                "stratakb.config.get_settings",
                return_value=openai_settings(learning_enabled=False),
            ),
            patch("stratakb.rag.retriever.search_chunks", search),
        ):
            result = await fn(query="test query")
        assert result["status"] == "error"
        assert "API key" in result["error"]

    @pytest.mark.asyncio
    async def test_ingest_needs_no_openai_key_with_base_url(self):
        from stratakb.knowledge.ingestion import IngestionResult
        from stratakb.knowledge.tools import register_knowledge_tools

        fn = _capture_tools(register_knowledge_tools, "postgresql://test", "")[
            "strata_ingest_documents"
        ]
        ingest = AsyncMock(return_value=IngestionResult(layer_name="docs", status="completed"))
        with (
            patch(
                "stratakb.config.get_settings",
                return_value=openai_settings(embedding_base_url=BASE_URL),
            ),
            patch("stratakb.knowledge.ingestion.ingest_documents", ingest),
            patch("stratakb.knowledge.tools.write_audit_log", AsyncMock()),
        ):
            result = await fn(layer_name="docs", source_path="/tmp/docs")
        assert result["status"] == "completed"
        ingest.assert_awaited_once()

        with patch("stratakb.config.get_settings", return_value=openai_settings()):
            result = await fn(layer_name="docs", source_path="/tmp/docs")
        assert result["status"] == "error"
        assert "API key" in result["error"]


# ── is_model_cached ──────────────────────────────────────


def _fake_hub_cache(cache: Path, repo: str, files: list[str]) -> None:
    """Lay out a Hugging Face cache entry holding these files."""
    entry = cache / f"models--{repo.replace('/', '--')}"
    snapshot = entry / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (entry / "refs").mkdir()
    (entry / "refs" / "main").write_text("abc123")
    for name in files:
        (snapshot / name).write_text("x")


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["config.json", "model.safetensors"], True),
        (["config.json", "pytorch_model.bin"], True),
        (["config.json"], False),  # a download that stopped before the weights
        (["model.safetensors"], False),
        ([], False),
    ],
)
def test_is_model_cached_needs_config_and_weights(tmp_path, monkeypatch, files, expected):
    from stratakb.knowledge.embeddings import is_model_cached

    _fake_hub_cache(tmp_path, "sentence-transformers/tiny-model", files)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path))
    assert is_model_cached("tiny-model") is expected


def test_is_model_cached_keeps_an_owner_prefix(tmp_path, monkeypatch):
    from stratakb.knowledge.embeddings import is_model_cached

    _fake_hub_cache(tmp_path, "acme/embedder", ["config.json", "model.safetensors"])
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path))
    assert is_model_cached("acme/embedder") is True
    assert is_model_cached("embedder") is False


def test_is_model_cached_accepts_a_local_folder(tmp_path):
    from stratakb.knowledge.embeddings import is_model_cached

    assert is_model_cached(str(tmp_path)) is True


@pytest.mark.parametrize("folder", [r"C:\no\such\folder", "/no/such/folder", "./mpnet"])
def test_is_model_cached_counts_a_folder_path(folder):
    from stratakb.knowledge.embeddings import is_model_cached

    lookup = MagicMock(side_effect=AssertionError("a folder path looked up the hub cache"))
    with patch("huggingface_hub.try_to_load_from_cache", lookup):
        assert is_model_cached(folder) is True
    lookup.assert_not_called()


def test_is_model_cached_uses_the_default_cache_without_the_variable(monkeypatch):
    from stratakb.knowledge.embeddings import is_model_cached

    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    lookup = MagicMock(return_value="/cache/file")
    with patch("huggingface_hub.try_to_load_from_cache", lookup):
        assert is_model_cached("all-mpnet-base-v2") is True
    repo, filename = lookup.call_args_list[0].args
    assert repo == "sentence-transformers/all-mpnet-base-v2"
    assert filename == "config.json"
    assert lookup.call_args_list[0].kwargs == {"cache_dir": None}


def test_is_model_cached_is_false_on_any_error():
    from stratakb.knowledge.embeddings import is_model_cached

    with patch("huggingface_hub.try_to_load_from_cache", side_effect=OSError("disk")):
        assert is_model_cached("all-mpnet-base-v2") is False


def test_is_model_cached_never_downloads(tmp_path, monkeypatch):
    from stratakb.knowledge.embeddings import is_model_cached

    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert is_model_cached("not-there") is False
    assert list(tmp_path.iterdir()) == []
