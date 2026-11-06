"""Tests for loading the embedding and reranking models.

Tests cover:
- A cached model loads from local files only, with one load
- A cache miss loads the model again with the network, so it downloads
- With STRATA_LOCAL_ONLY on, a cache miss fails and never loads online
- A folder that is missing or holds no model fails with a message naming it
- The reranker hint names the cross-encoder repository of a bare name
- With HF_HUB_OFFLINE set, a cache miss fails with a message naming it
- Importing the embeddings module leaves the Hugging Face offline flags alone

No model is needed: the sentence-transformers classes are patched.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from stratakb.config import Settings

_EMBEDDING = "all-mpnet-base-v2"
_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _settings(**overrides) -> Settings:
    values = {"strata_user": "test", "strata_role": "viewer"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def embeddings(monkeypatch):
    import stratakb.knowledge.embeddings as module

    monkeypatch.setattr(module, "_local_model", None)
    return module


@pytest.fixture
def reranker(monkeypatch):
    import stratakb.rag.reranker as module

    monkeypatch.setattr(module, "_reranker", None)
    return module


def _miss_then_load(model: object) -> MagicMock:
    """A model class whose local-files-only load misses the cache."""

    def load(name, local_files_only=False):
        if local_files_only:
            raise OSError("not in the cached files")
        return model

    return MagicMock(side_effect=load)


def test_cached_embedding_model_loads_local_files_only(embeddings):
    model = SimpleNamespace()
    model_cls = MagicMock(return_value=model)
    fake = SimpleNamespace(SentenceTransformer=model_cls)
    with (
        patch("stratakb.config.get_settings", return_value=_settings()),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
    ):
        assert embeddings._get_local_model(_EMBEDDING) is model
    model_cls.assert_called_once_with(_EMBEDDING, local_files_only=True)


def test_embedding_model_cache_miss_downloads(embeddings):
    model = SimpleNamespace()
    model_cls = _miss_then_load(model)
    fake = SimpleNamespace(SentenceTransformer=model_cls)
    with (
        patch("stratakb.config.get_settings", return_value=_settings()),
        patch("huggingface_hub.constants.is_offline_mode", return_value=False),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
    ):
        assert embeddings._get_local_model(_EMBEDDING) is model
    assert model_cls.call_args_list == [
        call(_EMBEDDING, local_files_only=True),
        call(_EMBEDDING),
    ]
    assert model._model_name == _EMBEDDING


def test_cached_reranker_loads_local_files_only(reranker):
    model = SimpleNamespace()
    model_cls = MagicMock(return_value=model)
    fake = SimpleNamespace(CrossEncoder=model_cls)
    with (
        patch("stratakb.config.get_settings", return_value=_settings()),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
    ):
        assert reranker._get_reranker(_RERANKER) is model
    model_cls.assert_called_once_with(_RERANKER, local_files_only=True)


def test_reranker_cache_miss_downloads(reranker):
    model = SimpleNamespace()
    model_cls = _miss_then_load(model)
    fake = SimpleNamespace(CrossEncoder=model_cls)
    with (
        patch("stratakb.config.get_settings", return_value=_settings()),
        patch("huggingface_hub.constants.is_offline_mode", return_value=False),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
    ):
        assert reranker._get_reranker(_RERANKER) is model
    assert model_cls.call_args_list == [
        call(_RERANKER, local_files_only=True),
        call(_RERANKER),
    ]


@pytest.mark.parametrize("attr", ["embedding", "reranker"])
def test_local_only_cache_miss_never_loads_online(embeddings, reranker, attr):
    model_cls = _miss_then_load(SimpleNamespace())
    fake = SimpleNamespace(SentenceTransformer=model_cls, CrossEncoder=model_cls)
    with (
        patch("stratakb.config.get_settings", return_value=_settings(strata_local_only=True)),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
        pytest.raises(RuntimeError, match="STRATA_LOCAL_ONLY"),
    ):
        if attr == "embedding":
            embeddings._get_local_model(_EMBEDDING)
        else:
            reranker._get_reranker(_RERANKER)
    model_cls.assert_called_once()
    assert model_cls.call_args.kwargs == {"local_files_only": True}


@pytest.mark.parametrize("folder", [r"C:\no\such\models\mpnet", "/no/such/models/mpnet"])
@pytest.mark.parametrize("local_only", [False, True])
def test_missing_folder_names_the_folder(embeddings, folder, local_only):
    model_cls = _miss_then_load(SimpleNamespace())
    fake = SimpleNamespace(SentenceTransformer=model_cls)
    with (
        patch(
            "stratakb.config.get_settings",
            return_value=_settings(strata_local_only=local_only),
        ),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
        pytest.raises(RuntimeError) as raised,
    ):
        embeddings._get_local_model(folder)
    message = str(raised.value)
    assert folder in message
    assert "EMBEDDING_MODEL" in message
    assert "does not exist" in message
    assert "hf download" not in message
    model_cls.assert_called_once_with(folder, local_files_only=True)


def test_empty_folder_holds_no_model(embeddings, tmp_path):
    model_cls = _miss_then_load(SimpleNamespace())
    fake = SimpleNamespace(SentenceTransformer=model_cls)
    with (
        patch("stratakb.config.get_settings", return_value=_settings()),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
        pytest.raises(RuntimeError, match="holds no embedding model that can be loaded"),
    ):
        embeddings._get_local_model(str(tmp_path))
    model_cls.assert_called_once()


@pytest.mark.parametrize(
    "name, repo",
    [
        ("ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        (_RERANKER, _RERANKER),
    ],
)
def test_reranker_hint_uses_cross_encoder_repo(reranker, name, repo):
    model_cls = _miss_then_load(SimpleNamespace())
    fake = SimpleNamespace(CrossEncoder=model_cls)
    with (
        patch("stratakb.config.get_settings", return_value=_settings(strata_local_only=True)),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
        pytest.raises(RuntimeError) as raised,
    ):
        reranker._get_reranker(name)
    message = str(raised.value)
    assert f"`uv run hf download {repo}`" in message
    assert message.endswith("or set RERANKING_ENABLED=false.")
    assert "RERANKING_MODEL" in message


@pytest.mark.parametrize("attr", ["embedding", "reranker"])
def test_offline_flag_cache_miss_names_the_variable(embeddings, reranker, attr):
    model_cls = _miss_then_load(SimpleNamespace())
    fake = SimpleNamespace(SentenceTransformer=model_cls, CrossEncoder=model_cls)
    with (
        patch("stratakb.config.get_settings", return_value=_settings()),
        patch("huggingface_hub.constants.is_offline_mode", return_value=True),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
        pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"),
    ):
        if attr == "embedding":
            embeddings._get_local_model(_EMBEDDING)
        else:
            reranker._get_reranker(_RERANKER)
    model_cls.assert_called_once()
    assert model_cls.call_args.kwargs == {"local_files_only": True}


def test_cached_load_builds_no_message(embeddings, reranker):
    model_cls = MagicMock(return_value=SimpleNamespace())
    fake = SimpleNamespace(SentenceTransformer=model_cls, CrossEncoder=model_cls)
    unexpected = MagicMock(side_effect=AssertionError("message built on success"))
    with (
        patch("stratakb.config.get_settings", return_value=_settings()),
        patch.object(embeddings, "_missing_model_error", unexpected),
        patch.object(embeddings, "_offline_error", unexpected),
        patch.dict(sys.modules, {"sentence_transformers": fake}),
    ):
        embeddings._get_local_model(_EMBEDDING)
        reranker._get_reranker(_RERANKER)
    unexpected.assert_not_called()


def test_import_leaves_offline_flags_alone():
    # A fresh interpreter: huggingface_hub reads the flags once, at import
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    code = (
        "import os, stratakb.knowledge.embeddings; "
        "print(os.environ.get('HF_HUB_OFFLINE'), os.environ.get('TRANSFORMERS_OFFLINE'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["None", "None"]
