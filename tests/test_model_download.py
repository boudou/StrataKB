"""Tests for loading the embedding and reranking models.

Tests cover:
- A cached model loads from local files only, with one load
- A cache miss loads the model again with the network, so it downloads
- With STRATA_LOCAL_ONLY on, a cache miss fails and never loads online
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
