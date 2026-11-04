"""Embedding API client.

Supports two backends:
- "local": sentence-transformers (free, runs on CPU, no API key needed)
- "openai": the OpenAI embeddings API, either OpenAI itself
  (text-embedding-3-large unless EMBEDDING_MODEL is set) or any
  OpenAI-compatible server (Ollama, LM Studio, vLLM) in EMBEDDING_BASE_URL

Backend is selected via EMBEDDING_BACKEND env var (default: "local").
Every call refuses vectors whose length differs from EMBEDDING_DIMENSIONS.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

# Skip HuggingFace HTTP checks — model is cached locally after first download.
# Prevents HEAD requests that add latency on cold start.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import structlog

if TYPE_CHECKING:
    from stratakb.config import Settings

logger = structlog.get_logger()

# OpenAI batch limit
MAX_BATCH_SIZE = 100

# Model used by the openai backend when EMBEDDING_MODEL is unset or left at
# the local default
_OPENAI_DEFAULT_MODEL = "text-embedding-3-large"

_MISSING_KEY_ERROR = (
    "OpenAI API key not configured. Set OPENAI_API_KEY in .env, "
    "or EMBEDDING_BASE_URL for an OpenAI-compatible server."
)

# Singleton for local model (loaded once, reused)
_local_model = None
_local_model_lock = threading.Lock()


def _get_local_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the sentence-transformers model (singleton, thread-safe)."""
    global _local_model
    if _local_model is not None and _local_model._model_name == model_name:
        return _local_model
    with _local_model_lock:
        # Double-check after acquiring lock
        if _local_model is not None and _local_model._model_name == model_name:
            return _local_model
        from sentence_transformers import SentenceTransformer

        logger.info("loading_local_embedding_model", model=model_name)
        try:
            _local_model = SentenceTransformer(model_name)
        except Exception:
            # If offline mode fails (model not cached), retry with network
            os.environ.pop("HF_HUB_OFFLINE", None)
            logger.info("retrying_model_load_online", model=model_name)
            _local_model = SentenceTransformer(model_name)
        _local_model._model_name = model_name  # tag for cache check
        logger.info("local_embedding_model_loaded", model=model_name)
    return _local_model


async def embed_texts(
    texts: list[str],
    api_key: str = "",
    model: str = "",
    dimensions: int = 0,
    backend: str = "",
) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        api_key: OpenAI API key (only needed for openai backend).
        model: Model name override. Defaults to config value.
        dimensions: Output vector dimensions override. Defaults to config value.
        backend: "local" or "openai". Defaults to config value.

    Returns:
        List of embedding vectors (same order as input texts).
    """
    if not texts:
        return []

    # Resolve config defaults
    from stratakb.config import get_settings

    settings = get_settings()
    backend = backend or settings.embedding_backend
    model = model or resolve_embedding_model(backend, settings.embedding_model)
    dimensions = dimensions or settings.embedding_dimensions

    if backend == "local":
        import asyncio

        # Run in thread to avoid blocking the event loop — model loading
        # can take 17s+ on first call (PyTorch import + weight loading),
        # and model.encode() is CPU-intensive on every call.
        vectors = await asyncio.to_thread(_embed_local, texts, model)
    else:
        if err := embedding_key_error(settings, api_key, backend):
            raise ValueError(err)
        # A custom endpoint gets its own key, never the OpenAI one
        base_url = settings.embedding_base_url
        key = settings.embedding_api_key if base_url else api_key
        vectors = await _embed_openai(texts, key, model, dimensions, base_url=base_url)

    _check_dimensions(vectors, model, dimensions)
    return vectors


def embedding_key_error(
    settings: Settings, openai_api_key: str, backend: str = ""
) -> str | None:
    """Return why embedding cannot run for lack of a key, or None if it can.

    The openai backend needs OPENAI_API_KEY unless EMBEDDING_BASE_URL points
    it at an OpenAI-compatible server, whose key is optional. The local
    backend needs no key.
    """
    backend = backend or settings.embedding_backend
    if backend == "openai" and not openai_api_key and not settings.embedding_base_url:
        return _MISSING_KEY_ERROR
    return None


def resolve_embedding_model(backend: str, configured: str) -> str:
    """Return the model a backend uses for a configured EMBEDDING_MODEL.

    The openai backend treats an empty value and the local default
    (all-mpnet-base-v2) as "not set", so a .env that only flips
    EMBEDDING_BACKEND keeps working.
    """
    from stratakb.config import Settings

    local_default = Settings.model_fields["embedding_model"].default
    if backend == "local":
        return configured or local_default
    if not configured or configured == local_default:
        return _OPENAI_DEFAULT_MODEL
    return configured


def _check_dimensions(vectors: list[list[float]], model: str, dimensions: int) -> None:
    """Refuse vectors whose length differs from EMBEDDING_DIMENSIONS."""
    for vector in vectors:
        n = len(vector)
        if n != dimensions:
            raise ValueError(
                f"The embedding model {model} returns {n}-dimensional vectors, "
                f"but EMBEDDING_DIMENSIONS is {dimensions}. Set EMBEDDING_DIMENSIONS "
                f"to {n} and re-embed every layer, or choose a model with "
                f"{dimensions} dimensions."
            )


def _embed_local(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed texts using a local sentence-transformers model."""
    model = _get_local_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    logger.info(
        "local_embeddings_complete",
        total_texts=len(texts),
        model=model_name,
        dimensions=embeddings.shape[1] if hasattr(embeddings, "shape") else len(embeddings[0]),
    )
    return [emb.tolist() for emb in embeddings]


async def _embed_openai(
    texts: list[str],
    api_key: str,
    model: str,
    dimensions: int,
    base_url: str = "",
) -> list[list[float]]:
    """Embed texts using the OpenAI API or an OpenAI-compatible server."""
    from openai import AsyncOpenAI

    # Always pass a string key: with None the SDK would read OPENAI_API_KEY
    # from the environment and send it to the custom server. base_url=None
    # keeps the SDK default (OpenAI, or OPENAI_BASE_URL).
    client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
    create_kwargs: dict = {"model": model}
    # Other servers may reject "dimensions" for models that cannot shorten
    # their output; _check_dimensions catches a server that ignores it.
    if not base_url or model.startswith("text-embedding-3"):
        create_kwargs["dimensions"] = dimensions
    # The SDK asks for base64 by default, which not every server implements
    if base_url:
        create_kwargs["encoding_format"] = "float"
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]

        try:
            response = await client.embeddings.create(input=batch, **create_kwargs)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                "openai_embeddings_generated",
                batch_start=i,
                batch_size=len(batch),
                model=model,
            )
        except Exception as e:
            logger.error(
                "openai_embedding_api_error",
                batch_start=i,
                batch_size=len(batch),
                error=str(e),
            )
            raise RuntimeError(f"Embedding API error: {e}") from e

    logger.info(
        "openai_embeddings_complete",
        total_texts=len(texts),
        total_embeddings=len(all_embeddings),
        model=model,
        endpoint="custom" if base_url else "openai",
    )
    return all_embeddings
