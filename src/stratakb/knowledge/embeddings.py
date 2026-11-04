"""Embedding API client.

Supports two backends:
- "local": sentence-transformers (free, runs on CPU, no API key needed)
- "openai": OpenAI text-embedding-3-large (paid, higher quality)

Backend is selected via EMBEDDING_BACKEND env var (default: "local").
"""

from __future__ import annotations

import os
import threading

# Skip HuggingFace HTTP checks — model is cached locally after first download.
# Prevents HEAD requests that add latency on cold start.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import structlog

logger = structlog.get_logger()

# OpenAI batch limit
MAX_BATCH_SIZE = 100

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
    if not backend or not model or not dimensions:
        from stratakb.config import get_settings

        settings = get_settings()
        backend = backend or settings.embedding_backend
        model = model or (
            settings.embedding_model
            if backend == "local"
            else "text-embedding-3-large"
        )
        dimensions = dimensions or settings.embedding_dimensions

    if backend == "local":
        import asyncio

        # Run in thread to avoid blocking the event loop — model loading
        # can take 17s+ on first call (PyTorch import + weight loading),
        # and model.encode() is CPU-intensive on every call.
        return await asyncio.to_thread(_embed_local, texts, model)
    else:
        return await _embed_openai(texts, api_key, model, dimensions)


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
) -> list[list[float]]:
    """Embed texts using OpenAI API."""
    if not api_key:
        raise ValueError("OpenAI API key is required for openai embedding backend.")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]

        try:
            response = await client.embeddings.create(
                input=batch,
                model=model,
                dimensions=dimensions,
            )
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
    )
    return all_embeddings
