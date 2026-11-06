"""Startup checks of the embedding setup, reported by strata_admin_health.

Two checks run once per start, in a background task of the server lifespan:

- the model check: a server in EMBEDDING_BASE_URL lists the embedding model
  (GET /models; "<model>:latest" counts as present);
- the column check: the strata_vectors.chunks.embedding column holds vectors
  of EMBEDDING_DIMENSIONS.

Neither the URL nor a key is ever logged or reported.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Iterable

import structlog

from stratakb.config import Settings
from stratakb.core.database import get_pool
from stratakb.core.egress import (
    keyless_request_options,
    openai_compatible_client,
    unauthorized_detail,
)
from stratakb.core.terminal_text import printable_only
from stratakb.knowledge.embeddings import embedding_local_only_error, resolve_embedding_model
from stratakb.knowledge.vector_store import vector_column_size

logger = structlog.get_logger()

_TIMEOUT_SECONDS = 5.0
_MAX_LISTED = 20
_MAX_ID_CHARS = 100

_MODELS_UNLISTED = "could not list the server's models"
_COLUMN_UNREAD = "could not read the vector column"

_checks: dict[str, dict] = {}


def _unchecked(event: str, reason: str, **fields) -> dict:
    """Log a check that could not run and return its "unchecked" result.

    The fields go into the result only; the log carries the reason, which
    never holds the URL or a key.
    """
    logger.warning(event, reason=reason)
    return {"status": "unchecked", **fields, "reason": reason}


def reset_embedding_checks() -> None:
    """Set both checks back to pending."""
    _checks.clear()
    _checks.update(model={"status": "pending"}, vector_column={"status": "pending"})


reset_embedding_checks()


def embedding_checks_status() -> dict:
    """Return the result of both checks (a copy)."""
    return copy.deepcopy(_checks)


def embedding_checks_degraded() -> bool:
    """True when the model is missing or the column size differs."""
    return (
        _checks["model"].get("status") == "missing"
        or _checks["vector_column"].get("status") == "mismatch"
    )


def model_listed(model: str, ids: list[str] | set[str]) -> bool:
    """True when `model` or `<model>:latest` is among the listed ids (exact)."""
    return model in ids or f"{model}:latest" in ids


def _shown_ids(ids: Iterable[object]) -> list[str]:
    """Return the model ids fit for the health report: printable, short, at most 20.

    The ids come from the model server and are shown to any viewer, so
    non-strings are skipped, unprintable characters dropped and long ids cut.
    """
    shown = set()
    for raw in ids:
        if not isinstance(raw, str):
            continue
        text = printable_only(raw)
        if len(text) > _MAX_ID_CHARS:
            text = text[: _MAX_ID_CHARS - 1] + "…"
        if text:
            shown.add(text)
    return sorted(shown)[:_MAX_LISTED]


async def check_embedding_model(settings: Settings) -> dict:
    """Check that the server in EMBEDDING_BASE_URL lists the embedding model."""
    if settings.embedding_backend == "local" or not settings.embedding_base_url:
        return {"status": "skipped", "reason": "only checked for an EMBEDDING_BASE_URL server"}
    if embedding_local_only_error(settings):
        return {
            "status": "skipped",
            "reason": "STRATA_LOCAL_ONLY is on and EMBEDDING_BASE_URL is not on this machine",
        }

    model = resolve_embedding_model(settings.embedding_backend, settings.embedding_model)

    key = settings.embedding_api_key
    client = openai_compatible_client(
        key,
        settings.embedding_base_url,
        timeout=_TIMEOUT_SECONDS,
        max_retries=0,
    )
    try:
        ids = {m.id async for m in client.models.list(**keyless_request_options(key))}
    except Exception as e:
        # Only the class name: an SDK error text can echo the request URL
        result = _unchecked(
            "embedding_model_unchecked",
            f"{_MODELS_UNLISTED}: {type(e).__name__}",
            model=model,
        )
        detail = unauthorized_detail(e, key, "EMBEDDING_API_KEY")
        if detail is not None:
            result["detail"] = detail
        return result
    finally:
        await client.close()

    if model_listed(model, ids):
        return {"status": "ok", "model": model}
    logger.warning("embedding_model_missing", model=model)
    return {
        "status": "missing",
        "model": model,
        "available": _shown_ids(ids),
        "detail": (
            f"The server in EMBEDDING_BASE_URL does not list the model {model} "
            "(EMBEDDING_MODEL). Pull or load it on that server, or set "
            "EMBEDDING_MODEL to a model it lists, then restart StrataKB."
        ),
    }


async def check_vector_column(database_url: str, dimensions: int) -> dict:
    """Compare the vector column's declared size with EMBEDDING_DIMENSIONS."""
    try:
        pool = await get_pool(database_url)
        size = await asyncio.wait_for(vector_column_size(pool), _TIMEOUT_SECONDS)
    except Exception as e:
        return _unchecked("vector_column_unchecked", f"{_COLUMN_UNREAD}: {type(e).__name__}")

    if size is None or size <= 0:
        return _unchecked("vector_column_unchecked", "the vector column has no fixed size")
    if size == dimensions:
        return {"status": "ok", "column_dimensions": size, "setting_dimensions": dimensions}
    logger.warning(
        "vector_column_mismatch", column_dimensions=size, setting_dimensions=dimensions
    )
    return {
        "status": "mismatch",
        "column_dimensions": size,
        "setting_dimensions": dimensions,
        "detail": (
            f"The vector column holds {size}-dimensional vectors, but "
            f"EMBEDDING_DIMENSIONS is {dimensions}. Set EMBEDDING_DIMENSIONS to "
            f"{size}, or re-embed every layer at once with "
            "vector_store.re_embed_all_layers(), which resizes the column, "
            "then restart StrataKB."
        ),
    }


async def run_embedding_checks(settings: Settings, db_ready: bool) -> None:
    """Run both checks once and store the results. Never raises but on cancel.

    Args:
        settings: The server settings.
        db_ready: Whether the lifespan's database warm-up succeeded; when not,
            the column check is not attempted (no second pool from here).
    """
    reset_embedding_checks()
    try:
        _checks["model"] = await check_embedding_model(settings)
    except Exception as e:
        _checks["model"] = _unchecked(
            "embedding_model_unchecked", f"{_MODELS_UNLISTED}: {type(e).__name__}"
        )

    if not db_ready:
        _checks["vector_column"] = _unchecked(
            "vector_column_unchecked", "the database was unreachable at startup"
        )
        return
    try:
        _checks["vector_column"] = await check_vector_column(
            settings.database_url, settings.embedding_dimensions
        )
    except Exception as e:
        _checks["vector_column"] = _unchecked(
            "vector_column_unchecked", f"{_COLUMN_UNREAD}: {type(e).__name__}"
        )
