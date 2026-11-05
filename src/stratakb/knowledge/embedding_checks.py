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

import structlog

from stratakb.config import Settings
from stratakb.core.database import get_pool
from stratakb.core.egress import embedding_provider, openai_compatible_client
from stratakb.knowledge.embeddings import resolve_embedding_model

logger = structlog.get_logger()

_TIMEOUT_SECONDS = 5.0
_MAX_LISTED = 20
_MAX_ID_CHARS = 100

_COLUMN_SIZE_QUERY = (
    "SELECT atttypmod FROM pg_attribute "
    "WHERE attrelid = 'strata_vectors.chunks'::regclass AND attname = 'embedding'"
)

_checks: dict[str, dict] = {}


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


def _shown_ids(ids) -> list[str]:
    """Return the model ids fit for the health report: printable, short, at most 20.

    The ids come from the model server and are shown to any viewer, so
    non-strings are skipped, unprintable characters dropped and long ids cut.
    """
    shown = set()
    for raw in ids:
        if not isinstance(raw, str):
            continue
        text = "".join(c for c in raw if c.isprintable())
        if len(text) > _MAX_ID_CHARS:
            text = text[: _MAX_ID_CHARS - 1] + "…"
        if text:
            shown.add(text)
    return sorted(shown)[:_MAX_LISTED]


async def check_embedding_model(settings: Settings) -> dict:
    """Check that the server in EMBEDDING_BASE_URL lists the embedding model."""
    if settings.embedding_backend == "local" or not settings.embedding_base_url:
        return {"status": "skipped", "reason": "only checked for an EMBEDDING_BASE_URL server"}
    if settings.strata_local_only and embedding_provider(settings) is not None:
        return {
            "status": "skipped",
            "reason": "STRATA_LOCAL_ONLY is on and EMBEDDING_BASE_URL is not on this machine",
        }

    model = resolve_embedding_model(settings.embedding_backend, settings.embedding_model)

    # Always pass a string key: with None the SDK would read OPENAI_API_KEY
    # from the environment and send it to the custom server.
    client = openai_compatible_client(
        settings.embedding_api_key,
        settings.embedding_base_url,
        timeout=_TIMEOUT_SECONDS,
        max_retries=0,
    )
    try:
        ids = {m.id async for m in client.models.list()}
    except Exception as e:
        # Only the class name: an SDK error text can echo the request URL
        logger.warning("embedding_model_unchecked", error=type(e).__name__)
        return {
            "status": "unchecked",
            "model": model,
            "reason": f"could not list the server's models: {type(e).__name__}",
        }
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
        size = await asyncio.wait_for(pool.fetchval(_COLUMN_SIZE_QUERY), _TIMEOUT_SECONDS)
    except Exception as e:
        reason = f"could not read the vector column: {type(e).__name__}"
        logger.warning("vector_column_unchecked", reason=reason)
        return {"status": "unchecked", "reason": reason}

    if size is None or size <= 0:
        reason = "the vector column has no fixed size"
        logger.warning("vector_column_unchecked", reason=reason)
        return {"status": "unchecked", "reason": reason}
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
        _checks["model"] = {
            "status": "unchecked",
            "reason": f"could not list the server's models: {type(e).__name__}",
        }

    if not db_ready:
        reason = "the database was unreachable at startup"
        logger.warning("vector_column_unchecked", reason=reason)
        _checks["vector_column"] = {"status": "unchecked", "reason": reason}
        return
    try:
        _checks["vector_column"] = await check_vector_column(
            settings.database_url, settings.embedding_dimensions
        )
    except Exception as e:
        _checks["vector_column"] = {
            "status": "unchecked",
            "reason": f"could not read the vector column: {type(e).__name__}",
        }
