"""Knowledge layer CRUD operations.

Manages the dynamic set of knowledge layers (one per documentation domain,
e.g. "iso20022", "runbooks", "product_docs") stored in
strata_core.knowledge_layers.
"""

from __future__ import annotations

import json
import re

import asyncpg
import structlog
from cachetools import TTLCache

from stratakb.core.database import get_pool

logger = structlog.get_logger()

# Validation constants
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,98}[a-z0-9]$")
REQUIRED_CHUNK_KEYS = {"chunk_size", "chunk_overlap"}
VALID_STATUSES = {"active", "inactive", "archived"}

# Routing patterns and descriptions change only when layers are created or
# deleted, so search tools read them through a short-lived cache.
_routing_cache: TTLCache = TTLCache(maxsize=1, ttl=60)


class LayerError(Exception):
    """Knowledge layer operation error."""


class LayerNotFoundError(LayerError):
    """Layer does not exist."""


class LayerExistsError(LayerError):
    """Layer with this name already exists."""


class LayerValidationError(LayerError):
    """Invalid layer configuration."""


def _validate_name(name: str) -> None:
    """Validate layer name: lowercase, alphanumeric + underscores, 3-100 chars."""
    if not name or not NAME_PATTERN.match(name):
        raise LayerValidationError(
            f"Invalid layer name '{name}'. Must be 3-100 chars, "
            "lowercase alphanumeric with underscores, starting with a letter."
        )


def _validate_routing_patterns(patterns: list[str]) -> list[str]:
    """Validate routing patterns: a list of compilable regular expressions."""
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        raise LayerValidationError("routing_patterns must be a list of strings.")
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as e:
            raise LayerValidationError(f"Invalid routing pattern '{pattern}': {e}")
    return patterns


def _validate_chunk_config(chunk_config: dict) -> dict:
    """Validate and normalize chunk configuration."""
    if not isinstance(chunk_config, dict):
        raise LayerValidationError("chunk_config must be a JSON object.")
    # Apply defaults
    config = {
        "chunk_size": chunk_config.get("chunk_size", 1000),
        "chunk_overlap": chunk_config.get("chunk_overlap", 200),
        "splitting_strategy": chunk_config.get("splitting_strategy", "recursive"),
    }
    if not isinstance(config["chunk_size"], int) or config["chunk_size"] < 100:
        raise LayerValidationError("chunk_size must be an integer >= 100.")
    if not isinstance(config["chunk_overlap"], int) or config["chunk_overlap"] < 0:
        raise LayerValidationError("chunk_overlap must be a non-negative integer.")
    if config["chunk_overlap"] >= config["chunk_size"]:
        raise LayerValidationError("chunk_overlap must be less than chunk_size.")
    return config


async def list_layers(database_url: str) -> list[dict]:
    """Return all configured knowledge layers."""
    pool = await get_pool(database_url)
    rows = await pool.fetch(
        """
        SELECT id, name, display_name, description, metadata_schema,
               chunk_config, routing_patterns, status, doc_count, chunk_count,
               last_ingested, created_at, updated_at
        FROM strata_core.knowledge_layers
        ORDER BY name
        """
    )
    return [dict(r) for r in rows]


async def create_layer(
    database_url: str,
    *,
    name: str,
    display_name: str,
    description: str = "",
    metadata_schema: dict | None = None,
    chunk_config: dict | None = None,
    routing_patterns: list[str] | None = None,
) -> dict:
    """Create a new knowledge layer.

    Returns the created layer record.
    Raises LayerExistsError if name already taken.
    Raises LayerValidationError if inputs are invalid.
    """
    _validate_name(name)

    if not display_name or not display_name.strip():
        raise LayerValidationError("display_name is required.")

    meta = metadata_schema or {}
    if not isinstance(meta, dict):
        raise LayerValidationError("metadata_schema must be a JSON object.")

    chunk_cfg = _validate_chunk_config(chunk_config or {})
    routing = _validate_routing_patterns(routing_patterns or [])

    pool = await get_pool(database_url)
    try:
        row = await pool.fetchrow(
            """
            INSERT INTO strata_core.knowledge_layers
                (name, display_name, description, metadata_schema, chunk_config,
                 routing_patterns)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb)
            RETURNING id, name, display_name, description, metadata_schema,
                      chunk_config, routing_patterns, status, doc_count, chunk_count,
                      last_ingested, created_at, updated_at
            """,
            name,
            display_name.strip(),
            description or "",
            _json_str(meta),
            _json_str(chunk_cfg),
            json.dumps(routing),
        )
    except asyncpg.UniqueViolationError:
        raise LayerExistsError(f"Knowledge layer '{name}' already exists.")

    _routing_cache.clear()
    logger.info("layer_created", layer=name)
    return dict(row)


async def get_layer_status(database_url: str, *, name: str) -> dict:
    """Get detailed status for a knowledge layer including doc/chunk counts.

    Raises LayerNotFoundError if layer doesn't exist.
    """
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        """
        SELECT kl.id, kl.name, kl.display_name, kl.description,
               kl.status, kl.doc_count, kl.chunk_count, kl.last_ingested,
               kl.chunk_config, kl.metadata_schema, kl.routing_patterns,
               kl.created_at, kl.updated_at,
               (SELECT count(*) FROM strata_core.ingestion_runs ir
                WHERE ir.layer_id = kl.id) AS ingestion_run_count,
               (SELECT max(ir.completed_at) FROM strata_core.ingestion_runs ir
                WHERE ir.layer_id = kl.id AND ir.status = 'completed')
                AS last_successful_ingestion
        FROM strata_core.knowledge_layers kl
        WHERE kl.name = $1
        """,
        name,
    )
    if row is None:
        raise LayerNotFoundError(f"Knowledge layer '{name}' not found.")
    return dict(row)


async def delete_layer(database_url: str, *, name: str) -> dict:
    """Delete a knowledge layer and all its chunks.

    Returns summary of what was deleted.
    Raises LayerNotFoundError if layer doesn't exist.
    """
    pool = await get_pool(database_url)

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Get layer first
            layer = await conn.fetchrow(
                "SELECT id, name, chunk_count FROM strata_core.knowledge_layers WHERE name = $1",
                name,
            )
            if layer is None:
                raise LayerNotFoundError(f"Knowledge layer '{name}' not found.")

            # Delete chunks from vector store
            result = await conn.execute(
                "DELETE FROM strata_vectors.chunks WHERE layer_name = $1",
                name,
            )
            # asyncpg returns "DELETE N" where N is number of rows deleted
            chunks_deleted = int(result.split()[-1])

            # Delete layer (cascades to ingestion_runs)
            await conn.execute(
                "DELETE FROM strata_core.knowledge_layers WHERE name = $1",
                name,
            )

    _routing_cache.clear()
    logger.info("layer_deleted", layer=name, chunks_deleted=chunks_deleted or 0)
    return {
        "deleted": True,
        "layer_name": name,
        "chunks_deleted": chunks_deleted or 0,
    }


async def get_layer_routing(database_url: str) -> dict:
    """Return routing patterns and descriptions for all active layers.

    Returns:
        {"patterns": {layer: [regex, ...]}, "descriptions": {layer: "Display: description"}}
    """
    cached = _routing_cache.get("routing")
    if cached is not None:
        return cached

    pool = await get_pool(database_url)
    rows = await pool.fetch(
        """
        SELECT name, display_name, description, routing_patterns
        FROM strata_core.knowledge_layers
        WHERE status = 'active'
        ORDER BY name
        """
    )
    routing: dict = {"patterns": {}, "descriptions": {}}
    for row in rows:
        patterns = row["routing_patterns"]
        if isinstance(patterns, str):
            patterns = json.loads(patterns)
        if patterns:
            routing["patterns"][row["name"]] = patterns
        description = row["description"] or ""
        routing["descriptions"][row["name"]] = (
            f"{row['display_name']}: {description}" if description else row["display_name"]
        )

    _routing_cache["routing"] = routing
    return routing


def _json_str(obj: dict) -> str:
    """Convert dict to JSON string for asyncpg JSONB parameters."""
    return json.dumps(obj)
