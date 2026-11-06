"""Knowledge layer CRUD operations.

Manages the dynamic set of knowledge layers (one per documentation domain,
e.g. "iso20022", "runbooks", "product_docs") stored in
strata_core.knowledge_layers.
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal, get_args

import asyncpg
import structlog
from cachetools import TTLCache

from stratakb.core.database import fetch_layers_stamp, get_pool
from stratakb.core.migrations import ensure_schema
from stratakb.core.stamp_watcher import StampWatcher

logger = structlog.get_logger()

# Validation constants
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,98}[a-z0-9]$")
REQUIRED_CHUNK_KEYS = {"chunk_size", "chunk_overlap"}
VALID_STATUSES = {"active", "inactive", "archived"}
# Chunk settings a layer gets when its chunk_config leaves them out
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
# How strata_ask answers for a layer (core/sql/12_answer_mode.sql)
ANSWER_MODES = ("grounded", "extractive")
DEFAULT_ANSWER_MODE = "grounded"
# Column sizes in core/sql/05_add_folder.sql
MAX_ORIGIN_LENGTH = 100
MAX_OWNER_LENGTH = 200

# Partial unique index on root_path (core/sql/05_add_folder.sql)
_ROOT_PATH_INDEX = "knowledge_layers_root_path_key"

_LAYER_COLUMNS = (
    "id, name, display_name, description, metadata_schema, chunk_config, "
    "routing_patterns, status, doc_count, chunk_count, last_ingested, "
    "created_at, updated_at, origin, owner, root_path, answer_mode"
)

# asyncpg returns jsonb as text (the pool sets no JSON codec): decoded by _decode_jsonb
_JSONB_COLUMNS = ("metadata_schema", "chunk_config", "routing_patterns")

# The only columns _write_columns puts into SQL
_LayerColumn = Literal[
    "display_name",
    "description",
    "metadata_schema",
    "chunk_config",
    "routing_patterns",
    "answer_mode",
    "origin",
    "root_path",
]
_WRITABLE_COLUMNS = frozenset(get_args(_LayerColumn))

# Search tools read routing patterns and descriptions through a short-lived
# cache. This process's create, delete and setters clear it at once; a layer
# change from another process clears it within 2 s (see _maybe_clear).
_routing_cache: TTLCache = TTLCache(maxsize=1, ttl=60)

_watcher = StampWatcher("routing_cache")
_stamps = _watcher.stamps
_checked_at = _watcher.checked_at


class LayerError(Exception):
    """Knowledge layer operation error."""


class LayerNotFoundError(LayerError):
    """Layer does not exist."""


class LayerExistsError(LayerError):
    """Layer with this name already exists."""


class LayerValidationError(LayerError):
    """Invalid layer configuration."""


class RootTakenError(LayerError):
    """Another layer already holds this folder."""


class LayerOriginError(LayerError):
    """The layer's origin is not the one a write claims; origin holds the stored one."""

    def __init__(self, message: str, origin: str | None) -> None:
        super().__init__(message)
        self.origin = origin


def _raise_if_root_taken(e: asyncpg.UniqueViolationError, root_path: object) -> None:
    """Raise RootTakenError when e is the root_path index's violation."""
    if e.constraint_name == _ROOT_PATH_INDEX:
        raise RootTakenError(f"Another layer already holds the folder '{root_path}'.") from e


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
        "chunk_size": chunk_config.get("chunk_size", DEFAULT_CHUNK_SIZE),
        "chunk_overlap": chunk_config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        "splitting_strategy": chunk_config.get("splitting_strategy", "recursive"),
    }
    if not isinstance(config["chunk_size"], int) or config["chunk_size"] < 100:
        raise LayerValidationError("chunk_size must be an integer >= 100.")
    if not isinstance(config["chunk_overlap"], int) or config["chunk_overlap"] < 0:
        raise LayerValidationError("chunk_overlap must be a non-negative integer.")
    if config["chunk_overlap"] >= config["chunk_size"]:
        raise LayerValidationError("chunk_overlap must be less than chunk_size.")
    return config


def _validate_answer_mode(answer_mode: str) -> str:
    """Validate an answer mode: one of ANSWER_MODES."""
    if answer_mode not in ANSWER_MODES:
        raise LayerValidationError(
            f"Invalid answer_mode '{answer_mode}'. Must be one of: {', '.join(ANSWER_MODES)}."
        )
    return answer_mode


def _validate_length(field: str, value: str | None, limit: int) -> None:
    """Reject a value longer than its column allows."""
    if value is not None and len(value) > limit:
        raise LayerValidationError(f"{field} must be at most {limit} characters.")


def _validate_display_name(display_name: object) -> str:
    """Validate a display name: a non-blank string. Returns it stripped."""
    if not isinstance(display_name, str) or not display_name.strip():
        raise LayerValidationError("display_name is required.")
    return display_name.strip()


def _validate_metadata_schema(metadata_schema: object) -> dict:
    """Validate a metadata schema: a JSON object."""
    if not isinstance(metadata_schema, dict):
        raise LayerValidationError("metadata_schema must be a JSON object.")
    return metadata_schema


def _decode_jsonb(value: object) -> object:
    """Decode a jsonb value asyncpg returned as text; other values as they are."""
    return json.loads(value) if isinstance(value, str) else value


def _layer_dict(row: asyncpg.Record) -> dict:
    """Return a layer row as a dict, with its jsonb columns decoded."""
    layer = dict(row)
    for column in _JSONB_COLUMNS:
        if column in layer:
            layer[column] = _decode_jsonb(layer[column])
    return layer


async def list_layers(database_url: str) -> list[dict]:
    """Return all configured knowledge layers."""
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    rows = await pool.fetch(
        f"""
        SELECT {_LAYER_COLUMNS}
        FROM strata_core.knowledge_layers
        ORDER BY name
        """
    )
    return [_layer_dict(r) for r in rows]


async def create_layer(
    database_url: str,
    *,
    name: str,
    display_name: str,
    description: str = "",
    metadata_schema: dict | None = None,
    chunk_config: dict | None = None,
    routing_patterns: list[str] | None = None,
    origin: str | None = None,
    owner: str | None = None,
    root_path: str | None = None,
    answer_mode: str = DEFAULT_ANSWER_MODE,
) -> dict:
    """Create a new knowledge layer.

    root_path is stored as given: callers pass Path(p).resolve().as_posix().

    Returns the created layer record.
    Raises LayerExistsError if name already taken.
    Raises RootTakenError if another layer already holds root_path.
    Raises LayerValidationError if inputs are invalid.
    """
    _validate_name(name)
    display = _validate_display_name(display_name)
    meta = _validate_metadata_schema(metadata_schema or {})
    chunk_cfg = _validate_chunk_config(chunk_config or {})
    routing = _validate_routing_patterns(routing_patterns or [])
    mode = _validate_answer_mode(answer_mode)
    _validate_length("origin", origin, MAX_ORIGIN_LENGTH)
    _validate_length("owner", owner, MAX_OWNER_LENGTH)

    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    try:
        row = await pool.fetchrow(
            f"""
            INSERT INTO strata_core.knowledge_layers
                (name, display_name, description, metadata_schema, chunk_config,
                 routing_patterns, origin, owner, root_path, answer_mode)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7, $8, $9, $10)
            RETURNING {_LAYER_COLUMNS}
            """,
            name,
            display,
            description or "",
            _json_str(meta),
            _json_str(chunk_cfg),
            json.dumps(routing),
            origin,
            owner,
            root_path,
            mode,
        )
    except asyncpg.UniqueViolationError as e:
        _raise_if_root_taken(e, root_path)
        raise LayerExistsError(f"Knowledge layer '{name}' already exists.")

    _routing_cache.clear()
    logger.info("layer_created", layer=name)
    return _layer_dict(row)


async def get_layer(database_url: str, *, name: str) -> dict:
    """Return one knowledge layer's row (no ingestion statistics).

    Raises LayerNotFoundError if layer doesn't exist.
    """
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        f"""
        SELECT {_LAYER_COLUMNS}
        FROM strata_core.knowledge_layers
        WHERE name = $1
        """,
        name,
    )
    if row is None:
        raise LayerNotFoundError(f"Knowledge layer '{name}' not found.")
    return _layer_dict(row)


async def get_layer_status(database_url: str, *, name: str) -> dict:
    """Get detailed status for a knowledge layer including doc/chunk counts.

    Raises LayerNotFoundError if layer doesn't exist.
    """
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        """
        SELECT kl.id, kl.name, kl.display_name, kl.description,
               kl.status, kl.doc_count, kl.chunk_count, kl.last_ingested,
               kl.chunk_config, kl.metadata_schema, kl.routing_patterns,
               kl.created_at, kl.updated_at, kl.origin, kl.owner, kl.root_path,
               kl.answer_mode,
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
    return _layer_dict(row)


async def find_layer_by_root(database_url: str, root_path: str) -> dict | None:
    """Return the layer that holds root_path, or None."""
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        f"""
        SELECT {_LAYER_COLUMNS}
        FROM strata_core.knowledge_layers
        WHERE root_path = $1
        """,
        root_path,
    )
    return _layer_dict(row) if row is not None else None


def root_key(path: str, fold: bool = False) -> str:
    """A comparison key for a folder: forward slashes, one trailing slash.

    Casefolded when fold is set (where the filesystem folds case). A
    filesystem root ("/" or "C:/") keeps its slash, so it stays a prefix of
    its children.
    """
    key = path.replace("\\", "/").rstrip("/") + "/"
    return key.casefold() if fold else key


def is_under(key: str, folder_key: str) -> bool:
    """Whether key is folder_key or lies below it.

    Both are keys from root_key, or relative paths with folder_key ending in
    "/", so the test is on whole components: /a/bc/ is not under /a/b/.
    """
    return key.startswith(folder_key)


async def find_overlapping_roots(
    database_url: str, root_path: str, *, fold: bool = False
) -> list[dict]:
    """Return the layers whose root is an ancestor or a descendant of root_path.

    Paths are compared on whole components: /a/b overlaps /a/b/c and /a, but
    not /a/bc. Case is folded when fold is set. A layer holding root_path
    itself is not included; with fold, one holding a case variant of it is.
    """
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    rows = await pool.fetch(
        f"""
        SELECT {_LAYER_COLUMNS}
        FROM strata_core.knowledge_layers
        WHERE root_path IS NOT NULL AND root_path <> $1
        ORDER BY name
        """,
        root_path,
    )
    key = root_key(root_path, fold)
    overlapping = []
    for row in rows:
        other = root_key(row["root_path"], fold)
        if is_under(key, other) or is_under(other, key):
            overlapping.append(_layer_dict(row))
    return overlapping


async def layer_holds_folder(database_url: str, layer_name: str, root_path: str) -> bool:
    """Return True when the layer holds nothing but files from root_path.

    True for a layer with no chunks and no ingested files, or for one whose
    ingested files all come from exactly root_path and that has no chunks
    without a file (an older ingestion). False otherwise.

    Raises LayerNotFoundError if the layer doesn't exist.
    """
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        """
        SELECT EXISTS (
                   SELECT 1 FROM strata_core.ingested_files f
                   WHERE f.layer_id = kl.id AND f.source_root <> $2
               ) AS other_root,
               EXISTS (
                   SELECT 1 FROM strata_vectors.chunks c
                   WHERE c.layer_name = kl.name AND c.file_id IS NULL
               ) AS legacy
        FROM strata_core.knowledge_layers kl
        WHERE kl.name = $1
        """,
        layer_name,
        root_path,
    )
    if row is None:
        raise LayerNotFoundError(f"Knowledge layer '{layer_name}' not found.")
    return not row["other_root"] and not row["legacy"]


def may_claim(stored_origin: str | None, claim_origin: str) -> bool:
    """Who may claim a layer for claim_origin: no origin recorded yet, or the same one.

    _write_columns applies the same rule in its SQL guard, so that the check
    and the write are one statement; a test keeps the two in agreement.
    """
    return stored_origin is None or stored_origin == claim_origin


async def _write_columns(
    database_url: str,
    name: str,
    values: dict[_LayerColumn, object],
    *,
    claim_origin: str | None = None,
) -> dict:
    """Write the given columns of a layer in one UPDATE and bump updated_at.

    values maps a column to its value; columns in _JSONB_COLUMNS are
    JSON-encoded and cast to jsonb. With claim_origin, the UPDATE also sets
    origin to it, and only succeeds while may_claim(stored origin, claim_origin).

    Returns the updated layer record.
    Raises ValueError if a column is not one of _WRITABLE_COLUMNS, or if values
    holds origin together with claim_origin.
    Raises LayerNotFoundError if the layer doesn't exist.
    Raises LayerOriginError if claim_origin is given and another origin is stored.
    Raises RootTakenError if another layer already holds the root_path written.
    """
    # Checked at run time too: the column names go into the SQL text
    unknown = [column for column in values if column not in _WRITABLE_COLUMNS]
    if unknown:
        raise ValueError(f"Not a writable layer column: {unknown[0]!r}")
    if claim_origin is not None and "origin" in values:
        raise ValueError("Pass origin through claim_origin only, not in values.")

    columns = dict(values)
    if claim_origin is not None:
        columns["origin"] = claim_origin
    params: list[object] = [name]
    assignments = []
    for column, value in columns.items():
        if column in _JSONB_COLUMNS:
            params.append(json.dumps(value))
            assignments.append(f"{column} = ${len(params)}::jsonb")
        else:
            params.append(value)
            assignments.append(f"{column} = ${len(params)}")
    guard = ""
    if claim_origin is not None:
        params.append(claim_origin)
        guard = f" AND (origin IS NULL OR origin = ${len(params)})"  # may_claim, in SQL

    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    try:
        row = await pool.fetchrow(
            f"""
            UPDATE strata_core.knowledge_layers
            SET {", ".join(assignments)}, updated_at = NOW()
            WHERE name = $1{guard}
            RETURNING {_LAYER_COLUMNS}
            """,
            *params,
        )
    except asyncpg.UniqueViolationError as e:
        _raise_if_root_taken(e, columns.get("root_path"))
        raise
    if row is None:
        # The guarded write changed nothing: find out why (the layer may be gone by now)
        stored = await pool.fetchrow(
            "SELECT origin FROM strata_core.knowledge_layers WHERE name = $1", name
        )
        if stored is None or claim_origin is None:
            raise LayerNotFoundError(f"Knowledge layer '{name}' not found.")
        raise LayerOriginError(
            f"Knowledge layer '{name}' has origin '{stored['origin']}', not '{claim_origin}'.",
            stored["origin"],
        )
    _routing_cache.clear()
    logger.info("layer_updated", layer=name, fields=list(columns))
    return _layer_dict(row)


async def set_root_path(database_url: str, layer_name: str, root_path: str | None) -> None:
    """Set the folder a layer holds (Path(p).resolve().as_posix(), or None).

    Raises LayerNotFoundError if the layer doesn't exist.
    Raises RootTakenError if another layer already holds root_path.
    """
    await _write_columns(database_url, layer_name, {"root_path": root_path})


def _validate_description(description: object) -> str:
    """Return description unchanged; raise LayerValidationError if it is not a string."""
    if not isinstance(description, str):
        raise LayerValidationError("description must be a string.")
    return description


async def set_description(database_url: str, layer_name: str, description: str) -> None:
    """Set a layer's description.

    Raises LayerNotFoundError if the layer doesn't exist.
    Raises LayerValidationError if description is not a string.
    """
    await _write_columns(
        database_url, layer_name, {"description": _validate_description(description)}
    )


async def set_origin(database_url: str, layer_name: str, origin: str | None) -> None:
    """Record where a layer came from ("manual", "preset:<name>").

    Raises LayerNotFoundError if the layer doesn't exist.
    Raises LayerValidationError if origin is longer than its column.
    """
    _validate_length("origin", origin, MAX_ORIGIN_LENGTH)
    await _write_columns(database_url, layer_name, {"origin": origin})


async def set_layer_answer_mode(database_url: str, layer_name: str, answer_mode: str) -> None:
    """Set how strata_ask answers for a layer ("grounded" or "extractive").

    Raises LayerNotFoundError if the layer doesn't exist.
    Raises LayerValidationError if answer_mode is not one of ANSWER_MODES.
    """
    _validate_answer_mode(answer_mode)
    await _write_columns(database_url, layer_name, {"answer_mode": answer_mode})


async def update_layer(
    database_url: str,
    name: str,
    *,
    display_name: str | None = None,
    description: str | None = None,
    metadata_schema: dict | None = None,
    chunk_config: dict | None = None,
    routing_patterns: list[str] | None = None,
    answer_mode: str | None = None,
    claim_origin: str | None = None,
) -> dict:
    """Set the given settings of a layer in one UPDATE; None leaves a setting as it is.

    With claim_origin, the same UPDATE sets the layer's origin to it, and only
    while the stored origin is NULL or already claim_origin.

    Returns the updated layer record (unchanged when nothing is given).
    Raises LayerNotFoundError if the layer doesn't exist.
    Raises LayerValidationError if a value is invalid (a non-string description
    too); nothing is written then.
    Raises LayerOriginError if another origin is stored; nothing is written then.
    """
    values: dict[_LayerColumn, object] = {}
    if display_name is not None:
        values["display_name"] = _validate_display_name(display_name)
    if description is not None:
        values["description"] = _validate_description(description)
    if metadata_schema is not None:
        values["metadata_schema"] = _validate_metadata_schema(metadata_schema)
    if chunk_config is not None:
        values["chunk_config"] = _validate_chunk_config(chunk_config)
    if routing_patterns is not None:
        values["routing_patterns"] = _validate_routing_patterns(routing_patterns)
    if answer_mode is not None:
        values["answer_mode"] = _validate_answer_mode(answer_mode)
    if claim_origin is not None:
        _validate_length("origin", claim_origin, MAX_ORIGIN_LENGTH)

    if not values and claim_origin is None:
        return await get_layer(database_url, name=name)
    return await _write_columns(database_url, name, values, claim_origin=claim_origin)


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
    await _maybe_clear(database_url)
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
        patterns = _decode_jsonb(row["routing_patterns"])
        if patterns:
            routing["patterns"][row["name"]] = patterns
        description = row["description"] or ""
        routing["descriptions"][row["name"]] = (
            f"{row['display_name']}: {description}" if description else row["display_name"]
        )

    _routing_cache["routing"] = routing
    return routing


async def _maybe_clear(database_url: str) -> None:
    """Clear the routing cache when any process has changed the layers."""
    await _watcher.check(
        database_url, _routing_cache.clear, now=time.monotonic(), fetch=fetch_layers_stamp
    )


def _json_str(obj: dict) -> str:
    """Convert dict to JSON string for asyncpg JSONB parameters."""
    return json.dumps(obj)
