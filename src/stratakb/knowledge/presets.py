"""Layer presets: a knowledge layer and its sources described in one YAML file.

A preset names the layer, its chunking and routing patterns, and the folders
(source groups) to ingest, relative to a documents root. Applying a preset
creates the layer if needed, clears its chunks and ingests every group, so a
whole documentation domain can be set up, or rebuilt, in one call.

Example (see examples/ for complete presets):

    name: sepa
    display_name: SEPA Payments
    description: SEPA rulebooks, implementation guidelines and reason codes.
    chunk_config: {chunk_size: 1500, chunk_overlap: 200}
    routing_patterns: ['\\bsepa\\b', '\\bsct\\s*inst\\b']
    sources:
      - name: rulebooks
        path: rulebooks
        description: Rulebook PDFs
        download: https://www.europeanpaymentscouncil.eu/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml

from stratakb.knowledge.ingestion import _chunk_document, _discover_files, _parse_file
from stratakb.knowledge.layers import (
    LayerExistsError,
    LayerValidationError,
    _validate_chunk_config,
    _validate_name,
    _validate_routing_patterns,
    create_layer,
    get_layer_status,
    set_origin,
)
from stratakb.knowledge.policy import is_secret_name

logger = structlog.get_logger()


@dataclass
class SourceGroup:
    """A folder of files to ingest into the preset's layer."""

    name: str
    path: str
    description: str = ""
    download: str = ""


@dataclass
class LayerPreset:
    """A knowledge layer definition plus the sources that fill it."""

    name: str
    display_name: str
    description: str = ""
    metadata_schema: dict = field(default_factory=dict)
    chunk_config: dict = field(default_factory=dict)
    routing_patterns: list[str] = field(default_factory=list)
    sources: list[SourceGroup] = field(default_factory=list)
    # Directory the source paths are relative to, unless the caller overrides it
    root: Path | None = None


def load_preset(preset_path: str | Path) -> LayerPreset:
    """Read and validate a preset YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        LayerValidationError: If the preset is malformed.
    """
    path = Path(preset_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise LayerValidationError(f"Preset {path} must be a YAML mapping.")

    for key in ("name", "display_name", "sources"):
        if not data.get(key):
            raise LayerValidationError(f"Preset {path} is missing '{key}'.")

    _validate_name(data["name"])
    chunk_config = _validate_chunk_config(data.get("chunk_config") or {})
    routing_patterns = _validate_routing_patterns(data.get("routing_patterns") or [])

    sources = []
    for raw in data["sources"]:
        if not isinstance(raw, dict) or not raw.get("name") or not raw.get("path"):
            raise LayerValidationError(f"Every source in {path} needs a 'name' and a 'path'.")
        sources.append(
            SourceGroup(
                name=raw["name"],
                path=raw["path"],
                description=raw.get("description", ""),
                download=raw.get("download", ""),
            )
        )

    root = path.parent / data["root"] if data.get("root") else path.parent

    return LayerPreset(
        name=data["name"],
        display_name=data["display_name"],
        description=data.get("description", ""),
        metadata_schema=data.get("metadata_schema") or {},
        chunk_config=chunk_config,
        routing_patterns=routing_patterns,
        sources=sources,
        root=root,
    )


# ─── Dry-run parsing ────────────────────────────────────────────────


@dataclass
class DryRunResult:
    """Result of parsing a preset's sources without a database or embeddings."""

    total_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    errors: list[dict] = field(default_factory=list)
    group_stats: dict[str, dict] = field(default_factory=dict)


def dry_run_parse(preset: LayerPreset, root: Path | None = None) -> DryRunResult:
    """Parse every source group of a preset and report file and chunk counts.

    Args:
        preset: The loaded preset.
        root: Documents root; defaults to the preset's own root.

    Returns:
        DryRunResult with file counts, chunk estimates, and any errors.
    """
    root = root or preset.root or Path(".")
    result = DryRunResult()

    for group in preset.sources:
        group_result = _parse_source_group(preset, root, group)
        result.total_files += group_result["files"]
        result.parsed_files += group_result["parsed"]
        result.failed_files += group_result["failed"]
        result.total_chunks += group_result["chunks"]
        result.errors.extend(group_result["errors"])
        result.group_stats[group.name] = group_result

    logger.info(
        "preset_dry_run_complete",
        preset=preset.name,
        total_files=result.total_files,
        parsed=result.parsed_files,
        failed=result.failed_files,
        total_chunks=result.total_chunks,
    )
    return result


def _parse_source_group(preset: LayerPreset, root: Path, group: SourceGroup) -> dict:
    """Parse a single source group and return stats."""
    stats: dict = {
        "name": group.name,
        "description": group.description,
        "files": 0,
        "parsed": 0,
        "failed": 0,
        "chunks": 0,
        "errors": [],
    }

    source_path = root / group.path
    if not source_path.exists():
        error = f"Source path not found: {source_path}"
        if group.download:
            error += f" (download from {group.download})"
        stats["errors"].append({"group": group.name, "error": error})
        return stats

    # Secret-looking names are never opened, as in a real ingest
    files = [f for f in _discover_files(source_path) if not is_secret_name(f.name)]
    stats["files"] = len(files)

    for file_path in files:
        parsed = _parse_file(file_path)
        if parsed.error:
            stats["failed"] += 1
            stats["errors"].append({
                "group": group.name,
                "file": str(file_path.name),
                "error": parsed.error,
            })
            continue

        chunks = _chunk_document(
            parsed,
            layer_name=preset.name,
            chunk_size=preset.chunk_config["chunk_size"],
            chunk_overlap=preset.chunk_config["chunk_overlap"],
        )
        stats["parsed"] += 1
        stats["chunks"] += len(chunks)

    return stats


# ─── Layer creation and ingestion ────────────────────────────────────


async def create_preset_layer(database_url: str, preset: LayerPreset) -> dict:
    """Create the preset's knowledge layer, or report that it already exists.

    An existing layer with no recorded origin (made before origins were
    recorded) is marked as coming from this preset.
    """
    origin = f"preset:{preset.name}"
    try:
        layer = await create_layer(
            database_url,
            name=preset.name,
            display_name=preset.display_name,
            description=preset.description,
            metadata_schema=preset.metadata_schema,
            chunk_config=preset.chunk_config,
            routing_patterns=preset.routing_patterns,
            origin=origin,
        )
        return {"status": "created", "layer": layer}
    except LayerExistsError:
        existing = await get_layer_status(database_url, name=preset.name)
        if existing.get("origin") is None:
            await set_origin(database_url, preset.name, origin)
        return {"status": "exists", "message": f"Layer '{preset.name}' already exists"}


async def ingest_preset(
    database_url: str,
    openai_api_key: str,
    preset: LayerPreset,
    root: Path | None = None,
) -> dict:
    """Replace the preset layer's chunks with a fresh ingestion of every source group.

    Args:
        database_url: PostgreSQL connection URL.
        openai_api_key: OpenAI API key (only needed for the openai embedding backend).
        preset: The loaded preset.
        root: Documents root; defaults to the preset's own root.

    Returns:
        Combined ingestion results across all source groups.
    """
    from stratakb.knowledge.ingestion import clear_layer, ingest_documents

    root = root or preset.root or Path(".")

    # Clear the layer once (full re-ingestion); each group then syncs from its own root
    await clear_layer(database_url, preset.name)

    results: dict = {
        "layer_name": preset.name,
        "groups": {},
        "total_docs": 0,
        "total_chunks": 0,
        "total_errors": 0,
    }

    for group in preset.sources:
        source_path = root / group.path

        if not source_path.exists():
            skipped: dict = {"status": "skipped", "error": f"Path not found: {source_path}"}
            if group.download:
                skipped["download"] = group.download
            results["groups"][group.name] = skipped
            continue

        logger.info("preset_ingesting_group", preset=preset.name, group=group.name)

        try:
            result = await ingest_documents(
                database_url,
                openai_api_key,
                layer_name=preset.name,
                source_path=str(source_path),
            )
            results["groups"][group.name] = {
                "status": result.status,
                "docs_processed": result.docs_processed,
                "docs_failed": result.docs_failed,
                "chunks_created": result.chunks_created,
                "duration_seconds": round(result.duration_seconds, 1),
            }
            results["total_docs"] += result.docs_processed
            results["total_chunks"] += result.chunks_created
            results["total_errors"] += result.docs_failed
        except Exception as e:
            results["groups"][group.name] = {"status": "error", "error": str(e)}
            results["total_errors"] += 1

    return results
