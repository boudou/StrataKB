"""Knowledge layer MCP tool definitions.

Registers knowledge layer management tools with the MCP server.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from mcp.server.fastmcp import FastMCP

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import check_role, get_current_user, require_role
from stratakb.core.logging import log_tool_call
from stratakb.knowledge.layers import (
    DEFAULT_ANSWER_MODE,
    LayerError,
    LayerExistsError,
    LayerNotFoundError,
    LayerValidationError,
    create_layer,
    delete_layer,
    get_layer_status,
    list_layers,
    update_layer,
)
from stratakb.knowledge.parsers import csv_parser

if TYPE_CHECKING:
    from stratakb.config import Settings

logger = structlog.get_logger()


def _ingest_description() -> str:
    """The strata_ingest_documents description; a docstring cannot name the row limit."""
    limit = f"{csv_parser.MAX_ROWS:,}"
    return f"""Ingest documents into a knowledge layer, syncing with the source folder.

    Parses files, chunks text, generates embeddings, and stores in the
    vector database. Re-running on the same folder is a sync: files whose
    content is unchanged are skipped, changed files are replaced, new files
    are added, and files ingested from that folder that no longer exist are
    removed. Other folders ingested into the layer are left alone, and a
    single-file source_path never removes anything. A file that cannot be
    read or fails to parse keeps its previous chunks and is listed in
    errors with file, error and reason ("read_error": locked or no
    permission, "parse_error", or "embed_error").

    Files whose names look like secrets (.env, keys, certificates,
    credentials, e.g. client_secret.json) are always skipped and never
    opened; one ingested earlier is removed. Empty files are reported as
    no text, not as failures: skipped, or removed if ingested earlier.
    files_skipped counts the skipped files. A PDF that looks scanned
    (more than half its pages have an image and under 200 characters of
    text) is stored for the text it has and counted in files_flagged;
    it also counts as added or updated. A CSV over {limit} rows is stored
    for its first {limit} rows and also counted in files_flagged. flagged
    lists these files with file, reason ("probably_scanned" or "truncated")
    and count: how many rows or other items were left out, null when the
    parser did not count.

    Unchanged means the same file content (SHA-256). A change to the
    layer's chunk_config or to the embedding model is not detected: use
    replace_all=True after such a change.

    Supported formats: PDF, DOCX, YAML/OpenAPI, CSV, JSON, XML/XSD, Markdown, HTML,
    plain text (.txt), reStructuredText (.rst), AsciiDoc (.adoc).

    Args:
        layer_name: Target knowledge layer (must exist). Example: "runbooks"
        source_path: Absolute path to a file or directory to ingest.
                    For directories, all supported files are discovered recursively.
        replace_all: If True, delete every chunk of the layer first and
                    ingest everything again (a full rebuild).
    """


def _serialize(obj: object) -> object:
    """Make asyncpg/datetime values JSON-safe."""
    import json
    from datetime import datetime
    from uuid import UUID

    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# Who made a layer, the server folder it holds and its archived files patterns
# are shown to admins only
_ADMIN_ONLY_LAYER_FIELDS = ("owner", "root_path", "archived_patterns")


# strata_apply_preset's dry run reports layer_check "unknown" after this many seconds
_LAYER_CHECK_TIMEOUT_S = 5.0


def _caller_is_admin() -> bool:
    """True when the current caller has the admin role."""
    _, role = get_current_user()
    return check_role(role, "admin")


def _for_caller(layer: dict, is_admin: bool) -> dict:
    """Drop the admin-only layer fields when the caller is below admin."""
    if is_admin:
        return layer
    return {k: v for k, v in layer.items() if k not in _ADMIN_ONLY_LAYER_FIELDS}


def register_knowledge_tools(
    mcp: FastMCP,
    database_url: str,
    openai_api_key: str = "",
    *,
    settings: Settings | None = None,
) -> None:
    """Register all knowledge layer management tools on the MCP server.

    `settings` are the server's: ingestion and preset application check the
    embedding path against them and embed with them. Without them the tools
    read the environment's settings on each call.
    """

    def _current_settings() -> Settings:
        """The server's settings; without any, the environment's, read per call."""
        if settings is not None:
            return settings
        from stratakb.config import get_settings

        return get_settings()

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_list_knowledge_layers() -> dict:
        """List all configured knowledge layers.

        Returns all layers with their names, descriptions, document/chunk counts,
        and last ingestion dates. Use this to discover what knowledge domains
        are available in the system.
        """
        layers = await list_layers(database_url)
        is_admin = _caller_is_admin()
        return {
            "layers": [_serialize(_for_caller(layer, is_admin)) for layer in layers],
            "total": len(layers),
        }

    @mcp.tool()
    @require_role("admin")
    @log_tool_call
    async def strata_create_knowledge_layer(
        name: str,
        display_name: str,
        description: str = "",
        metadata_schema: dict | None = None,
        chunk_config: dict | None = None,
        routing_patterns: list[str] | None = None,
        answer_mode: str = DEFAULT_ANSWER_MODE,
    ) -> dict:
        """Create a new knowledge layer for one documentation domain.

        A knowledge layer groups the documents of one domain, such as a
        standard ("iso20022"), a product's docs or a team's runbooks. Once
        created, documents can be ingested into it.

        Args:
            name: Unique identifier (lowercase, alphanumeric + underscores, 3-100 chars).
                  Examples: "iso20022", "sepa", "runbooks", "product_docs"
            display_name: Human-readable name. Examples: "ISO 20022", "Ops Runbooks"
            description: What this layer contains. Also tells the query rewriter
                         which terminology the layer uses.
            metadata_schema: Optional JSON schema for layer-specific metadata fields.
                            Example: {"message_type": "string", "version": "string"}
            chunk_config: Optional chunking configuration.
                         Defaults: {"chunk_size": 1000, "chunk_overlap": 200, "splitting_strategy": "recursive"}
            routing_patterns: Optional regular expressions; a query matching any of
                              them is routed to this layer when no layers are given.
                              Example: ["\\biso\\s*20022\\b", "\\bpacs\\.\\d{3}\\b"]
            answer_mode: "grounded" (default): the client writes answers from the
                         passages. "extractive": for text that must not be
                         paraphrased, strata_ask answers with the document's own
                         sentences.
        """
        user_id, _ = get_current_user()
        try:
            layer = await create_layer(
                database_url,
                name=name,
                display_name=display_name,
                description=description,
                metadata_schema=metadata_schema,
                chunk_config=chunk_config,
                routing_patterns=routing_patterns,
                answer_mode=answer_mode,
                origin="manual",
                owner=user_id,
            )
            await write_audit_log(
                database_url,
                action="create",
                tool_name="strata_create_knowledge_layer",
                parameters={"name": name, "display_name": display_name},
                result_summary=f"Created layer '{name}'",
            )
            return {"status": "created", "layer": _serialize(layer)}
        except LayerExistsError as e:
            return {"status": "error", "error": str(e)}
        except LayerValidationError as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_get_layer_status(name: str) -> dict:
        """Get detailed status of a knowledge layer.

        Returns document count, chunk count, last ingestion date, ingestion
        history count, and configuration details.

        Args:
            name: The layer name (e.g., "iso20022", "runbooks").
        """
        try:
            status = await get_layer_status(database_url, name=name)
            return {"status": "ok", "layer": _serialize(_for_caller(status, _caller_is_admin()))}
        except LayerNotFoundError as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    @require_role("admin")
    @log_tool_call
    async def strata_delete_knowledge_layer(name: str) -> dict:
        """Delete a knowledge layer and all its associated chunks.

        WARNING: This permanently removes the layer configuration and all
        ingested document chunks from the vector store. This cannot be undone.

        Args:
            name: The layer name to delete (e.g., "iso20022", "runbooks").
        """
        try:
            result = await delete_layer(database_url, name=name)
            await write_audit_log(
                database_url,
                action="delete",
                tool_name="strata_delete_knowledge_layer",
                parameters={"name": name},
                result_summary=f"Deleted layer '{name}', {result.get('chunks_deleted', 0)} chunks removed",
            )
            return {"status": "deleted", **result}
        except LayerNotFoundError as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    @require_role("admin")
    @log_tool_call
    async def strata_set_archived_patterns(layer_name: str, patterns: list[str]) -> dict:
        """Set which of a layer's files are archived, replacing the list it holds.

        Patterns are in .gitignore style, matched against each file's path
        relative to its folder: "archive/", "*-2024.md", "!keep.md". The last
        match wins, so the order counts. Each may hold at most three wildcards.
        An empty list clears them. Citations of a matching file are labelled
        as archived from the next ask on, with no new ingest. Archived files
        are not the layer's "archived" status, which covers the whole layer.

        Args:
            layer_name: The layer to set them on (e.g., "runbooks").
            patterns: The whole new list of patterns.
        """
        try:
            layer = await update_layer(database_url, layer_name, archived_patterns=patterns)
        except (LayerNotFoundError, LayerValidationError) as e:
            return {"status": "error", "error": str(e)}
        await write_audit_log(
            database_url,
            action="set_archived_patterns",
            tool_name="strata_set_archived_patterns",
            parameters={"layer": layer_name, "patterns": patterns},
        )
        return {"status": "updated", "layer": _serialize(layer)}

    @mcp.tool(description=_ingest_description())
    @require_role("admin")
    @log_tool_call
    async def strata_ingest_documents(
        layer_name: str,
        source_path: str,
        replace_all: bool = False,
    ) -> dict:
        from stratakb.knowledge.embeddings import embedding_key_error
        from stratakb.knowledge.ingestion import ingest_documents

        _settings = _current_settings()
        if err := embedding_key_error(_settings, openai_api_key):
            return {"status": "error", "error": err}

        try:
            result = await ingest_documents(
                database_url,
                openai_api_key,
                layer_name=layer_name,
                source_path=source_path,
                replace_all=replace_all,
                settings=_settings,
            )
            await write_audit_log(
                database_url,
                action="ingest",
                tool_name="strata_ingest_documents",
                parameters={
                    "layer_name": layer_name,
                    "source_path": source_path,
                    "replace_all": replace_all,
                },
                result_summary=(
                    f"Ingested into '{layer_name}': {result.files_added} added, "
                    f"{result.files_updated} updated, {result.files_unchanged} unchanged, "
                    f"{result.files_removed} removed, {result.chunks_created} chunks"
                ),
            )
            return {
                "status": result.status,
                "layer_name": result.layer_name,
                "docs_processed": result.docs_processed,
                "docs_failed": result.docs_failed,
                "chunks_created": result.chunks_created,
                "files_added": result.files_added,
                "files_updated": result.files_updated,
                "files_unchanged": result.files_unchanged,
                "files_removed": result.files_removed,
                "files_skipped": result.files_skipped,
                "files_flagged": result.files_flagged,
                "rebuilt_legacy": result.rebuilt_legacy,
                "duration_seconds": round(result.duration_seconds, 1),
                "errors": result.errors if result.errors else None,
                "flagged": [
                    {"file": f.relative_path, "reason": f.reason, "count": f.count}
                    for f in result.files
                    if f.outcome == "flagged"
                ] or None,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    @require_role("admin")
    @log_tool_call
    async def strata_apply_preset(
        preset_path: str,
        docs_root: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Create or update a knowledge layer from a preset file and ingest all its sources.

        A preset is a YAML file describing one layer (name, description,
        chunking, routing patterns, archived file patterns) and the folders to
        ingest into it. This tool:
        1. Creates the layer, or updates an existing layer made by this preset
           to the preset's settings; a layer with the same name made another
           way is refused and left unchanged
        2. Deletes the layer's existing chunks
        3. Ingests every source folder listed in the preset

        The result lists the layer settings that changed (layer_changes).
        Source folders that are missing are skipped and reported together with
        their download link, if the preset gives one. Use dry_run=True to check
        that all files parse, without embeddings; it also reports which layer
        settings would change (layer_check). layer_check is 'would_claim'
        when a layer with no recorded origin already matches the preset and
        would be marked as made by it, and 'unknown' when the database does
        not answer within 5 seconds.

        Args:
            preset_path: Path to the preset YAML file. Example: "examples/sepa/preset.yaml"
            docs_root: Directory the preset's source paths are relative to.
                       Defaults to the preset's own folder.
            dry_run: If True, only parse files and report stats.
        """
        from pathlib import Path

        from stratakb.knowledge.embeddings import embedding_key_error
        from stratakb.knowledge.layers import LayerValidationError
        from stratakb.knowledge.presets import (
            check_preset_layer,
            create_preset_layer,
            dry_run_parse,
            ingest_preset,
            load_preset,
        )

        _settings = _current_settings()
        try:
            preset = load_preset(preset_path)
        except FileNotFoundError:
            return {"status": "error", "error": f"Preset not found: {preset_path}"}
        except (LayerValidationError, ValueError) as e:
            return {"status": "error", "error": str(e)}

        root = Path(docs_root) if docs_root else None
        if root is not None and not root.exists():
            return {"status": "error", "error": f"Documents root not found: {docs_root}"}

        if dry_run:
            result = await asyncio.to_thread(dry_run_parse, preset, root)
            try:
                layer_check = await asyncio.wait_for(
                    check_preset_layer(database_url, preset), _LAYER_CHECK_TIMEOUT_S
                )
            except Exception as e:
                # No message: it can hold the database URL
                logger.warning("preset_layer_check_failed", error_type=type(e).__name__)
                layer_check = {"status": "unknown", "changes": []}
            return {
                "status": "dry_run_complete",
                "layer": preset.name,
                "total_files": result.total_files,
                "parsed_files": result.parsed_files,
                "failed_files": result.failed_files,
                "total_chunks": result.total_chunks,
                "group_stats": {
                    name: {
                        "files": stats["files"],
                        "parsed": stats["parsed"],
                        "failed": stats["failed"],
                        "chunks": stats["chunks"],
                    }
                    for name, stats in result.group_stats.items()
                },
                "errors": result.errors[:20] if result.errors else None,
                "layer_check": layer_check,
            }

        if err := embedding_key_error(_settings, openai_api_key):
            return {"status": "error", "error": err}

        try:
            layer_result = await create_preset_layer(database_url, preset)
            error = layer_result["message"] if layer_result["status"] == "refused" else None
        except LayerError as e:
            error = str(e)
        if error is not None:
            # Nothing was changed, so there is nothing to audit
            return {"status": "error", "error": error}
        # Written before the ingest: the patterns are stored, and it can raise
        if "archived_patterns" in layer_result["changes"] or (
            layer_result["status"] == "created" and preset.archived_patterns
        ):
            await write_audit_log(
                database_url,
                action="set_archived_patterns",
                tool_name="strata_apply_preset",
                parameters={"layer": preset.name, "patterns": preset.archived_patterns},
            )
        ingest_result = await ingest_preset(
            database_url, openai_api_key, preset, root, settings=_settings
        )

        layer_text = f"Layer '{preset.name}': {layer_result['status']}"
        if layer_result["changes"]:
            layer_text += f" ({', '.join(layer_result['changes'])})"
        await write_audit_log(
            database_url,
            action="apply_preset",
            tool_name="strata_apply_preset",
            parameters={"preset_path": preset_path, "docs_root": docs_root},
            result_summary=(
                f"{layer_text}, ingested {ingest_result['total_docs']} docs, "
                f"{ingest_result['total_chunks']} chunks"
            ),
        )

        return {
            "status": "completed",
            "layer": layer_result["status"],
            "layer_changes": layer_result["changes"],
            "ingestion": ingest_result,
        }
