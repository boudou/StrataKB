"""Knowledge layer MCP tool definitions.

Registers knowledge layer management tools with the MCP server.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import get_current_user, require_role
from stratakb.core.logging import log_tool_call
from stratakb.knowledge.layers import (
    LayerError,
    LayerExistsError,
    LayerNotFoundError,
    LayerValidationError,
    create_layer,
    delete_layer,
    get_layer_status,
    list_layers,
)


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


def register_knowledge_tools(
    mcp: FastMCP, database_url: str, openai_api_key: str = ""
) -> None:
    """Register all knowledge layer management tools on the MCP server."""

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
        return {
            "layers": [_serialize(layer) for layer in layers],
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
            )
            await write_audit_log(
                database_url,
                user_id=user_id,
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
            return {"status": "ok", "layer": _serialize(status)}
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
        user_id, _ = get_current_user()
        try:
            result = await delete_layer(database_url, name=name)
            await write_audit_log(
                database_url,
                user_id=user_id,
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
    async def strata_ingest_documents(
        layer_name: str,
        source_path: str,
        replace_all: bool = False,
    ) -> dict:
        """Ingest documents into a knowledge layer, syncing with the source folder.

        Parses files, chunks text, generates embeddings, and stores in the
        vector database. Re-running on the same folder is a sync: files whose
        content is unchanged are skipped, changed files are replaced, new files
        are added, and files ingested from that folder that no longer exist are
        removed. Other folders ingested into the layer are left alone, and a
        single-file source_path never removes anything. A file that fails to
        parse keeps its previous chunks and is listed in errors.

        Unchanged means the same file content (SHA-256). A change to the
        layer's chunk_config or to the embedding model is not detected: use
        replace_all=True after such a change.

        Supported formats: PDF, DOCX, YAML/OpenAPI, CSV, JSON, XML/XSD, Markdown, HTML.

        Args:
            layer_name: Target knowledge layer (must exist). Example: "runbooks"
            source_path: Absolute path to a file or directory to ingest.
                        For directories, all supported files are discovered recursively.
            replace_all: If True, delete every chunk of the layer first and
                        ingest everything again (a full rebuild).
        """
        from stratakb.config import get_settings
        from stratakb.knowledge.ingestion import ingest_documents

        _settings = get_settings()
        if _settings.embedding_backend == "openai" and not openai_api_key:
            return {
                "status": "error",
                "error": "OpenAI API key not configured. Set OPENAI_API_KEY in .env",
            }

        user_id, _ = get_current_user()
        try:
            result = await ingest_documents(
                database_url,
                openai_api_key,
                layer_name=layer_name,
                source_path=source_path,
                replace_all=replace_all,
            )
            await write_audit_log(
                database_url,
                user_id=user_id,
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
                "rebuilt_legacy": result.rebuilt_legacy,
                "duration_seconds": round(result.duration_seconds, 1),
                "errors": result.errors if result.errors else None,
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
        """Create a knowledge layer from a preset file and ingest all its sources.

        A preset is a YAML file describing one layer (name, description,
        chunking, routing patterns) and the folders to ingest into it. This
        tool:
        1. Creates the layer if it does not exist yet
        2. Deletes the layer's existing chunks
        3. Ingests every source folder listed in the preset

        Source folders that are missing are skipped and reported together with
        their download link, if the preset gives one. Use dry_run=True to check
        that all files parse, without a database or embeddings.

        Args:
            preset_path: Path to the preset YAML file. Example: "examples/sepa/preset.yaml"
            docs_root: Directory the preset's source paths are relative to.
                       Defaults to the preset's own folder.
            dry_run: If True, only parse files and report stats.
        """
        from pathlib import Path

        from stratakb.config import get_settings
        from stratakb.knowledge.layers import LayerValidationError
        from stratakb.knowledge.presets import (
            create_preset_layer,
            dry_run_parse,
            ingest_preset,
            load_preset,
        )

        _settings = get_settings()
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
            result = dry_run_parse(preset, root)
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
            }

        if _settings.embedding_backend == "openai" and not openai_api_key:
            return {
                "status": "error",
                "error": "OpenAI API key not configured. Set OPENAI_API_KEY in .env",
            }

        user_id, _ = get_current_user()
        layer_result = await create_preset_layer(database_url, preset)
        ingest_result = await ingest_preset(database_url, openai_api_key, preset, root)

        await write_audit_log(
            database_url,
            user_id=user_id,
            action="apply_preset",
            tool_name="strata_apply_preset",
            parameters={"preset_path": preset_path, "docs_root": docs_root},
            result_summary=(
                f"Layer '{preset.name}': {layer_result['status']}, "
                f"ingested {ingest_result['total_docs']} docs, "
                f"{ingest_result['total_chunks']} chunks"
            ),
        )

        return {
            "status": "completed",
            "layer": layer_result["status"],
            "ingestion": ingest_result,
        }
