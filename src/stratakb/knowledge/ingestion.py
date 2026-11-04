"""Document ingestion pipeline orchestrator.

Pipeline: discover files → skip secret-looking names → skip unchanged → skip
empty → parse → chunk → embed → store per file → remove gone files → record
run → recount layer stats.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog

from stratakb.core.database import get_pool
from stratakb.core.migrations import ensure_schema
from stratakb.knowledge.embeddings import embed_texts
from stratakb.knowledge.layers import LayerNotFoundError, get_layer_status
from stratakb.knowledge.parsers.csv_parser import parse_csv
from stratakb.knowledge.parsers.docx_parser import parse_docx
from stratakb.knowledge.parsers.html_parser import parse_html
from stratakb.knowledge.parsers.json_parser import parse_json
from stratakb.knowledge.parsers.markdown_parser import parse_markdown
from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, parse_pdf
from stratakb.knowledge.parsers.xml_parser import parse_xml
from stratakb.knowledge.parsers.yaml_parser import parse_yaml
from stratakb.knowledge.policy import is_secret_name
from stratakb.knowledge.vector_store import ChunkRecord, insert_chunks

logger = structlog.get_logger()

# Supported file extensions and their parsers
SUPPORTED_PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".yaml": parse_yaml,
    ".yml": parse_yaml,
    ".csv": parse_csv,
    ".xml": parse_xml,
    ".xsd": parse_xml,
    ".json": parse_json,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
}

EMBED_BATCH_SIZE = 100

# Formats read as bytes by their parser: never decoded to check for empty text
_BINARY_FORMATS = {".pdf", ".docx"}


@dataclass(frozen=True)
class FileOutcome:
    """What one ingest run did with one discovered or tracked file.

    outcome: "added", "updated", "unchanged", "removed", "skipped" or "failed".
    reason: why a file was removed ("deleted", "secret", "no_text"), skipped
    ("secret", "no_text") or failed ("parse_error", "embed_error"); None
    otherwise.
    """

    relative_path: str
    outcome: str
    reason: str | None = None


@dataclass
class IngestionResult:
    """Result of an ingestion run."""

    layer_name: str
    status: str  # "completed", "partial", "failed"
    docs_processed: int = 0  # files parsed and stored in this run (added + updated)
    docs_failed: int = 0
    chunks_created: int = 0
    files_added: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_skipped: int = 0  # secret-looking names and files without text, not tracked
    # True when chunks from before per-file tracking were dropped and the layer rebuilt
    rebuilt_legacy: bool = False
    errors: list[dict] = field(default_factory=list)
    files: list[FileOutcome] = field(default_factory=list)
    duration_seconds: float = 0.0


async def _embed_chunks(
    api_key: str,
    chunks: list[ChunkRecord],
    embedding_model: str = "",
    embedding_dimensions: int = 0,
) -> None:
    """Fill in the chunks' embeddings, in batches of EMBED_BATCH_SIZE to bound memory."""
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        embeddings = await embed_texts(
            [c.content for c in batch],
            api_key=api_key,
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        for chunk, emb in zip(batch, embeddings):
            chunk.embedding = emb


async def _store_file(
    database_url: str,
    *,
    layer_id: str,
    source_root: str,
    relative_path: str,
    content_hash: str,
    size_bytes: int,
    chunks: list[ChunkRecord],
) -> None:
    """Replace one file's row and chunks in a single transaction."""
    pool = await get_pool(database_url)
    async with pool.acquire() as conn:
        async with conn.transaction():
            file_id = await conn.fetchval(
                """
                INSERT INTO strata_core.ingested_files
                    (layer_id, source_root, relative_path, content_hash, size_bytes, chunk_count)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (layer_id, source_root, relative_path) DO UPDATE
                SET content_hash = EXCLUDED.content_hash,
                    size_bytes = EXCLUDED.size_bytes,
                    chunk_count = EXCLUDED.chunk_count,
                    ingested_at = NOW()
                RETURNING id
                """,
                layer_id,
                source_root,
                relative_path,
                content_hash,
                size_bytes,
                len(chunks),
            )
            await conn.execute("DELETE FROM strata_vectors.chunks WHERE file_id = $1", file_id)
            for chunk in chunks:
                chunk.file_id = file_id
            await insert_chunks(database_url, chunks, conn=conn)


async def clear_layer(database_url: str, layer_name: str) -> None:
    """Delete every chunk and ingested_files row of a layer."""
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    async with pool.acquire() as conn:
        async with conn.transaction():
            chunks = await conn.execute(
                "DELETE FROM strata_vectors.chunks WHERE layer_name = $1", layer_name
            )
            files = await conn.execute(
                """
                DELETE FROM strata_core.ingested_files
                WHERE layer_id = (SELECT id FROM strata_core.knowledge_layers WHERE name = $1)
                """,
                layer_name,
            )
    logger.info(
        "layer_cleared",
        layer=layer_name,
        chunks=int(chunks.split()[-1]),
        files=int(files.split()[-1]),
    )


async def ingest_documents(
    database_url: str,
    openai_api_key: str,
    *,
    layer_name: str,
    source_path: str,
    embedding_model: str = "",
    embedding_dimensions: int = 0,
    replace_all: bool = False,
) -> IngestionResult:
    """Sync a file or folder into a knowledge layer.

    Files are tracked per layer and root (the resolved folder ingested, or a
    single file's parent folder). A re-run skips files whose SHA-256 is
    unchanged, replaces changed files, adds new ones and, for a folder, removes
    files ingested from that same folder that no longer exist. A file that
    fails to parse keeps its previous chunks. Each file's chunks are replaced
    in one transaction.

    A file whose name looks like a secret (see policy.is_secret_name) is
    never read: it is skipped, or removed when it was tracked. A file without
    text (0 bytes, whitespace only, or a parse that yields nothing) is not a
    failure: it is skipped, or removed when it was tracked. Every file gets a
    FileOutcome in result.files.

    Steps:
    1. Validate layer exists (and apply the schema migration if needed)
    2. Discover files at source_path
    3. Clear the layer (replace_all) or its pre-tracking chunks (legacy)
    4. For each file: check its name; for a changed or new file, check for
       text, then parse, chunk, embed, store
    5. Remove files that are gone from the root
    6. Record ingestion run
    7. Recount layer stats

    Args:
        database_url: PostgreSQL connection URL.
        openai_api_key: OpenAI API key (only needed for the openai embedding backend).
        layer_name: Target knowledge layer name.
        source_path: File or directory path to ingest.
        embedding_model: Model for generating embeddings.
        embedding_dimensions: Embedding vector dimensions.
        replace_all: Delete every chunk and tracked file of the layer first,
            then ingest everything (a full rebuild).

    Returns:
        IngestionResult with stats and any errors.
    """
    import time

    start = time.monotonic()
    result = IngestionResult(layer_name=layer_name, status="failed")

    await ensure_schema(database_url)

    # Step 1: Validate layer
    try:
        layer = await get_layer_status(database_url, name=layer_name)
    except LayerNotFoundError:
        result.errors.append({"error": f"Layer '{layer_name}' not found"})
        await _record_run(database_url, layer_id=None, result=result, source_path=source_path)
        return result

    layer_id = layer["id"]
    chunk_config = layer.get("chunk_config", {})
    chunk_size = chunk_config.get("chunk_size", 1000) if isinstance(chunk_config, dict) else 1000
    chunk_overlap = chunk_config.get("chunk_overlap", 200) if isinstance(chunk_config, dict) else 200

    # Step 2: Discover files
    source = Path(source_path)
    if not source.exists():
        result.errors.append({"error": f"Source path not found: {source_path}"})
        await _record_run(database_url, layer_id=layer_id, result=result, source_path=source_path)
        return result

    source = source.resolve()
    single_file = source.is_file()
    root = source.parent if single_file else source
    source_root = root.as_posix()

    files = _discover_files(source)
    pool = await get_pool(database_url)

    # Step 3: Full rebuild, or a one-off rebuild of a layer ingested before tracking
    if replace_all:
        await clear_layer(database_url, layer_name)
    else:
        legacy = await pool.execute(
            "DELETE FROM strata_vectors.chunks WHERE layer_name = $1 AND file_id IS NULL",
            layer_name,
        )
        legacy_count = int(legacy.split()[-1])
        if legacy_count > 0:
            result.rebuilt_legacy = True
            logger.info("legacy_chunks_removed", layer=layer_name, count=legacy_count)

    rows = await pool.fetch(
        """
        SELECT id, relative_path, content_hash FROM strata_core.ingested_files
        WHERE layer_id = $1 AND source_root = $2
        """,
        layer_id,
        source_root,
    )
    tracked = {r["relative_path"]: r for r in rows}

    if not files and (single_file or not tracked):
        result.errors.append({"error": f"No supported files found at: {source_path}"})
        await _record_run(database_url, layer_id=layer_id, result=result, source_path=source_path)
        if result.rebuilt_legacy:
            await _update_layer_stats(database_url, layer_name)
        return result

    logger.info("ingestion_started", layer=layer_name, files=len(files))

    async def drop(relative_path: str, previous, reason: str) -> None:
        """Skip a file, or remove it (row and chunks) when it was tracked."""
        if previous is None:
            result.files.append(FileOutcome(relative_path, "skipped", reason))
            result.files_skipped += 1
            return
        await pool.execute("DELETE FROM strata_core.ingested_files WHERE id = $1", previous["id"])
        result.files.append(FileOutcome(relative_path, "removed", reason))
        result.files_removed += 1

    def fail(relative_path: str, error: str, reason: str) -> None:
        result.files.append(FileOutcome(relative_path, "failed", reason))
        result.docs_failed += 1
        result.errors.append({"file": relative_path, "error": error})

    # Step 4: Parse, chunk, embed and store each new or changed file
    seen: set[str] = set()
    for file_path in files:
        relative_path = file_path.relative_to(root).as_posix()
        seen.add(relative_path)
        previous = tracked.get(relative_path)

        # Checked on the name alone: a secret-looking file is never opened
        if is_secret_name(relative_path):
            await drop(relative_path, previous, "secret")
            continue

        try:
            data = file_path.read_bytes()
        except OSError as e:
            fail(relative_path, str(e), "parse_error")
            continue
        content_hash = hashlib.sha256(data).hexdigest()

        if previous is not None and previous["content_hash"] == content_hash:
            result.files_unchanged += 1
            result.files.append(FileOutcome(relative_path, "unchanged"))
            continue

        if not data or (
            file_path.suffix.lower() not in _BINARY_FORMATS
            and not data.decode("utf-8-sig", errors="replace").strip()
        ):
            await drop(relative_path, previous, "no_text")
            continue

        try:
            parsed = _parse_file(file_path)
        except Exception as e:
            logger.warning(
                "file_parse_raised", layer=layer_name, file=relative_path,
                error_type=type(e).__name__,
            )
            fail(relative_path, str(e), "parse_error")
            continue
        if parsed.error:
            fail(relative_path, parsed.error, "parse_error")
            continue
        parsed.source_file = relative_path

        doc_chunks = _chunk_document(
            parsed,
            layer_name=layer_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not doc_chunks:
            await drop(relative_path, previous, "no_text")
            continue

        try:
            await _embed_chunks(
                openai_api_key,
                doc_chunks,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
            )
            await _store_file(
                database_url,
                layer_id=layer_id,
                source_root=source_root,
                relative_path=relative_path,
                content_hash=content_hash,
                size_bytes=len(data),
                chunks=doc_chunks,
            )
        except Exception as e:
            logger.warning(
                "file_ingestion_failed", layer=layer_name, file=relative_path, error=str(e)
            )
            fail(relative_path, str(e), "embed_error")
            continue

        if previous is None:
            result.files_added += 1
            result.files.append(FileOutcome(relative_path, "added"))
        else:
            result.files_updated += 1
            result.files.append(FileOutcome(relative_path, "updated"))
        result.docs_processed += 1
        result.chunks_created += len(doc_chunks)

    # Step 5: Remove files ingested from this root that no longer exist
    if not single_file:
        gone = [
            (path, r["id"]) for path, r in tracked.items()
            if path not in seen and not (root / path).exists()
        ]
        if gone:
            await pool.execute(
                "DELETE FROM strata_core.ingested_files WHERE id = ANY($1::uuid[])",
                [file_id for _, file_id in gone],
            )
            result.files.extend(FileOutcome(path, "removed", "deleted") for path, _ in gone)
            result.files_removed += len(gone)

    # Step 6: Record ingestion run
    if result.docs_failed == 0:
        result.status = "completed"
    elif result.docs_processed or result.files_unchanged or result.files_removed:
        result.status = "partial"
    else:
        result.status = "failed"
    result.duration_seconds = time.monotonic() - start
    await _record_run(database_url, layer_id=layer_id, result=result, source_path=source_path)

    # Step 7: Recount layer stats
    await _update_layer_stats(database_url, layer_name)

    # Invalidate cached search results for this layer
    try:
        from stratakb.rag.cache import _get_cache
        cache = _get_cache()
        if cache:
            evicted = cache.invalidate_layer(layer_name)
            logger.info("cache_invalidated_after_ingestion", layer=layer_name, evicted=evicted)
    except Exception:
        pass  # Cache invalidation is best-effort

    logger.info(
        "ingestion_completed",
        layer=layer_name,
        status=result.status,
        docs_processed=result.docs_processed,
        docs_failed=result.docs_failed,
        chunks_created=result.chunks_created,
        files_added=result.files_added,
        files_updated=result.files_updated,
        files_unchanged=result.files_unchanged,
        files_removed=result.files_removed,
        files_skipped=result.files_skipped,
        duration=round(result.duration_seconds, 1),
    )

    return result


def _discover_files(source: Path) -> list[Path]:
    """Find all supported files in a path (file or directory)."""
    if source.is_file():
        ext = source.suffix.lower()
        if ext in SUPPORTED_PARSERS and SUPPORTED_PARSERS[ext] is not None:
            return [source]
        return []

    files = []
    for ext, parser in SUPPORTED_PARSERS.items():
        if parser is not None:  # skip extensions mapped to None (e.g. .fpr)
            files.extend(source.rglob(f"*{ext}"))
    return sorted(files)


def _parse_file(file_path: Path) -> ParsedDocument:
    """Parse a file using the appropriate parser based on extension."""
    ext = file_path.suffix.lower()
    parser = SUPPORTED_PARSERS.get(ext)
    if parser is None:
        return ParsedDocument(source_file=file_path.name, error=f"Unsupported format: {ext}")
    return parser(file_path)


def _build_context_prefix(
    layer_name: str,
    source_file: str,
    page_number: int | None = None,
    section: str | None = None,
) -> str:
    """Build a context prefix for a chunk to improve embedding quality.

    Prepends [layer > filename > page/section] to give the embedding model
    contextual information about where this chunk comes from.
    """
    file_stem = Path(source_file).stem if source_file else "unknown"

    parts = [layer_name, file_stem]
    if section:
        parts.append(section)
    elif page_number is not None:
        parts.append(f"Page {page_number}")

    return f"[{' > '.join(parts)}]\n"


def _chunk_document(
    doc: ParsedDocument,
    *,
    layer_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    """Split a parsed document into chunks using LangChain text splitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Derive doc_type from file extension
    ext = Path(doc.source_file).suffix.lower()
    doc_type_map = {".pdf": "pdf", ".docx": "docx", ".yaml": "yaml", ".yml": "yaml", ".csv": "csv", ".xml": "xml", ".xsd": "xsd", ".json": "json", ".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html"}
    doc_type = doc_type_map.get(ext, ext.lstrip(".") or "unknown")

    chunks: list[ChunkRecord] = []

    for page in doc.pages:
        page_chunks = splitter.split_text(page.text)
        for chunk_text in page_chunks:
            prefix = _build_context_prefix(
                layer_name=layer_name,
                source_file=doc.source_file,
                page_number=page.page_number,
            )
            chunks.append(
                ChunkRecord(
                    content=prefix + chunk_text,
                    embedding=[],  # Filled later
                    layer_name=layer_name,
                    metadata={
                        "page": page.page_number,
                        "total_pages": doc.total_pages,
                    },
                    source_file=doc.source_file,
                    source_section=f"page_{page.page_number}",
                    doc_type=doc_type,
                )
            )

    return chunks


async def _record_run(
    database_url: str,
    *,
    layer_id: str | None,
    result: IngestionResult,
    source_path: str,
) -> None:
    """Record an ingestion run in strata_core.ingestion_runs."""
    if layer_id is None:
        return

    pool = await get_pool(database_url)
    await pool.execute(
        """
        INSERT INTO strata_core.ingestion_runs
            (layer_id, source_path, status, docs_processed, docs_failed,
             chunks_created, error_log, completed_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
        """,
        layer_id,
        source_path,
        result.status,
        result.docs_processed,
        result.docs_failed,
        result.chunks_created,
        json.dumps(result.errors) if result.errors else None,
        datetime.now(timezone.utc),
    )


async def _update_layer_stats(database_url: str, layer_name: str) -> None:
    """Recount doc_count and chunk_count from the database and set last_ingested."""
    pool = await get_pool(database_url)
    await pool.execute(
        """
        UPDATE strata_core.knowledge_layers AS kl
        SET doc_count = (
                SELECT COUNT(*) FROM strata_core.ingested_files f WHERE f.layer_id = kl.id
            ),
            chunk_count = (
                SELECT COUNT(*) FROM strata_vectors.chunks c WHERE c.layer_name = kl.name
            ),
            last_ingested = NOW(),
            updated_at = NOW()
        WHERE kl.name = $1
        """,
        layer_name,
    )
