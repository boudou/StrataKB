"""Document ingestion pipeline orchestrator.

Pipeline: discover files (or take a given file list) → skip secret-looking
names → skip unchanged → skip empty → parse → chunk → embed → store per file
→ remove gone (or unlisted) files → record run → recount layer stats.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import asyncpg
import structlog

from stratakb.core.database import get_pool
from stratakb.core.migrations import ensure_schema
from stratakb.knowledge.documents import (
    build_document_text,
    page_starts,
    prune_documents,
    store_document_version,
)
from stratakb.knowledge.embeddings import embed_texts
from stratakb.knowledge.layers import LayerNotFoundError, get_layer_status
from stratakb.knowledge.parsers.pdf_parser import ParsedDocument
from stratakb.knowledge.parsers.registry import SUPPORTED_PARSERS
from stratakb.knowledge.policy import is_secret_name
from stratakb.knowledge.vector_store import ChunkRecord, chunk_embed_text, insert_chunks

if TYPE_CHECKING:
    from stratakb.knowledge.selection import FileEntry

logger = structlog.get_logger()

EMBED_BATCH_SIZE = 100

# Formats read as bytes by their parser: never decoded to check for empty text
_BINARY_FORMATS = {".pdf", ".docx"}

# A PDF averaging fewer characters per page than this probably holds scanned images
_SCANNED_MIN_CHARS_PER_PAGE = 200


@dataclass(frozen=True)
class FileOutcome:
    """What one ingest run did with one discovered or tracked file.

    outcome: "added", "updated", "unchanged", "removed", "skipped", "flagged"
    or "failed".
    reason: why a file was removed ("deleted", "not_selected" when a file list
    left out a file that still exists, "secret", "no_text"), skipped
    ("secret", "no_text"), flagged ("probably_scanned": stored, and counted
    as added or updated) or failed ("parse_error", "embed_error"); None
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
    files_flagged: int = 0  # PDFs that look scanned; also counted as added or updated
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
            [chunk_embed_text(c.layer_name, c.source_file, c.metadata, c.content) for c in batch],
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
    document_text: str,
    page_map: list[list[int]] | None = None,
    text_origin: str = "text",
) -> None:
    """Replace one file's row, document version and chunks in a single transaction.

    The text is stored as the file's current document version (reused when
    unchanged); versions no chunk references any more are pruned afterwards.
    """
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
            document_id, _ = await store_document_version(
                conn,
                layer_id=layer_id,
                file_id=file_id,
                source_root=source_root,
                relative_path=relative_path,
                text=document_text,
                page_map=page_map,
                text_origin=text_origin,
            )
            for chunk in chunks:
                chunk.file_id = file_id
                chunk.document_id = document_id
                # Checked here so a stored span always holds the chunk's text
                if chunk.char_start is not None and (
                    document_text[chunk.char_start:chunk.char_end] != chunk.content
                ):
                    logger.warning(
                        "chunk_offsets_mismatch", file=relative_path, char_start=chunk.char_start
                    )
                    chunk.char_start = chunk.char_end = None
            await insert_chunks(database_url, chunks, conn=conn)
            # Only after the insert: until then the current version is unreferenced
            await prune_documents(conn, layer_id)


async def clear_layer(database_url: str, layer_name: str) -> None:
    """Delete every chunk and ingested_files row of a layer, and its unreferenced documents."""
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
            layer_id = await conn.fetchval(
                "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", layer_name
            )
            if layer_id is not None:
                await prune_documents(conn, layer_id)
    # After the commit: the recount runs on another connection
    await _update_layer_stats(database_url, layer_name)
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
    files: Sequence[FileEntry] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IngestionResult:
    """Sync a file, a folder or a list of files under a folder into a knowledge layer.

    Files are tracked per layer and root (the resolved folder ingested, or a
    single file's parent folder). A re-run skips files whose SHA-256 is
    unchanged, replaces changed files, adds new ones and, for a folder, removes
    files ingested from that same folder that no longer exist. A file that
    fails to parse keeps its previous chunks. Each file's chunks are replaced
    in one transaction.

    A file whose name looks like a secret (see policy.is_secret_name) is
    never read: it is skipped, or removed when it was tracked. A file without
    text (0 bytes, whitespace only, or a parse that yields nothing) is not a
    failure: it is skipped, or removed when it was tracked. A PDF that looks
    scanned (see _probably_scanned) is stored for the text it has and flagged.
    Every file gets a FileOutcome in result.files.

    With files, source_path is the root folder and the list is the full
    ingest set: nothing is discovered, and every tracked file of that root
    that is not in the list is removed ("not_selected" when it still exists,
    "deleted" when it is gone). An empty list removes everything tracked.
    A database connection lost while a file is stored raises instead of
    being recorded as a failed file.

    Steps:
    1. Validate layer exists (and apply the schema migration if needed)
    2. Discover files at source_path, or take the given file list
    3. Clear the layer (replace_all) or its pre-tracking chunks (legacy)
    4. For each file: check its name; for a changed or new file, check for
       text, then parse, chunk, embed, store
    5. Remove files that are gone from the root (for a file list: every
       tracked file not in it)
    6. Record ingestion run
    7. Recount layer stats (always, even when the run fails or is interrupted)

    Args:
        database_url: PostgreSQL connection URL.
        openai_api_key: OpenAI API key (only needed for the openai embedding backend).
        layer_name: Target knowledge layer name.
        source_path: File or directory path to ingest.
        embedding_model: Model for generating embeddings.
        embedding_dimensions: Embedding vector dimensions.
        replace_all: Delete every chunk and tracked file of the layer first,
            then ingest everything (a full rebuild).
        files: Ingest exactly these files (paths relative to source_path,
            which must be a folder). An entry's sha256, when set, is used
            instead of hashing the file. Can't be combined with replace_all.
        progress: Called as progress(done, total) after each added or
            updated file is stored. total counts the files that may need
            storing (not secret-named, and without a carried hash equal to
            the tracked one), so done can end below total. Must not raise.

    Returns:
        IngestionResult with stats and any errors.

    Raises:
        ValueError: files with replace_all, files with a source_path that is
            not a folder, or an entry path that is absolute or contains "..".
    """
    import time

    if files is not None:
        if replace_all:
            raise ValueError("files can't be combined with replace_all")
        if not Path(source_path).is_dir():
            raise ValueError(f"With files, source_path must be a folder: {source_path}")
        for entry in files:
            # Checked in POSIX and native form: on Windows "C:/x" and "..\\x" escape too
            for rel in (PurePosixPath(entry.relative_path), Path(entry.relative_path)):
                if rel.is_absolute() or rel.anchor or ".." in rel.parts:
                    raise ValueError(
                        f"File path must be relative to the root: {entry.relative_path}"
                    )

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
    single_file = files is None and source.is_file()
    root = source.parent if single_file else source
    source_root = root.as_posix()

    # (relative path, path to read, carried hash) for each file to sync
    work: list[tuple[str, Path, str | None]]
    if files is None:
        work = [(p.relative_to(root).as_posix(), p, None) for p in _discover_files(source)]
    else:
        listed: dict[str, str | None] = {}
        for entry in files:
            listed.setdefault(entry.relative_path, entry.sha256)
        # Read under the root, so the file read is always the file tracked
        work = [(rel, root / rel, sha) for rel, sha in listed.items()]
    pool = await get_pool(database_url)

    try:
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

        # A file list may be empty: it then removes everything tracked
        if files is None and not work and (single_file or not tracked):
            result.errors.append({"error": f"No supported files found at: {source_path}"})
            await _record_run(
                database_url, layer_id=layer_id, result=result, source_path=source_path
            )
            return result

        logger.info("ingestion_started", layer=layer_name, files=len(work))

        # Files that may need storing, for progress: a carried hash equal to
        # the tracked one is known to be unchanged
        total = sum(
            1 for rel, _, carried in work
            if not is_secret_name(rel)
            and (carried is None or rel not in tracked or tracked[rel]["content_hash"] != carried)
        )
        done = 0

        async def drop(relative_path: str, previous, reason: str) -> None:
            """Skip a file, or remove it (row and chunks) when it was tracked."""
            if previous is None:
                result.files.append(FileOutcome(relative_path, "skipped", reason))
                result.files_skipped += 1
                return
            async with pool.acquire() as conn, conn.transaction():
                await conn.execute(
                    "DELETE FROM strata_core.ingested_files WHERE id = $1", previous["id"]
                )
                await prune_documents(conn, layer_id)
            result.files.append(FileOutcome(relative_path, "removed", reason))
            result.files_removed += 1

        def fail(relative_path: str, error: str, reason: str) -> None:
            result.files.append(FileOutcome(relative_path, "failed", reason))
            result.docs_failed += 1
            result.errors.append({"file": relative_path, "error": error})

        # Step 4: Parse, chunk, embed and store each new or changed file
        seen: set[str] = set()
        for relative_path, file_path, carried_hash in work:
            seen.add(relative_path)
            previous = tracked.get(relative_path)

            # Checked on the name alone: a secret-looking file is never opened
            if is_secret_name(relative_path):
                await drop(relative_path, previous, "secret")
                continue

            # A carried hash equal to the tracked one: unchanged, never opened
            if (
                carried_hash is not None
                and previous is not None
                and previous["content_hash"] == carried_hash
            ):
                result.files_unchanged += 1
                result.files.append(FileOutcome(relative_path, "unchanged"))
                continue

            try:
                data = file_path.read_bytes()
            except OSError as e:
                fail(relative_path, str(e), "parse_error")
                continue
            content_hash = carried_hash or hashlib.sha256(data).hexdigest()

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
            except Exception as e:
                logger.warning(
                    "file_ingestion_failed", layer=layer_name, file=relative_path, error=str(e)
                )
                fail(relative_path, str(e), "embed_error")
                continue

            document_text, page_map = build_document_text(parsed)
            try:
                await _store_file(
                    database_url,
                    layer_id=layer_id,
                    source_root=source_root,
                    relative_path=relative_path,
                    content_hash=content_hash,
                    size_bytes=len(data),
                    chunks=doc_chunks,
                    document_text=document_text,
                    page_map=page_map,
                )
            except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError, OSError):
                # A lost database is an error for the whole run, not one failed file
                raise
            except Exception as e:
                logger.warning(
                    "file_ingestion_failed", layer=layer_name, file=relative_path, error=str(e)
                )
                fail(relative_path, str(e), "embed_error")
                continue

            if previous is None:
                result.files_added += 1
                outcome = FileOutcome(relative_path, "added")
            else:
                result.files_updated += 1
                outcome = FileOutcome(relative_path, "updated")
            if file_path.suffix.lower() == ".pdf" and _probably_scanned(parsed):
                result.files_flagged += 1
                outcome = FileOutcome(relative_path, "flagged", "probably_scanned")
            result.files.append(outcome)
            result.docs_processed += 1
            result.chunks_created += len(doc_chunks)
            done += 1
            if progress is not None:
                progress(done, total)

        # Step 5: Remove files ingested from this root that no longer exist
        # or, for a file list, every tracked file that is not in it
        if not single_file:
            gone = [
                (path, r["id"]) for path, r in tracked.items()
                if path not in seen and (files is not None or not (root / path).exists())
            ]
            if gone:
                async with pool.acquire() as conn, conn.transaction():
                    await conn.execute(
                        "DELETE FROM strata_core.ingested_files WHERE id = ANY($1::uuid[])",
                        [file_id for _, file_id in gone],
                    )
                    await prune_documents(conn, layer_id)
                result.files.extend(
                    FileOutcome(
                        path, "removed", "deleted" if not (root / path).exists() else "not_selected"
                    )
                    for path, _ in gone
                )
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
    finally:
        # Step 7: Recount layer stats, even when the run failed or was interrupted
        await _update_layer_stats(database_url, layer_name)

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
        files_flagged=result.files_flagged,
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


def _probably_scanned(doc: ParsedDocument) -> bool:
    """Whether a parsed PDF looks scanned: under half its pages have text, or
    too few characters per page on average. The PDF parser keeps only pages
    with text, so len(doc.pages) counts those."""
    if doc.total_pages <= 0 or not doc.pages:
        return False
    return (
        2 * len(doc.pages) < doc.total_pages
        or sum(len(p.text) for p in doc.pages) / doc.total_pages < _SCANNED_MIN_CHARS_PER_PAGE
    )


def _chunk_document(
    doc: ParsedDocument,
    *,
    layer_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    """Split a parsed document into chunks using LangChain text splitter.

    Each chunk gets char_start and char_end into the document text (see
    documents.build_document_text), or None when its text is not found.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Derive doc_type from file extension
    ext = Path(doc.source_file).suffix.lower()
    doc_type_map = {".pdf": "pdf", ".docx": "docx", ".yaml": "yaml", ".yml": "yaml", ".csv": "csv", ".xml": "xml", ".xsd": "xsd", ".json": "json", ".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html", ".txt": "text", ".rst": "rst", ".adoc": "adoc"}
    doc_type = doc_type_map.get(ext, ext.lstrip(".") or "unknown")

    chunks: list[ChunkRecord] = []

    for page, page_start in zip(doc.pages, page_starts(doc), strict=True):
        page_chunks = splitter.split_text(page.text)
        metadata: dict = {
            "page": page.page_number,
            "total_pages": doc.total_pages,
        }
        if page.section_path:
            metadata["section_path"] = list(page.section_path)
        # Overlapping chunks repeat text, and a page can repeat itself: a chunk
        # starts after the previous one's start and at most chunk_overlap
        # characters before its end, so search from there
        prev_start, prev_end = -1, 0
        for chunk_text in page_chunks:
            pos = page.text.find(chunk_text, max(prev_start + 1, prev_end - chunk_overlap))
            if pos < 0:
                pos = page.text.find(chunk_text, prev_start + 1)
            if pos < 0:
                logger.warning(
                    "chunk_offsets_not_found", file=doc.source_file, page=page.page_number
                )
                char_start = char_end = None
            else:
                prev_start, prev_end = pos, pos + len(chunk_text)
                char_start = page_start + pos
                char_end = char_start + len(chunk_text)
            # content is the page's own text; the context prefix is only embedded
            chunks.append(
                ChunkRecord(
                    content=chunk_text,
                    embedding=[],  # Filled later
                    layer_name=layer_name,
                    metadata=dict(metadata),
                    source_file=doc.source_file,
                    source_section=f"page_{page.page_number}",
                    doc_type=doc_type,
                    char_start=char_start,
                    char_end=char_end,
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
    """Recount doc_count and chunk_count from the database and set last_ingested.

    Also bumps updated_at, which tells every process's caches the layer
    changed. A failure is logged, not raised.
    """
    try:
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
    except Exception as e:
        logger.warning("layer_stats_update_failed", layer=layer_name, error_type=type(e).__name__)
