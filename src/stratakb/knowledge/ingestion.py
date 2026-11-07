"""Document ingestion pipeline orchestrator.

Pipeline: discover files (or take a given file list) → skip secret-looking
names → skip unchanged → skip empty → parse → map structure → chunk along it
→ embed → store per file → remove gone (or unlisted) files → record run → recount layer stats.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, get_args

import asyncpg
import structlog

from stratakb.core.database import get_pool
from stratakb.core.migrations import ensure_schema
from stratakb.knowledge.document_dates import backfill_dates, document_date
from stratakb.knowledge.documents import (
    FileScope,
    build_document_text,
    document_headings,
    lock_file_rows,
    page_starts,
    prune_documents,
    store_document_version,
    store_sections,
)
from stratakb.knowledge.embeddings import embed_texts
from stratakb.knowledge.layers import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    LayerNotFoundError,
    get_layer_status,
)
from stratakb.knowledge.parsers.document import FlagReason, ParsedDocument
from stratakb.knowledge.parsers.registry import FORMATS, Format
from stratakb.knowledge.policy import is_secret_name
from stratakb.knowledge.structure import (
    Section,
    chunk_spans,
    detect_sections,
)
from stratakb.knowledge.supersedes import declared_supersedes
from stratakb.knowledge.vector_store import ChunkRecord, chunk_embed_text, insert_chunks

if TYPE_CHECKING:
    from uuid import UUID

    from stratakb.knowledge.selection import FileEntry

logger = structlog.get_logger()

EMBED_BATCH_SIZE = 100

# The longest error message recorded for one file
_ERROR_MAX_CHARS = 300

Outcome = Literal["added", "updated", "unchanged", "removed", "skipped", "flagged", "failed"]
SkippedReason = Literal["secret", "no_text"]
RemovedReason = Literal["deleted", "not_selected", SkippedReason]
FailedReason = Literal["read_error", "parse_error", "embed_error"]
Reason = Literal[RemovedReason, FlagReason, FailedReason]

# The reasons each outcome may carry; None: no reason
REASONS: dict[Outcome, frozenset[Reason | None]] = {
    "added": frozenset({None}),
    "updated": frozenset({None}),
    "unchanged": frozenset({None}),
    "removed": frozenset(get_args(RemovedReason)),
    "skipped": frozenset(get_args(SkippedReason)),
    "flagged": frozenset(get_args(FlagReason)),
    "failed": frozenset(get_args(FailedReason)),
}


@dataclass(frozen=True)
class FileOutcome:
    """What one ingest run did with one discovered or tracked file.

    outcome: "added", "updated", "unchanged", "removed", "skipped", "flagged"
    or "failed".
    reason: why a file was removed ("deleted", "not_selected" when a file list
    left out a file that still exists, "secret", "no_text"), skipped
    ("secret", "no_text"), flagged ("probably_scanned", "truncated": stored,
    and counted as added or updated) or failed ("read_error" when it could not
    be read, "parse_error", "embed_error"); None otherwise. REASONS holds the
    allowed pairs; any other pair raises ValueError.
    count: how many items (rows) the parser's limit left out of a flagged
    file; None when not counted. A count on any other outcome, or below 1,
    raises ValueError.
    stored: how a flagged file was stored, "added" or "updated"; required on a
    flagged file, and on any other outcome a value raises ValueError.
    """

    relative_path: str
    outcome: Outcome
    reason: Reason | None = None
    count: int | None = None
    stored: Literal["added", "updated"] | None = None

    def __post_init__(self) -> None:
        if self.reason not in REASONS.get(self.outcome, frozenset()):
            raise ValueError(
                f"Not an ingest outcome and reason: {self.outcome!r}, {self.reason!r}"
            )
        if self.count is not None and (self.outcome != "flagged" or self.count < 1):
            raise ValueError(
                f"Not a count for this outcome: {self.outcome!r}, {self.count!r}"
            )
        if self.stored not in (("added", "updated") if self.outcome == "flagged" else (None,)):
            raise ValueError(
                f"Not a stored value for this outcome: {self.outcome!r}, {self.stored!r}"
            )


@dataclass
class IngestionResult:
    """Result of an ingestion run; files holds what the file counters count."""

    layer_name: str
    status: str  # "completed", "partial", "failed"
    chunks_created: int = 0
    # True when chunks from before per-file tracking were dropped and the layer rebuilt
    rebuilt_legacy: bool = False
    errors: list[dict] = field(default_factory=list)
    files: list[FileOutcome] = field(default_factory=list)
    duration_seconds: float = 0.0

    def _count(self, outcome: Outcome) -> int:
        return sum(1 for f in self.files if f.outcome == outcome)

    def _count_stored(self, stored: Literal["added", "updated"]) -> int:
        return sum(1 for f in self.files if f.outcome == stored or f.stored == stored)

    @property
    def docs_processed(self) -> int:
        """Files parsed and stored in this run (added + updated)."""
        return self.files_added + self.files_updated

    @property
    def docs_failed(self) -> int:
        return self._count("failed")

    @property
    def files_added(self) -> int:
        return self._count_stored("added")

    @property
    def files_updated(self) -> int:
        return self._count_stored("updated")

    @property
    def files_unchanged(self) -> int:
        return self._count("unchanged")

    @property
    def files_removed(self) -> int:
        return self._count("removed")

    @property
    def files_skipped(self) -> int:
        """Secret-looking names and files without text, not tracked."""
        return self._count("skipped")

    @property
    def files_flagged(self) -> int:
        """Files their parser flagged; also counted as added or updated."""
        return self._count("flagged")


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
    sections: Sequence[Section] = (),
    doc_date: str | None = None,
    doc_date_source: str | None = None,
    supersedes: str | None = None,
) -> None:
    """Replace one file's row, document version and chunks in a single transaction.

    The text is stored as the file's current document version (reused when
    unchanged), with its date, what it replaces and its structure map
    (sections); this file's
    versions no chunk references any more are pruned afterwards.
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
                doc_date=doc_date,
                doc_date_source=doc_date_source,
                supersedes=supersedes,
            )
            await store_sections(conn, document_id, sections)
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
            await prune_documents(conn, [layer_id], files=FileScope(source_root, [relative_path]))


async def _remove_files(
    pool: asyncpg.Pool,
    *,
    layer_id: str,
    source_root: str,
    relative_paths: Sequence[str],
) -> None:
    """Remove tracked files of a layer and root, in one transaction.

    Deleting the rows cascades to their chunks and unlinks their documents,
    so their versions are pruned afterwards. The rows are locked before
    anything is deleted (see lock_file_rows). The delete targets the locked
    ids, so a row committed after the lock is left alone.
    """
    scope = FileScope(source_root, relative_paths)
    async with pool.acquire() as conn, conn.transaction():
        file_ids = await lock_file_rows(conn, layer_id, files=scope)
        await conn.execute(
            "DELETE FROM strata_core.ingested_files WHERE id = ANY($1::uuid[])", file_ids
        )
        await prune_documents(conn, [layer_id], files=scope)


async def tracked_files(
    database_url: str, layer_id: UUID, source_root: str
) -> dict[str, asyncpg.Record]:
    """The tracked files of a layer and root, keyed by relative path.

    Each row holds relative_path and content_hash. The caller has already run
    ensure_schema.
    """
    pool = await get_pool(database_url)
    rows = await pool.fetch(
        """
        SELECT relative_path, content_hash FROM strata_core.ingested_files
        WHERE layer_id = $1 AND source_root = $2
        """,
        layer_id,
        source_root,
    )
    return {r["relative_path"]: r for r in rows}


async def clear_layer(database_url: str, layer_name: str) -> None:
    """Delete every chunk and ingested_files row of a layer, and its unreferenced documents.

    The layer's file rows are locked before anything is deleted (see
    lock_file_rows). A file first ingested meanwhile is not among them and is
    not cleared; the prune meets its ingest on the file's versions only, which
    both lock in id order.
    """
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    files = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            layer_id = await conn.fetchval(
                "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", layer_name
            )
            file_ids = await lock_file_rows(conn, layer_id)
            # Only the locked files' chunks, plus legacy chunks that have no file row:
            # a file first ingested meanwhile is not cleared
            chunks = await conn.execute(
                """
                DELETE FROM strata_vectors.chunks
                WHERE layer_name = $1 AND (file_id IS NULL OR file_id = ANY($2::uuid[]))
                """,
                layer_name,
                file_ids,
            )
            if layer_id is not None:
                deleted = await conn.execute(
                    "DELETE FROM strata_core.ingested_files WHERE id = ANY($1::uuid[])", file_ids
                )
                files = int(deleted.split()[-1])
                await prune_documents(conn, [layer_id])
    # After the commit: the recount runs on another connection
    await _update_layer_stats(database_url, layer_name)
    logger.info(
        "layer_cleared",
        layer=layer_name,
        chunks=int(chunks.split()[-1]),
        files=files,
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
    cannot be read or fails to parse keeps its previous chunks, and its entry
    in result.errors carries the reason. Each file's chunks are replaced in
    one transaction.

    A file whose name looks like a secret (see policy.is_secret_name) is
    never read: it is skipped, or removed when it was tracked. A file without
    text (0 bytes, whitespace only, or a parse that yields nothing) is not a
    failure: it is skipped, or removed when it was tracked. A file its parser
    flags (a PDF that looks scanned, a CSV cut at its row limit) is stored for
    the text it has and flagged, with the parser's count.
    Every file gets a FileOutcome in result.files.

    With files, source_path is the root folder and the list is the full
    ingest set: nothing is discovered, and every tracked file of that root
    that is not in the list is removed ("not_selected" when it still exists,
    "deleted" when it is gone). An empty list removes everything tracked.
    A database connection lost while a file is stored raises instead of
    being recorded as a failed file.

    Each stored document version records a date: the front matter's updated or
    date, otherwise the file's modification time. An unchanged file whose
    latest version has none yet gets one, with nothing embedded.

    Steps:
    1. Validate layer exists (and apply the schema migration if needed)
    2. Discover files at source_path, or take the given file list
    3. Clear the layer (replace_all) or its pre-tracking chunks (legacy)
    4. For each file: check its name; for a changed or new file, check for
       text, then parse, chunk, embed, store; then date unchanged files
       stored without a date
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
    chunk_config = layer["chunk_config"]
    chunk_size = chunk_config.get("chunk_size", DEFAULT_CHUNK_SIZE)
    chunk_overlap = chunk_config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
    # The layer's own section starts (set by a preset), compiled once per run
    structure_patterns = [re.compile(p) for p in layer.get("structure_patterns") or []]

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

        tracked = await tracked_files(database_url, layer_id, source_root)

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
            if not is_secret_name(rel) and not _matches_tracked(carried, tracked.get(rel))
        )
        done = 0

        async def drop(relative_path: str, previous, reason: Reason) -> None:
            """Skip a file, or remove it (row and chunks) when it was tracked."""
            if previous is None:
                result.files.append(FileOutcome(relative_path, "skipped", reason))
                return
            await _remove_files(
                pool,
                layer_id=layer_id,
                source_root=source_root,
                relative_paths=[relative_path],
            )
            result.files.append(FileOutcome(relative_path, "removed", reason))

        def fail(relative_path: str, file_path: Path, error: str, reason: Reason) -> None:
            result.files.append(FileOutcome(relative_path, "failed", reason))
            error = _file_error(error, file_path=file_path, relative_path=relative_path, root=root)
            result.errors.append({"file": relative_path, "error": error, "reason": reason})

        # Step 4: Parse, chunk, embed and store each new or changed file
        seen: set[str] = set()
        unchanged: dict[str, Path] = {}
        for relative_path, file_path, carried_hash in work:
            seen.add(relative_path)
            previous = tracked.get(relative_path)

            # Checked on the name alone: a secret-looking file is never opened
            if is_secret_name(relative_path):
                await drop(relative_path, previous, "secret")
                continue

            # A carried hash equal to the tracked one: unchanged, not opened
            # here, and later only while its stored version has no date
            if _matches_tracked(carried_hash, previous):
                result.files.append(FileOutcome(relative_path, "unchanged"))
                unchanged[relative_path] = file_path
                continue

            try:
                data = file_path.read_bytes()
            except OSError as e:
                logger.warning(
                    "file_read_failed", layer=layer_name, file=relative_path,
                    error_type=type(e).__name__,
                )
                fail(relative_path, file_path, str(e), "read_error")
                continue
            content_hash = carried_hash or hashlib.sha256(data).hexdigest()

            if _matches_tracked(content_hash, previous):
                result.files.append(FileOutcome(relative_path, "unchanged"))
                unchanged[relative_path] = file_path
                continue

            # A listed file may have an unknown extension: _parse_file rejects it
            fmt = FORMATS.get(file_path.suffix.lower())
            if _has_no_text(fmt, data):
                await drop(relative_path, previous, "no_text")
                continue

            try:
                # Parsing, mapping and chunking are CPU work: off the event loop
                parsed = await asyncio.to_thread(_parse_file, file_path)
            except Exception as e:
                logger.warning(
                    "file_parse_raised", layer=layer_name, file=relative_path,
                    error_type=type(e).__name__,
                )
                fail(relative_path, file_path, str(e), "parse_error")
                continue
            if parsed.error:
                fail(relative_path, file_path, parsed.error, "parse_error")
                continue
            parsed.source_file = relative_path

            mapped, doc_chunks = await asyncio.to_thread(
                _map_and_chunk,
                parsed,
                layer_name=layer_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                structure_patterns=structure_patterns,
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
                    "file_ingestion_failed", layer=layer_name, file=relative_path,
                    error_type=type(e).__name__,
                )
                fail(relative_path, file_path, str(e), "embed_error")
                continue

            doc_date, doc_date_source = await asyncio.to_thread(document_date, file_path, data)
            supersedes = await asyncio.to_thread(declared_supersedes, fmt, data, relative_path)
            try:
                await _store_file(
                    database_url,
                    layer_id=layer_id,
                    source_root=source_root,
                    relative_path=relative_path,
                    content_hash=content_hash,
                    size_bytes=len(data),
                    chunks=doc_chunks,
                    document_text=mapped.text,
                    page_map=mapped.page_map,
                    sections=mapped.sections,
                    doc_date=doc_date,
                    doc_date_source=doc_date_source,
                    supersedes=supersedes,
                )
            except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError, OSError):
                # A lost database is an error for the whole run, not one failed file
                raise
            except Exception as e:
                logger.warning(
                    "file_ingestion_failed", layer=layer_name, file=relative_path,
                    error_type=type(e).__name__,
                )
                fail(relative_path, file_path, str(e), "embed_error")
                continue

            stored = "added" if previous is None else "updated"
            if parsed.flag is not None:
                outcome = FileOutcome(
                    relative_path, "flagged", parsed.flag.reason, parsed.flag.count, stored
                )
            else:
                outcome = FileOutcome(relative_path, stored)
            result.files.append(outcome)
            result.chunks_created += len(doc_chunks)
            done += 1
            if progress is not None:
                progress(done, total)

        await backfill_dates(
            pool,
            layer_id=layer_id,
            layer_name=layer_name,
            source_root=source_root,
            unchanged=unchanged,
        )

        # Step 5: Remove files ingested from this root that no longer exist
        # or, for a file list, every tracked file that is not in it
        if not single_file:
            gone = [
                path for path in tracked
                if path not in seen and (files is not None or not (root / path).exists())
            ]
            if gone:
                await _remove_files(
                    pool, layer_id=layer_id, source_root=source_root, relative_paths=gone
                )
                result.files.extend(
                    FileOutcome(
                        path, "removed", "deleted" if not (root / path).exists() else "not_selected"
                    )
                    for path in gone
                )

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
        return [source] if ext in FORMATS else []

    files = []
    for ext in FORMATS:
        files.extend(source.rglob(f"*{ext}"))
    return sorted(files)


def _file_error(message: str, *, file_path: Path, relative_path: str, root: Path) -> str:
    """An error message for one file, safe to return and store: the absolute
    path (which can hold the user's name) replaced by the relative one, the
    root by ".", and the text cut to _ERROR_MAX_CHARS. A path is replaced only
    where it is not part of a longer name (/data/docs2 stays with root
    /data/docs), and in any letter case when the folder's filesystem ignores
    case."""
    replacements = []
    for absolute, relative in ((file_path, relative_path), (root, ".")):
        if str(absolute) == absolute.anchor:
            continue  # a bare "/" or "C:\" would rewrite every slash
        for form in {str(absolute), absolute.as_posix()}:
            # OSError and repr() show a Windows path with doubled backslashes
            replacements += [(form.replace("\\", "\\\\"), relative), (form, relative)]
    flags = re.IGNORECASE if replacements and _case_insensitive_fs(root) else 0
    for form, relative in replacements:
        pattern = _PATH_BEFORE + re.escape(form) + _PATH_AFTER
        message = re.sub(pattern, lambda _, r=relative: r, message, flags=flags)
    if len(message) > _ERROR_MAX_CHARS:
        message = message[: _ERROR_MAX_CHARS - 3] + "..."
    return message


# A path in an error message starts where no name character precedes it and
# ends at the end, a separator, whitespace, a quote, closing punctuation or a
# sentence-ending period.
_PATH_BEFORE = r"(?<![\w.\-])"
_PATH_AFTER = r"(?=$|[\\/\s'\"),:;\]>]|\.(?:$|\s))"


def _case_insensitive_fs(path: Path) -> bool:
    """Whether the filesystem holding path ignores letter case: the path with
    its case swapped names the same file. True on Windows and default macOS
    volumes, False on Linux; False when the path has no letters or cannot be
    checked."""
    swapped = Path(str(path).swapcase())
    if str(swapped) == str(path):
        return False
    try:
        return swapped.exists() and os.path.samefile(path, swapped)
    except OSError:
        return False


def _parse_file(file_path: Path) -> ParsedDocument:
    """Parse a file using the appropriate parser based on extension."""
    ext = file_path.suffix.lower()
    fmt = FORMATS.get(ext)
    if fmt is None:
        return ParsedDocument(source_file=file_path.name, error=f"Unsupported format: {ext}")
    return fmt.parser(file_path)


def _matches_tracked(content_hash: str | None, previous) -> bool:
    """Whether a file's hash equals the hash tracked for it (so it is unchanged)."""
    return (
        content_hash is not None
        and previous is not None
        and previous["content_hash"] == content_hash
    )


def _has_no_text(fmt: Format | None, data: bytes) -> bool:
    """Whether a file's bytes hold no text: 0 bytes, or a text format (or an
    unknown one) that decodes to whitespace only. A binary format's bytes are
    left to its parser."""
    return not data or (
        (fmt is None or not fmt.binary)
        and not data.decode("utf-8-sig", errors="replace").strip()
    )


@dataclass(frozen=True)
class _DocumentMap:
    """A parsed document's text, page map and structure map, built once."""

    text: str
    page_map: list[list[int]] | None
    sections: list[Section]


def _map_document(
    doc: ParsedDocument, structure_patterns: Sequence[str | re.Pattern[str]] = ()
) -> _DocumentMap:
    """Build doc's text (documents.build_document_text) and its structure map.

    structure_patterns are the layer's own section starts, matched at line
    starts next to the generic detectors (structure.detect_sections). The
    headings the parser read outside the text (a PDF outline, Word heading
    styles) come first.
    """
    text, page_map = build_document_text(doc)
    fmt = FORMATS.get(Path(doc.source_file).suffix.lower())
    markdown = fmt is not None and fmt.doc_type == "markdown"
    sections = detect_sections(
        text, structure_patterns, markdown=markdown, explicit=document_headings(doc)
    )
    return _DocumentMap(text, page_map, sections)


def _map_and_chunk(
    doc: ParsedDocument,
    *,
    layer_name: str,
    chunk_size: int,
    chunk_overlap: int,
    structure_patterns: Sequence[str | re.Pattern[str]] = (),
) -> tuple[_DocumentMap, list[ChunkRecord]]:
    """doc's map and its chunks along it; sync, run in a worker thread."""
    mapped = _map_document(doc, structure_patterns)
    return mapped, _chunk_document(
        doc,
        layer_name=layer_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        mapped=mapped,
    )


def _chunk_document(
    doc: ParsedDocument,
    *,
    layer_name: str,
    chunk_size: int,
    chunk_overlap: int,
    mapped: _DocumentMap | None = None,
    structure_patterns: Sequence[str | re.Pattern[str]] = (),
) -> list[ChunkRecord]:
    """Split a parsed document into chunks along its structure map.

    mapped is the document's text and structure map (_map_document, built
    here with structure_patterns when None). The text is cut at every
    section and page start, and a chunk carries the clause_path of the
    section it lies in. Without sections, each page is split by size alone.

    Each chunk gets char_start and char_end into the document text (see
    documents.build_document_text).
    """
    # Derive doc_type from file extension
    ext = Path(doc.source_file).suffix.lower()
    fmt = FORMATS.get(ext)
    doc_type = fmt.doc_type if fmt is not None else ext.lstrip(".") or "unknown"

    if mapped is None:
        mapped = _map_document(doc, structure_patterns)
    text, sections = mapped.text, mapped.sections
    starts = page_starts(doc)

    def record(page, content: str, char_start: int | None, clause_path: str | None):
        metadata: dict = {
            "page": page.page_number,
            "total_pages": doc.total_pages,
        }
        if page.section_path:
            metadata["section_path"] = list(page.section_path)
        # content is the page's own text; the context prefix is only embedded
        return ChunkRecord(
            content=content,
            embedding=[],  # Filled later
            layer_name=layer_name,
            metadata=metadata,
            source_file=doc.source_file,
            source_section=f"page_{page.page_number}",
            doc_type=doc_type,
            char_start=char_start,
            char_end=char_start + len(content) if char_start is not None else None,
            clause_path=clause_path,
        )

    # Without sections, the cuts are the page starts alone: each page split by size
    return [
        # The page an offset lies in: page starts are cuts, so a chunk is in one page
        record(
            doc.pages[bisect_right(starts, char_start) - 1],
            text[char_start:char_end],
            char_start,
            path or None,
        )
        for char_start, char_end, path in chunk_spans(
            sections, text, chunk_size, chunk_overlap, boundaries=starts
        )
    ]


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
