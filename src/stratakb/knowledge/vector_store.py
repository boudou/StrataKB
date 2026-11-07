"""Vector store operations for strata_vectors.chunks.

Handles inserting, deleting, and querying document chunks with pgvector embeddings.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg
import structlog

from stratakb.core.database import get_pool

logger = structlog.get_logger()


@dataclass
class ChunkRecord:
    """A document chunk ready for storage."""

    content: str
    embedding: list[float]
    layer_name: str
    metadata: dict = field(default_factory=dict)
    source_file: str = ""
    source_section: str = ""
    doc_type: str = ""
    # strata_core.ingested_files row the chunk came from (None for chunks stored directly)
    file_id: str | None = None
    # strata_vectors.documents version and the span of its text the chunk holds
    # (None for chunks stored directly, or when the span could not be found)
    document_id: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    # Path of the innermost section the chunk lies in ('7 Termination > 7.2'),
    # None outside any section or without a structure map
    clause_path: str | None = None


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


def chunk_embed_text(layer_name: str, source_file: str, metadata: dict, content: str) -> str:
    """Text sent to the embedding model for a chunk: the context prefix, then the content.

    Built from the stored columns only, so ingestion and re-embedding produce the same text.
    """
    prefix = _build_context_prefix(
        layer_name=layer_name,
        source_file=source_file,
        page_number=metadata.get("page"),
        section=" > ".join(metadata.get("section_path") or []),
    )
    return prefix + content


async def delete_layer_chunks(database_url: str, layer_name: str) -> int:
    """Delete all chunks for a layer. Returns count of deleted rows."""
    pool = await get_pool(database_url)
    result = await pool.execute(
        "DELETE FROM strata_vectors.chunks WHERE layer_name = $1",
        layer_name,
    )
    count = int(result.split()[-1])
    logger.info("chunks_deleted", layer=layer_name, count=count)
    return count


async def insert_chunks(
    database_url: str,
    chunks: list[ChunkRecord],
    *,
    conn: asyncpg.Connection | None = None,
) -> int:
    """Batch insert chunks into the vector store.

    Uses asyncpg's executemany for efficient bulk insert. Pass conn to insert
    inside the caller's transaction.

    Returns:
        Number of chunks inserted.
    """
    if not chunks:
        return 0

    import json

    records = [
        (
            chunk.layer_name,
            chunk.content,
            _embedding_to_pgvector(chunk.embedding),
            json.dumps(chunk.metadata),
            chunk.source_file,
            chunk.source_section,
            chunk.doc_type,
            chunk.file_id,
            chunk.document_id,
            chunk.char_start,
            chunk.char_end,
            chunk.clause_path,
        )
        for chunk in chunks
    ]
    sql = """
        INSERT INTO strata_vectors.chunks
            (layer_name, content, embedding, metadata, source_file, source_section,
             doc_type, file_id, document_id, char_start, char_end, clause_path)
        VALUES ($1, $2, $3::vector, $4::jsonb, $5, $6, $7, $8, $9, $10, $11, $12)
        """

    if conn is not None:
        await conn.executemany(sql, records)
    else:
        pool = await get_pool(database_url)
        async with pool.acquire() as pooled:
            # Use copy or executemany for batch efficiency
            await pooled.executemany(sql, records)

    logger.info("chunks_inserted", layer=chunks[0].layer_name, count=len(chunks))
    return len(chunks)


async def ensure_hnsw_index(database_url: str) -> None:
    """Create HNSW index if it doesn't exist. Safe to call repeatedly."""
    pool = await get_pool(database_url)
    exists = await pool.fetchval(
        "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_embedding_hnsw'"
    )
    if not exists:
        logger.info("creating_hnsw_index")
        await pool.execute(
            "CREATE INDEX idx_chunks_embedding_hnsw "
            "ON strata_vectors.chunks USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 200)"
        )
        logger.info("hnsw_index_created")


async def re_embed_layer(
    database_url: str,
    layer_name: str,
    *,
    batch_size: int = 100,
    new_dimensions: int = 0,
) -> dict:
    """Re-embed all chunks in a layer using the current embedding model.

    Reads existing chunk content from the DB, generates new embeddings,
    and updates in-place. No re-parsing of source files needed.

    Every layer shares one vector column, so this never resizes it: when
    new_dimensions differs from the column size it refuses before touching
    anything. Use re_embed_all_layers() to change the size.

    Args:
        database_url: PostgreSQL connection URL.
        layer_name: Layer to re-embed.
        batch_size: Number of chunks to embed per batch.
        new_dimensions: New vector dimensions (0 = auto from config).

    Returns:
        Stats dict with layer_name, chunks_updated, model, dimensions, duration_seconds.

    Raises:
        ValueError: new_dimensions differs from the shared column's size.
        RuntimeError: an all-layer re-embed resized the column while this one ran.
    """
    import time

    from stratakb.config import get_settings

    settings = get_settings()
    if not new_dimensions:
        new_dimensions = settings.embedding_dimensions

    pool = await get_pool(database_url)
    start = time.monotonic()

    # The column is shared by every layer: one layer cannot change its size
    size = await vector_column_size(pool)
    if size > 0 and size != new_dimensions:
        raise ValueError(
            f"The vector column holds {size}-dimensional vectors and every layer shares it, "
            f"so layer {layer_name} cannot be re-embedded at {new_dimensions} dimensions on "
            f"its own. Keep EMBEDDING_DIMENSIONS at {size}, or re-embed every layer at once "
            "with vector_store.re_embed_all_layers(), which resizes the column."
        )

    async def write(ids: list, embeddings: list[list[float]]) -> None:
        async with pool.acquire() as conn, conn.transaction():
            # Holds off an all-layer resize until this batch is written
            await conn.execute("LOCK TABLE strata_vectors.chunks IN ROW EXCLUSIVE MODE")
            current = await vector_column_size(conn)
            if current > 0 and current != new_dimensions:
                raise RuntimeError(
                    f"The vector column was resized to {current} dimensions while layer "
                    f"{layer_name} was being re-embedded at {new_dimensions}: an all-layer "
                    "re-embed ran at the same time and already re-embedded this layer at the "
                    "new size. Nothing more was written; set EMBEDDING_DIMENSIONS to "
                    f"{current} before re-embedding a single layer again."
                )
            await _update_embeddings(conn, ids, embeddings)

    counts = await _re_embed_batches(
        pool,
        new_dimensions,
        start,
        batch_size=batch_size,
        write=write,
        layer_name=layer_name,
        layer=layer_name,
    )
    updated = sum(counts.values())

    duration = time.monotonic() - start
    logger.info(
        "re_embed_complete",
        layer=layer_name,
        chunks_updated=updated,
        model=settings.embedding_model,
        dimensions=new_dimensions,
        duration=round(duration, 1),
    )

    return {
        "layer_name": layer_name,
        "chunks_updated": updated,
        "model": settings.embedding_model,
        "dimensions": new_dimensions,
        "duration_seconds": round(duration, 1),
    }


# An HNSW index refuses vector columns wider than this
_HNSW_MAX_DIMENSIONS = 2000

# Session temporary table that holds the new vectors between the two phases
_STAGING = "pg_temp.re_embed_staging"

# Fewest staged ids copied per statement in phase 2 (fewer statements under the lock)
_SWAP_BATCH_MIN = 1000

# Default seconds each statement of the locked phase may take (the pool allows 30)
_LOCKED_STATEMENT_TIMEOUT = 600.0

# Extra unlocked passes for chunks added during phase 1, before the lock is taken
_MAX_STAGING_PASSES = 3


async def re_embed_all_layers(
    database_url: str,
    *,
    batch_size: int = 100,
    new_dimensions: int = 0,
    locked_timeout: float = _LOCKED_STATEMENT_TIMEOUT,
) -> dict:
    """Re-embed every chunk of every layer, resizing the shared vector column once.

    Runs in two phases on one connection. Phase 1 embeds every chunk, one batch
    of rows at a time, and stages the new vectors in a temporary table; it takes
    no lock on the chunks table, so searches and ingests continue. While more
    than batch_size chunks were added meanwhile, it stages those too, still
    without the lock (at most _MAX_STAGING_PASSES extra passes). Phase 2 is one
    transaction: it locks the table, resizes the column when the size changes,
    copies the staged vectors in, and embeds the few chunks still unstaged.
    Writes wait for all of phase 2, and searches wait too while it resizes the
    column. On any error, the column size and every stored vector stay as they were.

    Args:
        database_url: PostgreSQL connection URL.
        batch_size: Number of chunks to embed per batch.
        new_dimensions: New vector dimensions (0 = auto from config).
        locked_timeout: Seconds each statement of phase 2 may take, instead of
            the pool's command timeout.

    Returns:
        Stats dict with per-layer counts, chunks_updated, dimensions,
        previous_dimensions, resized and duration_seconds.

    Raises:
        ValueError: the new size exceeds what the search index supports.
        TimeoutError: a phase 2 statement took longer than locked_timeout;
            nothing changed.
    """
    import time

    from stratakb.config import get_settings

    settings = get_settings()
    target = int(new_dimensions or settings.embedding_dimensions)

    # The shipped schema always has the HNSW index, so no column can be wider than this
    if target > _HNSW_MAX_DIMENSIONS:
        raise ValueError(
            f"The search index takes at most {_HNSW_MAX_DIMENSIONS} dimensions; {target} is "
            f"too many. Choose a model or EMBEDDING_DIMENSIONS of {_HNSW_MAX_DIMENSIONS} or fewer."
        )

    pool = await get_pool(database_url)
    start = time.monotonic()
    # Every statement stays on this connection: the staging table belongs to its session
    async with pool.acquire() as conn, _staging_table(conn):
        await _stage_all(conn, target, start, batch_size=batch_size)
        try:
            layers, size = await _swap_in(
                conn, target, start, batch_size=batch_size, timeout=locked_timeout
            )
        except (TimeoutError, asyncpg.QueryCanceledError) as e:
            raise TimeoutError(
                "A statement in the locked phase of the re-embed took longer than "
                f"{locked_timeout:g} s, so the re-embed was rolled back: the column size and "
                "every stored vector are as they were. Run it again at a quieter moment, or "
                "pass a larger locked_timeout (seconds)."
            ) from e
    updated = sum(layers.values())

    duration = time.monotonic() - start
    logger.info(
        "re_embed_complete",
        layers="all",
        layer_count=len(layers),
        chunks_updated=updated,
        model=settings.embedding_model,
        dimensions=target,
        previous_dimensions=size,
        duration=round(duration, 1),
    )

    return {
        "layers": layers,
        "chunks_updated": updated,
        "model": settings.embedding_model,
        "dimensions": target,
        "previous_dimensions": size,
        "resized": size != target,
        "duration_seconds": round(duration, 1),
    }


@contextlib.asynccontextmanager
async def _staging_table(conn):
    """Create the staging table on conn and drop it on exit, without hiding an error."""
    # A pooled connection keeps temporary tables, so drop any leftover first
    await conn.execute(f"DROP TABLE IF EXISTS {_STAGING}")
    await conn.execute(
        "CREATE TEMP TABLE re_embed_staging (id uuid PRIMARY KEY, embedding text NOT NULL)"
    )
    try:
        yield
    finally:
        # A failed drop must not replace the error in flight; this is safe because the
        # temp table goes with the session, and the next run drops a leftover on entry
        if not conn.is_closed():
            try:
                await conn.execute(f"DROP TABLE IF EXISTS {_STAGING}")
            except Exception as e:
                logger.warning("re_embed_staging_not_dropped", error=type(e).__name__)


async def _stage_all(conn, target: int, start: float, *, batch_size: int) -> None:
    """Phase 1: embed every chunk and stage the vectors, without locking the chunks table.

    Chunks added meanwhile are staged in up to _MAX_STAGING_PASSES extra passes,
    so that phase 2 embeds at most about batch_size chunks under the lock.
    """

    async def stage(ids: list, embeddings: list[list[float]]) -> None:
        await conn.execute(
            f"INSERT INTO {_STAGING} (id, embedding) "
            "SELECT * FROM unnest($1::uuid[], $2::text[])",
            *_vector_params(ids, embeddings),
        )

    await _re_embed_batches(conn, target, start, batch_size=batch_size, write=stage, layers="all")
    # The unstaged-only anti-join reads the staging table: give the planner its size
    await conn.execute(f"ANALYZE {_STAGING}")
    for _ in range(_MAX_STAGING_PASSES):
        gap = await _count_chunks(conn, unstaged_only=True)
        if gap <= batch_size:
            return
        await _re_embed_batches(
            conn,
            target,
            start,
            batch_size=batch_size,
            write=stage,
            unstaged_only=True,
            layers="all",
            phase="stage_new",
        )
        await conn.execute(f"ANALYZE {_STAGING}")
    gap = await _count_chunks(conn, unstaged_only=True)
    if gap > batch_size:
        # Still correct under the lock; only the locked catch-up is longer than usual
        logger.info("re_embed_catch_up_large", unstaged=gap)


async def _swap_in(
    conn, target: int, start: float, *, batch_size: int, timeout: float
) -> tuple[dict[str, int], int]:
    """Phase 2: one transaction that locks, resizes if needed and writes the new vectors.

    Every statement runs with timeout seconds instead of the pool's command timeout.

    Returns:
        Number of chunks updated per layer, and the column size found under the lock.
    """
    async with conn.transaction():
        # Searches keep reading; writes to chunks (and a second re-embed) wait
        await conn.execute(
            "LOCK TABLE strata_vectors.chunks IN SHARE ROW EXCLUSIVE MODE", timeout=timeout
        )
        size = await vector_column_size(conn, timeout=timeout)
        if size != target:
            logger.info("resizing_vector_column", old=size, new=target)
            # USING NULL: the old vectors cannot be cast; the staged ones replace them
            await conn.execute(
                "ALTER TABLE strata_vectors.chunks "
                f"ALTER COLUMN embedding TYPE vector({target}) USING NULL",
                timeout=timeout,
            )
        layers = await _swap_staged(conn, max(batch_size, _SWAP_BATCH_MIN), timeout=timeout)

        # Chunks added after the last staging pass: embed them under the lock
        async def update(ids: list, embeddings: list[list[float]]) -> None:
            await _update_embeddings(conn, ids, embeddings, timeout=timeout)

        caught_up = await _re_embed_batches(
            conn,
            target,
            start,
            batch_size=batch_size,
            write=update,
            unstaged_only=True,
            timeout=timeout,
            layers="all",
            phase="catch_up",
        )
    for layer, n in caught_up.items():
        layers[layer] = layers.get(layer, 0) + n
    return layers, size


def _chunk_filter(
    after_id, *, layer_name: str | None, unstaged_only: bool
) -> tuple[str, list]:
    """Build the WHERE clause and its arguments for a chunk read (empty when nothing is set)."""
    clauses: list[str] = []
    args: list = []
    if after_id is not None:
        args.append(after_id)
        clauses.append(f"c.id > ${len(args)}")
    if layer_name is not None:
        args.append(layer_name)
        clauses.append(f"c.layer_name = ${len(args)}")
    if unstaged_only:
        clauses.append(f"NOT EXISTS (SELECT 1 FROM {_STAGING} s WHERE s.id = c.id)")
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), args


async def _count_chunks(
    db,
    *,
    layer_name: str | None = None,
    unstaged_only: bool = False,
    timeout: float | None = None,
) -> int:
    """Count the chunks a re-embed would read (takes a pool or a connection)."""
    where, args = _chunk_filter(None, layer_name=layer_name, unstaged_only=unstaged_only)
    return await db.fetchval(
        f"SELECT count(*) FROM strata_vectors.chunks c{where}", *args, timeout=timeout
    )


async def _fetch_chunk_batch(
    db,
    after_id,
    limit: int,
    *,
    layer_name: str | None = None,
    unstaged_only: bool = False,
    timeout: float | None = None,
) -> list:
    """Read the next batch of chunk rows after after_id, in id order (keyset pagination).

    Takes a pool or a connection: both have fetch. Each call is its own short
    statement, so no snapshot or cursor stays open between batches. A timeout
    of None keeps the pool's command timeout.
    """
    where, args = _chunk_filter(after_id, layer_name=layer_name, unstaged_only=unstaged_only)
    args.append(limit)
    return await db.fetch(
        "SELECT c.id, c.content, c.layer_name, c.source_file, c.metadata "
        f"FROM strata_vectors.chunks c{where} ORDER BY c.id LIMIT ${len(args)}",
        *args,
        timeout=timeout,
    )


async def _re_embed_batches(
    db,
    dimensions: int,
    start: float,
    *,
    batch_size: int,
    write,
    layer_name: str | None = None,
    unstaged_only: bool = False,
    timeout: float | None = None,
    **log_ctx,
) -> dict[str, int]:
    """Read chunk rows one batch at a time, embed them and hand the vectors to write.

    Takes a pool or a connection, so a caller can keep every statement on its
    own connection or transaction. write(ids, embeddings) stores the vectors.
    timeout applies to the reads here (None keeps the pool's command timeout).
    log_ctx tags the log lines.

    Returns:
        Number of chunks embedded per layer.
    """
    import time

    from stratakb.config import get_settings
    from stratakb.knowledge.embeddings import embed_texts

    total = await _count_chunks(
        db, layer_name=layer_name, unstaged_only=unstaged_only, timeout=timeout
    )
    logger.info(
        "re_embed_started", **log_ctx, total_chunks=total, model=get_settings().embedding_model
    )

    counts: dict[str, int] = {}
    updated = 0
    batches = 0
    after_id = None
    while True:
        batch = await _fetch_chunk_batch(
            db,
            after_id,
            batch_size,
            layer_name=layer_name,
            unstaged_only=unstaged_only,
            timeout=timeout,
        )
        if not batch:
            break
        embeddings = await embed_texts([_re_embed_text(r) for r in batch], dimensions=dimensions)
        await write([r["id"] for r in batch], embeddings)
        for r in batch:
            counts[r["layer_name"]] = counts.get(r["layer_name"], 0) + 1
        updated += len(batch)
        batches += 1
        after_id = batch[-1]["id"]

        if batches % 5 == 0 or updated >= total:
            logger.info(
                "re_embed_progress",
                **log_ctx,
                done=updated,
                total=total,
                elapsed=round(time.monotonic() - start, 1),
            )

    return counts


async def _swap_staged(conn, batch_size: int, *, timeout: float | None = None) -> dict[str, int]:
    """Copy the staged vectors into the chunks they belong to, batch_size ids at a time.

    Bounded batches keep each statement under timeout (locked_timeout in
    phase 2). A chunk deleted since it was staged matches nothing and is not
    counted.

    Returns:
        Number of chunks updated per layer.
    """
    counts: dict[str, int] = {}
    # Chunk ids come from gen_random_uuid(), which never yields the nil uuid
    after_id = uuid.UUID(int=0)
    while True:
        ids = await conn.fetch(
            f"SELECT id FROM {_STAGING} WHERE id > $1 ORDER BY id LIMIT $2",
            after_id,
            batch_size,
            timeout=timeout,
        )
        if not ids:
            break
        rows = await conn.fetch(
            f"""
            UPDATE strata_vectors.chunks AS c
            SET embedding = s.embedding::vector
            FROM {_STAGING} s
            WHERE s.id = ANY($1::uuid[]) AND c.id = s.id
            RETURNING c.layer_name
            """,
            [r["id"] for r in ids],
            timeout=timeout,
        )
        for r in rows:
            counts[r["layer_name"]] = counts.get(r["layer_name"], 0) + 1
        after_id = ids[-1]["id"]
    return counts


async def vector_column_size(conn, *, timeout: float | None = None) -> int:
    """Return the size of the shared embedding column (-1 when it has none).

    Takes a pool or a connection: both have fetchval.
    """
    return await conn.fetchval(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'strata_vectors.chunks'::regclass AND attname = 'embedding'",
        timeout=timeout,
    )


async def _update_embeddings(
    conn, ids: list, embeddings: list[list[float]], *, timeout: float | None = None
) -> None:
    """Write new embeddings for the given chunk ids in one bulk UPDATE."""
    # Bulk UPDATE using unnest — single round-trip instead of N
    await conn.execute(
        """
        UPDATE strata_vectors.chunks AS c
        SET embedding = v.emb::vector
        FROM unnest($1::uuid[], $2::text[]) AS v(id, emb)
        WHERE c.id = v.id
        """,
        *_vector_params(ids, embeddings),
        timeout=timeout,
    )


def _vector_params(ids: list, embeddings: list[list[float]]) -> tuple[list[str], list[str]]:
    """Encode chunk ids and vectors as the uuid[] and text[] arguments of an unnest."""
    return [str(uid) for uid in ids], [_embedding_to_pgvector(e) for e in embeddings]


def _re_embed_text(row) -> str:
    """Rebuild the text ingestion embedded for a stored chunk row.

    Chunks stored before the prefix left content still start with it; embed those as they are.
    """
    import json

    content = row["content"]
    if content.startswith(f"[{row['layer_name']} >"):
        return content
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return chunk_embed_text(
        row["layer_name"],
        row["source_file"] or "",
        meta if isinstance(meta, dict) else {},
        content,
    )


def _embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert a Python list of floats to pgvector string format '[1.0,2.0,...]'."""
    return "[" + ",".join(str(f) for f in embedding) + "]"
