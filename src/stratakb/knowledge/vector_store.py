"""Vector store operations for strata_vectors.chunks.

Handles inserting, deleting, and querying document chunks with pgvector embeddings.
"""

from __future__ import annotations

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
        )
        for chunk in chunks
    ]
    sql = """
        INSERT INTO strata_vectors.chunks
            (layer_name, content, embedding, metadata, source_file, source_section,
             doc_type, file_id)
        VALUES ($1, $2, $3::vector, $4::jsonb, $5, $6, $7, $8)
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

    If new_dimensions differs from the current column size, the vector
    column is altered first.

    Args:
        database_url: PostgreSQL connection URL.
        layer_name: Layer to re-embed.
        batch_size: Number of chunks to embed per batch.
        new_dimensions: New vector dimensions (0 = auto from config).

    Returns:
        Stats dict with chunks_updated, batches, duration_seconds.
    """
    import time

    from stratakb.config import get_settings
    from stratakb.knowledge.embeddings import embed_texts

    settings = get_settings()
    if not new_dimensions:
        new_dimensions = settings.embedding_dimensions

    pool = await get_pool(database_url)
    start = time.monotonic()

    # Resize vector column if dimensions changed
    current_dim = await pool.fetchval(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'strata_vectors.chunks'::regclass AND attname = 'embedding'"
    )
    if current_dim != new_dimensions:
        logger.info("resizing_vector_column", old=current_dim, new=new_dimensions)
        await pool.execute(
            f"ALTER TABLE strata_vectors.chunks ALTER COLUMN embedding TYPE vector({new_dimensions})"
        )

    # Load all chunk IDs, content and the columns the context prefix is built from
    rows = await pool.fetch(
        "SELECT id, content, layer_name, source_file, metadata "
        "FROM strata_vectors.chunks WHERE layer_name = $1 ORDER BY id",
        layer_name,
    )
    total = len(rows)
    logger.info("re_embed_started", layer=layer_name, total_chunks=total, model=settings.embedding_model)

    updated = 0
    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        texts = [_re_embed_text(r) for r in batch]
        ids = [r["id"] for r in batch]

        embeddings = await embed_texts(texts)

        async with pool.acquire() as conn:
            # Bulk UPDATE using unnest — single round-trip instead of N
            id_array = [str(uid) for uid in ids]
            emb_array = [_embedding_to_pgvector(emb) for emb in embeddings]
            await conn.execute(
                """
                UPDATE strata_vectors.chunks AS c
                SET embedding = v.emb::vector
                FROM unnest($1::uuid[], $2::text[]) AS v(id, emb)
                WHERE c.id = v.id
                """,
                id_array,
                emb_array,
            )
        updated += len(batch)

        if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= total:
            elapsed = time.monotonic() - start
            logger.info(
                "re_embed_progress",
                layer=layer_name,
                done=updated,
                total=total,
                elapsed=round(elapsed, 1),
            )

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
