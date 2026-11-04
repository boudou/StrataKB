"""Vector search retriever for the StrataKB knowledge base.

Embeds a query and performs cosine similarity search against pgvector,
with optional layer and metadata filtering.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

import structlog

from stratakb.core.database import fetch_layers_stamp, get_pool
from stratakb.knowledge.embeddings import embed_texts
from stratakb.rag.cache import QueryCache

logger = structlog.get_logger()

# Server-side cap (seconds) on each search query, below the pool's 30 s command_timeout
_SEARCH_TIMEOUT = 10

_cache: QueryCache | None = None


def _get_cache() -> QueryCache | None:
    global _cache
    if _cache is not None:
        return _cache
    try:
        from stratakb.config import get_settings
        s = get_settings()
        if s.cache_enabled:
            _cache = QueryCache(maxsize=s.cache_maxsize, ttl=s.cache_ttl_seconds)
            return _cache
    except Exception:
        pass
    return None


# The layers stamp is read at most this often (seconds) per database URL
_STAMP_INTERVAL = 2.0
_stamps: dict[str, tuple[int, datetime | None]] = {}
_checked_at: dict[str, float] = {}


async def _maybe_clear(database_url: str) -> None:
    """Clear the query cache when any process has changed the layers."""
    now = time.monotonic()
    last = _checked_at.get(database_url)
    if last is not None and now - last < _STAMP_INTERVAL:
        return
    # Set before the await: concurrent searches skip, a failing DB is retried every 2 s
    _checked_at[database_url] = now
    try:
        stamp = await fetch_layers_stamp(database_url)
    except Exception as e:
        logger.warning("query_cache_stamp_failed", error_type=type(e).__name__)
        return
    previous = _stamps.get(database_url)
    _stamps[database_url] = stamp
    if previous is not None and stamp != previous:
        cache = _get_cache()
        if cache:
            cache.clear()
        logger.info("query_cache_cleared")


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk_id: str  # UUID from strata_vectors.chunks.id
    content: str
    score: float
    layer_name: str
    source_file: str
    source_section: str
    doc_type: str
    metadata: dict = field(default_factory=dict)
    # Query/chunk cosine similarity (-1..1); None when the chunk has no embedding
    similarity: float | None = None


async def search_chunks(
    database_url: str,
    openai_api_key: str,
    query: str,
    *,
    layers: list[str] | None = None,
    top_k: int = 5,
    metadata_filter: dict[str, str] | None = None,
) -> list[SearchResult]:
    """Search the knowledge base for chunks relevant to a query.

    Args:
        database_url: PostgreSQL connection URL.
        openai_api_key: OpenAI API key for query embedding.
        query: Natural language search query.
        layers: Optional list of layer names to restrict search to.
        top_k: Number of results to return (default 5, max 50).
        metadata_filter: Optional key-value pairs to filter on metadata JSONB.

    Returns:
        List of SearchResult ordered by relevance (highest score first).
    """
    if not query or not query.strip():
        return []

    top_k = max(1, min(top_k, 50))

    cache = _get_cache()
    if cache:
        await _maybe_clear(database_url)
        cached = cache.get(query, layers, top_k, metadata_filter)
        if cached is not None:
            return cached

    # Embed the query using the same model as ingestion
    embeddings = await embed_texts([query], api_key=openai_api_key)
    query_embedding = embeddings[0]

    # Build the SQL query with optional filters
    query_vector = _embedding_to_pgvector(query_embedding)

    from stratakb.config import get_settings
    settings = get_settings()

    if settings.hybrid_search_enabled:
        pool = await get_pool(database_url)
        results = await _hybrid_search(
            pool, query, query_vector, layers, top_k, metadata_filter, settings
        )
        if cache:
            cache.put(query, layers, top_k, results, metadata_filter)
        return results

    conditions: list[str] = []
    params: list[object] = [query_vector, top_k]
    param_idx = 3  # $1 = embedding, $2 = top_k

    if layers:
        placeholders = ", ".join(f"${param_idx + i}" for i in range(len(layers)))
        conditions.append(f"layer_name IN ({placeholders})")
        params.extend(layers)
        param_idx += len(layers)

    if metadata_filter:
        for key, value in metadata_filter.items():
            conditions.append(f"metadata->>'{_sanitize_key(key)}' = ${param_idx}")
            params.append(value)
            param_idx += 1

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT
            id,
            content,
            1 - (embedding <=> $1::vector) AS score,
            layer_name,
            source_file,
            source_section,
            doc_type,
            metadata
        FROM strata_vectors.chunks
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """

    pool = await get_pool(database_url)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL hnsw.ef_search = {settings.hnsw_ef_search}")
            await conn.execute(f"SET LOCAL statement_timeout = '{_SEARCH_TIMEOUT}s'")
            rows = await conn.fetch(sql, *params)

    results = []
    for row in rows:
        import json

        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)

        results.append(
            SearchResult(
                chunk_id=str(row["id"]),
                content=row["content"],
                score=round(float(row["score"]), 4),
                layer_name=row["layer_name"],
                source_file=row["source_file"] or "",
                source_section=row["source_section"] or "",
                doc_type=row["doc_type"] or "",
                metadata=meta if isinstance(meta, dict) else {},
                similarity=round(float(row["score"]), 4),
            )
        )

    if cache:
        cache.put(query, layers, top_k, results, metadata_filter)

    logger.info(
        "vector_search_complete",
        query_length=len(query),
        layers=layers,
        metadata_filter=metadata_filter,
        top_k=top_k,
        results_count=len(results),
    )

    return results


def _embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert a Python list of floats to pgvector string format."""
    return "[" + ",".join(str(f) for f in embedding) + "]"


def _sanitize_key(key: str) -> str:
    """Sanitize a metadata key to prevent SQL injection.

    Only allows alphanumeric characters and underscores.
    """
    return "".join(c for c in key if c.isalnum() or c == "_")


# Quoted phrases, OR and -exclusion: run the query through websearch_to_tsquery as written
_WEBSEARCH_SYNTAX = re.compile(r'"|\bOR\b|(?:^|\s)-\w')


def _fuse_rrf(
    semantic: list[tuple[str, int]],
    keyword: list[tuple[str, int]],
    k: int = 60,
    sem_weight: float = 0.7,
    kw_weight: float = 0.3,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of semantic and keyword result ranks.

    Args:
        semantic: List of (chunk_id, rank) from vector search.
        keyword: List of (chunk_id, rank) from full-text search.
        k: RRF constant (default 60).
        sem_weight: Weight for semantic results.
        kw_weight: Weight for keyword results.

    Returns:
        Sorted list of (chunk_id, fused_score) highest first.
    """
    scores: dict[str, float] = {}
    for chunk_id, rank in semantic:
        scores[chunk_id] = scores.get(chunk_id, 0) + sem_weight / (k + rank)
    for chunk_id, rank in keyword:
        scores[chunk_id] = scores.get(chunk_id, 0) + kw_weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _reserve_keyword_slots(
    fused: list[tuple[str, float]],
    keyword: list[tuple[str, int]],
    top_k: int,
    slots: int,
) -> list[tuple[str, float]]:
    """Keep the best full-text hits in the top_k fused results.

    A chunk only the full-text leg found scores below almost every vector
    rank, so an exact term the embedding misses would never be returned.
    The keyword ranks 1..n (n = slots, at most top_k - 1, so the fused #1
    always stays) each replace the lowest-scored unprotected entry when they
    are missing, and keep their own fused score.

    Returns:
        At most top_k (chunk_id, fused_score), highest first.
    """
    n = max(0, min(slots, top_k - 1))
    chosen = list(fused[:top_k])
    protected = [cid for cid, _ in sorted(keyword, key=lambda x: x[1])[:n]]
    present = {cid for cid, _ in chosen}
    missing = [cid for cid in protected if cid not in present]
    if not missing:
        return chosen

    scores = dict(fused)
    keep = set(protected)
    for cid in missing:
        # chosen is sorted highest first: drop the last entry that is not protected
        drop = max(i for i, (c, _) in enumerate(chosen) if c not in keep)
        chosen.pop(drop)
        chosen.append((cid, scores[cid]))
    return sorted(chosen, key=lambda x: x[1], reverse=True)


def _build_hybrid_sql(
    query_vector: str,
    query_text: str,
    layers: list[str] | None,
    top_k: int,
    metadata_filter: dict[str, str] | None = None,
) -> tuple[str, list]:
    """Build a hybrid search SQL query combining vector + full-text.

    Returns (sql, params) tuple.
    """
    over_retrieve = top_k * 3
    conditions: list[str] = []
    params: list = [query_vector, over_retrieve]
    param_idx = 3

    if layers:
        placeholders = ", ".join(f"${param_idx + i}" for i in range(len(layers)))
        conditions.append(f"layer_name IN ({placeholders})")
        params.extend(layers)
        param_idx += len(layers)

    if metadata_filter:
        for key, value in metadata_filter.items():
            conditions.append(f"metadata->>'{_sanitize_key(key)}' = ${param_idx}")
            params.append(value)
            param_idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    params.append(query_text)
    if _WEBSEARCH_SYNTAX.search(query_text):
        tsquery = f"websearch_to_tsquery('english', ${param_idx})"
    else:
        # plainto_tsquery stems, drops stopwords and quotes each lexeme; ' & ' -> ' | '
        # turns its AND into an OR, since an AND of every word matches almost nothing
        tsquery = f"replace(plainto_tsquery('english', ${param_idx})::text, ' & ', ' | ')::tsquery"

    ft_where = f"WHERE tsv @@ q AND {' AND '.join(conditions)}" if conditions else "WHERE tsv @@ q"

    sql = f"""
        WITH semantic AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
            FROM strata_vectors.chunks
            {where}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        ),
        fulltext AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, q) DESC, id) AS rank
            FROM strata_vectors.chunks, (SELECT {tsquery} AS q) AS query
            {ft_where}
            ORDER BY rank
            LIMIT $2
        )
        SELECT 'semantic' AS source, id, rank FROM semantic
        UNION ALL
        SELECT 'fulltext' AS source, id, rank FROM fulltext
    """
    return sql, params


async def _hybrid_search(
    pool, query: str, query_vector: str,
    layers: list[str] | None, top_k: int,
    metadata_filter: dict[str, str] | None,
    settings,
) -> list[SearchResult]:
    """Execute hybrid search with RRF fusion."""
    sql, params = _build_hybrid_sql(query_vector, query, layers, top_k, metadata_filter)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL hnsw.ef_search = {settings.hnsw_ef_search}")
            await conn.execute(f"SET LOCAL statement_timeout = '{_SEARCH_TIMEOUT}s'")
            rows = await conn.fetch(sql, *params)

    semantic_ranks = [(str(r["id"]), r["rank"]) for r in rows if r["source"] == "semantic"]
    keyword_ranks = [(str(r["id"]), r["rank"]) for r in rows if r["source"] == "fulltext"]

    fused = _fuse_rrf(
        semantic_ranks, keyword_ranks,
        k=settings.hybrid_rrf_k,
        sem_weight=settings.hybrid_semantic_weight,
        kw_weight=settings.hybrid_keyword_weight,
    )

    kept = _reserve_keyword_slots(fused, keyword_ranks, top_k, settings.hybrid_keyword_slots)
    top_ids = [f[0] for f in kept]
    fused_scores = {f[0]: f[1] for f in fused}

    if not top_ids:
        return []

    import uuid as _uuid
    uuid_ids = [_uuid.UUID(tid) for tid in top_ids]
    chunk_sql = """
        SELECT id, content, layer_name, source_file, source_section, doc_type, metadata,
               1 - (embedding <=> $2::vector) AS similarity
        FROM strata_vectors.chunks
        WHERE id = ANY($1::uuid[])
    """
    async with pool.acquire() as conn:
        chunk_rows = await conn.fetch(chunk_sql, uuid_ids, query_vector)

    import json
    results = []
    for row in chunk_rows:
        chunk_id = str(row["id"])
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        results.append(SearchResult(
            chunk_id=chunk_id,
            content=row["content"],
            score=round(fused_scores.get(chunk_id, 0.0), 4),
            layer_name=row["layer_name"],
            source_file=row["source_file"] or "",
            source_section=row["source_section"] or "",
            doc_type=row["doc_type"] or "",
            metadata=meta if isinstance(meta, dict) else {},
            similarity=(
                round(float(row["similarity"]), 4)
                if row["similarity"] is not None else None
            ),
        ))

    results.sort(key=lambda r: r.score, reverse=True)

    # Normalize RRF scores to 0-1 for ranking and display only: the top result is
    # always 1.0, so confidence is judged on `similarity`, never on this score.
    if results:
        max_score = results[0].score
        if max_score > 0:
            for r in results:
                r.score = round(r.score / max_score, 4)

    logger.info(
        "hybrid_search_complete",
        query_length=len(query),
        layers=layers,
        semantic_results=len(semantic_ranks),
        keyword_results=len(keyword_ranks),
        keyword_only_results=len(set(top_ids) - {cid for cid, _ in semantic_ranks}),
        fused_results=len(results),
    )
    return results
