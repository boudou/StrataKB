"""Vector search retriever for the StrataKB knowledge base.

Embeds a query and performs cosine similarity search against pgvector,
with optional layer and metadata filtering.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from stratakb.core.database import get_pool
from stratakb.core.stamp_watcher import StampWatcher
from stratakb.knowledge.embeddings import embed_texts
from stratakb.rag.cache import QueryCache

if TYPE_CHECKING:
    from stratakb.config import Settings

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


def _clear_cache() -> None:
    # A function, not _cache.clear: _cache is built on first use. The watcher
    # is only checked after _get_cache() returned it.
    assert _cache is not None
    _cache.clear()


# Clears the query cache when any process has changed the layers
_watcher = StampWatcher(
    _clear_cache,
    cleared_event="query_cache_cleared",
    failed_event="query_cache_stamp_failed",
)


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
    # The stored document and the span of its text this chunk holds; None for
    # chunks ingested before documents were stored
    document_id: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    # The clause or section the chunk sits in (" > "-joined); None when the
    # document had no detectable structure or predates the structure map
    clause_path: str | None = None


async def search_chunks(
    database_url: str,
    openai_api_key: str,
    query: str,
    *,
    layers: list[str] | None = None,
    top_k: int = 5,
    metadata_filter: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> list[SearchResult]:
    """Search the knowledge base for chunks relevant to a query.

    Args:
        database_url: PostgreSQL connection URL.
        openai_api_key: OpenAI API key for query embedding.
        query: Natural language search query.
        layers: Optional list of layer names to restrict search to.
        top_k: Number of results to return (default 5, max 50).
        metadata_filter: Optional key-value pairs to filter on metadata JSONB.
        settings: The settings to use. Defaults to the environment's, read here.

    Returns:
        List of SearchResult ordered by relevance (highest score first).
    """
    if not query or not query.strip():
        return []

    top_k = max(1, min(top_k, 50))

    cache = _get_cache()
    if cache:
        await _watcher.check(database_url)
        cached = cache.get(query, layers, top_k, metadata_filter)
        if cached is not None:
            return cached

    # Embed the query using the same model as ingestion
    embeddings = await embed_texts([query], api_key=openai_api_key, settings=settings)
    query_embedding = embeddings[0]

    # Build the SQL query with optional filters
    query_vector = _embedding_to_pgvector(query_embedding)

    if settings is None:
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

    # $1 = embedding, $2 = top_k
    conditions, filter_params = _filter_conditions(layers, metadata_filter, 3)
    params: list[object] = [query_vector, top_k, *filter_params]

    where_clause = _semantic_where(conditions)

    sql = f"""
        SELECT
            id,
            content,
            1 - (embedding <=> $1::vector) AS score,
            layer_name,
            source_file,
            source_section,
            doc_type,
            metadata,
            document_id,
            char_start,
            char_end,
            clause_path
        FROM strata_vectors.chunks
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """

    pool = await get_pool(database_url)
    rows = await _search_fetch(pool, settings, sql, params)
    results = [_row_to_result(row, row["score"], row["score"]) for row in rows]
    # An iterative index scan in relaxed order can return rows slightly out of order
    results.sort(key=lambda r: r.score, reverse=True)

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


def _filter_conditions(
    layers: list[str] | None,
    metadata_filter: dict[str, str] | None,
    start_idx: int,
) -> tuple[list[str], list]:
    """Build the layer and metadata WHERE conditions shared by both search paths.

    Placeholders are numbered from $start_idx; the params follow in that order.
    """
    conditions: list[str] = []
    params: list = []
    idx = start_idx
    if layers:
        placeholders = ", ".join(f"${idx + i}" for i in range(len(layers)))
        conditions.append(f"layer_name IN ({placeholders})")
        params.extend(layers)
        idx += len(layers)
    if metadata_filter:
        for key, value in metadata_filter.items():
            conditions.append(f"metadata->>'{_sanitize_key(key)}' = ${idx}")
            params.append(value)
            idx += 1
    return conditions, params


def _semantic_where(conditions: list[str]) -> str:
    """WHERE clause for cosine ranking, with the shared filter conditions.

    A chunk without an embedding has no cosine distance, so cosine ranking
    leaves it out (keyword search still finds it).
    """
    return "WHERE " + " AND ".join(["embedding IS NOT NULL", *conditions])


def _supports_iterative_scan(extversion: object) -> bool:
    """True when the pgvector extension version is 0.8 or later."""
    if not isinstance(extversion, str):
        return False
    match = re.match(r"(\d+)\.(\d+)", extversion)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (0, 8)


# Set once the missing iterative-scan support has been logged in this process
_iterative_scan_warned = False


async def _search_fetch(pool, settings, sql: str, params: list) -> list:
    """Run one search query in a transaction with the search settings.

    Without iterative index scans (pgvector 0.8+) the HNSW index returns at
    most hnsw.ef_search rows, and a filter applied afterwards can leave fewer
    than the LIMIT asks for.
    """
    global _iterative_scan_warned
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {settings.hnsw_ef_search}")
        await conn.execute(f"SET LOCAL statement_timeout = '{_SEARCH_TIMEOUT}s'")
        version = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        # "hnsw" is a reserved prefix once pgvector loads: never send an unknown setting
        if _supports_iterative_scan(version):
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        elif not _iterative_scan_warned:
            _iterative_scan_warned = True
            logger.warning(
                "hnsw_iterative_scan_unavailable",
                pgvector_version=version if isinstance(version, str) else None,
            )
        return await conn.fetch(sql, *params)


def _row_to_result(row, score: float, similarity: float | None) -> SearchResult:
    """Map a chunk row to a SearchResult with the given score and similarity."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return SearchResult(
        chunk_id=str(row["id"]),
        content=row["content"],
        score=round(float(score), 4),
        layer_name=row["layer_name"],
        source_file=row["source_file"] or "",
        source_section=row["source_section"] or "",
        doc_type=row["doc_type"] or "",
        metadata=meta if isinstance(meta, dict) else {},
        similarity=round(float(similarity), 4) if similarity is not None else None,
        document_id=row.get("document_id"),
        char_start=row.get("char_start"),
        char_end=row.get("char_end"),
        clause_path=row.get("clause_path"),
    )


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


def fuse_rrf(
    rank_lists: Sequence[Sequence[tuple[str, int]]],
    *,
    k: int,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion of any number of rank lists.

    A chunk earns weight / (k + rank) from each list that holds it, counted
    once per list at its first appearance.

    Args:
        rank_lists: Lists of (chunk_id, rank), one per search.
        k: RRF constant.
        weights: One weight per list; None weighs every list 1.0.

    Returns:
        Sorted list of (chunk_id, fused_score) highest first; ties keep the
        order in which the chunks were first seen.

    Raises:
        ValueError: If weights does not hold one weight per list.
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    elif len(weights) != len(rank_lists):
        raise ValueError(
            f"fuse_rrf got {len(weights)} weights for {len(rank_lists)} rank lists"
        )
    scores: dict[str, float] = {}
    for ranks, weight in zip(rank_lists, weights, strict=True):
        seen: set[str] = set()
        for chunk_id, rank in ranks:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
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
    Rule: keep the keyword ranks 1..n (n = slots, at most top_k - 1, so the
    fused #1 always stays), then fill the other top_k - n places with the best
    unprotected fused entries. Every entry keeps its own fused score.

    Returns:
        At most top_k (chunk_id, fused_score), highest first.
    """
    n = max(0, min(slots, top_k - 1))
    protected = {cid for cid, _ in sorted(keyword, key=lambda x: x[1])[:n]}
    room = top_k - len(protected)
    kept = []
    for cid, score in fused:
        if cid in protected:
            kept.append((cid, score))
        elif room > 0:
            kept.append((cid, score))
            room -= 1
    return kept


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
    conditions, filter_params = _filter_conditions(layers, metadata_filter, 3)
    params: list = [query_vector, over_retrieve, *filter_params]
    param_idx = 3 + len(filter_params)

    where = _semantic_where(conditions)

    params.append(query_text)
    if _WEBSEARCH_SYNTAX.search(query_text):
        tsquery = f"websearch_to_tsquery('english', ${param_idx})"
    else:
        # plainto_tsquery stems, drops stopwords and quotes each lexeme; ' & ' -> ' | '
        # turns its AND into an OR, since an AND of every word matches almost nothing
        tsquery = f"replace(plainto_tsquery('english', ${param_idx})::text, ' & ', ' | ')::tsquery"

    ft_where = f"WHERE tsv @@ q AND {' AND '.join(conditions)}" if conditions else "WHERE tsv @@ q"

    # The index scan may return rows in relaxed order; "distance + 0" makes the
    # ranking a real sort instead of reusing the index order
    sql = f"""
        WITH semantic AS (
            SELECT id, embedding <=> $1::vector AS distance
            FROM strata_vectors.chunks
            {where}
            ORDER BY distance
            LIMIT $2
        ),
        fulltext AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, q) DESC, id) AS rank
            FROM strata_vectors.chunks, (SELECT {tsquery} AS q) AS query
            {ft_where}
            ORDER BY rank
            LIMIT $2
        )
        SELECT 'semantic' AS source, id, ROW_NUMBER() OVER (ORDER BY distance + 0, id) AS rank
        FROM semantic
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

    rows = await _search_fetch(pool, settings, sql, params)

    semantic_ranks = [(str(r["id"]), r["rank"]) for r in rows if r["source"] == "semantic"]
    keyword_ranks = [(str(r["id"]), r["rank"]) for r in rows if r["source"] == "fulltext"]

    fused = fuse_rrf(
        [semantic_ranks, keyword_ranks],
        k=settings.hybrid_rrf_k,
        weights=[settings.hybrid_semantic_weight, settings.hybrid_keyword_weight],
    )

    kept = _reserve_keyword_slots(fused, keyword_ranks, top_k, settings.hybrid_keyword_slots)
    top_ids = [f[0] for f in kept]
    fused_scores = {f[0]: f[1] for f in fused}

    if not top_ids:
        return []

    uuid_ids = [uuid.UUID(tid) for tid in top_ids]
    chunk_sql = """
        SELECT id, content, layer_name, source_file, source_section, doc_type, metadata,
               document_id, char_start, char_end, clause_path,
               1 - (embedding <=> $2::vector) AS similarity
        FROM strata_vectors.chunks
        WHERE id = ANY($1::uuid[])
    """
    async with pool.acquire() as conn:
        chunk_rows = await conn.fetch(chunk_sql, uuid_ids, query_vector)

    results = [
        _row_to_result(row, fused_scores.get(str(row["id"]), 0.0), row["similarity"])
        for row in chunk_rows
    ]
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
