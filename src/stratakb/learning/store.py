"""CRUD operations for the strata_learning schema.

Provides async methods for storing interactions, feedback,
and retrieving learned patterns (rewrites, synonyms, affinities).
All methods are safe to call when the schema doesn't exist — they
return empty/None results (read-optional principle).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import structlog
from cachetools import TTLCache

from stratakb.core.database import get_pool

logger = structlog.get_logger()

_rewrite_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)
_synonym_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

# Mined rows are written at this confidence and applied from it. Rows below it
# came from an older rule with no evidence bar.
MINED_SYNONYM_CONFIDENCE = 0.7


def _decode_results(value):
    """An interaction's results column: asyncpg returns JSONB as text."""
    return json.loads(value) if isinstance(value, str) else value


@dataclass
class LearnedRewrite:
    """A learned query rewrite pattern."""

    pattern: str
    optimized_rewrite: str
    layers: list[str]
    success_rate: float
    total_count: int

    @property
    def queries(self) -> list[str]:
        """The search queries to run: optimized_rewrite holds a JSON array or one plain query."""
        try:
            parsed = json.loads(self.optimized_rewrite)
        except ValueError:
            return [self.optimized_rewrite]
        if isinstance(parsed, list) and parsed and all(isinstance(q, str) for q in parsed):
            return parsed
        return [self.optimized_rewrite]


class LearningStore:
    """Async CRUD for strata_learning tables."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    async def _fetch(self, sql: str, *args) -> list[dict]:
        """Execute a SELECT and return rows as dicts. Returns [] on error."""
        try:
            pool = await get_pool(self.database_url)
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *args)
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("learning_store_fetch_failed", error_type=type(e).__name__)
            return []

    async def _execute(self, sql: str, *args) -> str | None:
        """Execute a write statement and return its status ("UPDATE 3").

        Returns None if it fails, for example when the schema is unavailable.
        """
        try:
            pool = await get_pool(self.database_url)
            async with pool.acquire() as conn:
                return await conn.execute(sql, *args)
        except Exception as e:
            logger.warning("learning_store_execute_failed", error_type=type(e).__name__)
            return None

    async def _transaction(self, *statements: tuple) -> list[str] | None:
        """Run (sql, *args) statements in one transaction and return their statuses.

        Returns None if it fails, and then none of them took effect.
        """
        try:
            pool = await get_pool(self.database_url)
            async with pool.acquire() as conn, conn.transaction():
                return [await conn.execute(sql, *args) for sql, *args in statements]
        except Exception as e:
            logger.warning("learning_store_transaction_failed", error_type=type(e).__name__)
            return None

    # ── Sessions ──────────────────────────────────────────────

    async def get_or_create_session(
        self, user_id: str | None, timeout_minutes: int = 30
    ) -> dict | None:
        """Get the active session or create a new one.

        A session is 'active' if it was updated within timeout_minutes.
        """
        if user_id:
            rows = await self._fetch(
                """
                SELECT id, session_start, query_count
                FROM strata_learning.query_sessions
                WHERE user_id = $1
                  AND session_end > now() - interval '1 minute' * $2
                ORDER BY session_start DESC LIMIT 1
                """,
                user_id,
                timeout_minutes,
            )
            if rows:
                return rows[0]

        # Create new session
        rows = await self._fetch(
            """
            INSERT INTO strata_learning.query_sessions (user_id, session_end)
            VALUES ($1, now())
            RETURNING id, session_start, query_count
            """,
            user_id,
        )
        return rows[0] if rows else None

    async def update_session(self, session_id: str) -> None:
        """Increment query_count and update session_end."""
        await self._execute(
            """
            UPDATE strata_learning.query_sessions
            SET query_count = query_count + 1,
                session_end = now()
            WHERE id = $1
            """,
            session_id,
        )

    # ── Interactions ──────────────────────────────────────────

    async def store_interaction(
        self,
        *,
        session_id: str | None,
        query_text: str,
        rewritten_queries: list[str] | None = None,
        layers_detected: list[str] | None = None,
        layers_searched: list[str] | None = None,
        results: list[dict] | None = None,
        result_count: int = 0,
        confidence_level: str | None = None,
        confidence_score: float | None = None,
        tool_used: str | None = None,
        response_time_ms: int | None = None,
    ) -> dict | None:
        """Record a query interaction."""
        rows = await self._fetch(
            """
            INSERT INTO strata_learning.interaction_events
                (session_id, query_text, rewritten_queries,
                 layers_detected, layers_searched, results, result_count,
                 confidence_level, confidence_score, tool_used, response_time_ms)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
            RETURNING id, created_at
            """,
            session_id,
            query_text,
            json.dumps(rewritten_queries) if rewritten_queries else None,
            layers_detected,
            layers_searched,
            json.dumps(results or []),
            result_count,
            confidence_level,
            confidence_score,
            tool_used,
            response_time_ms,
        )
        return rows[0] if rows else None

    async def get_recent_interactions(
        self, session_id: str, limit: int = 5
    ) -> list[dict]:
        """Get recent interactions in a session (for signal detection)."""
        rows = await self._fetch(
            """
            SELECT id, query_text, layers_detected, layers_searched,
                   confidence_level, confidence_score, results, created_at
            FROM strata_learning.interaction_events
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
        for row in rows:
            row["results"] = _decode_results(row.get("results"))
        return rows

    async def get_interaction_results(self, interaction_id: str) -> list[dict] | None:
        """The stored results of one interaction, in order; None if it is unknown."""
        rows = await self._fetch(
            "SELECT results FROM strata_learning.interaction_events WHERE id = $1",
            interaction_id,
        )
        if not rows:
            return None
        return _decode_results(rows[0]["results"]) or []

    # ── Feedback ──────────────────────────────────────────────

    async def store_feedback(
        self,
        interaction_id: str,
        signal_type: str,
        signal_name: str,
        signal_value: float,
        metadata: dict | None = None,
    ) -> bool:
        """Store a feedback signal. Returns False if not stored (e.g. unknown interaction)."""
        status = await self._execute(
            """
            INSERT INTO strata_learning.feedback_signals
                (interaction_id, signal_type, signal_name, signal_value, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            interaction_id,
            signal_type,
            signal_name,
            signal_value,
            json.dumps(metadata) if metadata else None,
        )
        return status is not None

    # ── Synonyms ──────────────────────────────────────────────

    async def find_synonyms(self, query: str) -> list[dict]:
        """Find known synonyms for terms in the query."""
        cache_key = query.lower().strip()
        if cache_key in _synonym_cache:
            return _synonym_cache[cache_key]

        # Extract candidate terms (words and digit sequences)
        terms = set(re.findall(r"\b[\w]+\b", query.lower()))
        if not terms:
            _synonym_cache[cache_key] = []
            return []

        placeholders = ", ".join(f"${i+1}" for i in range(len(terms)))
        result = await self._fetch(
            f"""
            SELECT term, synonym, layer, confidence
            FROM strata_learning.query_synonyms
            WHERE lower(term) IN ({placeholders})
              AND confidence >= ${len(terms) + 1}
            ORDER BY confidence DESC
            """,
            *list(terms),
            MINED_SYNONYM_CONFIDENCE,
        )
        _synonym_cache[cache_key] = result
        return result

    async def upsert_synonym(
        self,
        term: str,
        synonym: str,
        layer: str | None = None,
        source: str = "pattern_mining",
        confidence: float = 0.5,
        usage_count: int = 0,
    ) -> None:
        """Insert or update a synonym mapping.

        Updates first: UNIQUE(term, synonym, layer) treats NULL layers as
        distinct, so ON CONFLICT alone would add a row on every call. Keeps the
        higher of the old and new values, so repeating a call changes nothing.
        """
        _synonym_cache.clear()
        status = await self._execute(
            """
            UPDATE strata_learning.query_synonyms
            SET confidence = GREATEST(confidence, $4),
                usage_count = GREATEST(usage_count, $5)
            WHERE term = $1 AND synonym = $2 AND layer IS NOT DISTINCT FROM $3
            """,
            term,
            synonym,
            layer,
            confidence,
            usage_count,
        )
        if status == "UPDATE 0":
            await self._execute(
                """
                INSERT INTO strata_learning.query_synonyms
                    (term, synonym, layer, source, confidence, usage_count)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (term, synonym, layer) DO NOTHING
                """,
                term,
                synonym,
                layer,
                source,
                confidence,
                usage_count,
            )

    async def purge_weak_mined_synonyms(self) -> None:
        """Delete mined rows below the current bar: only the older rule wrote them."""
        _synonym_cache.clear()
        await self._execute(
            """
            DELETE FROM strata_learning.query_synonyms
            WHERE source = 'pattern_mining' AND confidence < $1
            """,
            MINED_SYNONYM_CONFIDENCE,
        )

    # ── Learned Rewrites ──────────────────────────────────────

    async def find_rewrites(
        self, query: str, min_success_rate: float = 0.7
    ) -> LearnedRewrite | None:
        """Find the best learned rewrite for a query."""
        cache_key = (query.lower().strip(), min_success_rate)
        if cache_key in _rewrite_cache:
            return _rewrite_cache[cache_key]

        rows = await self._fetch(
            """
            SELECT pattern, optimized_rewrite, layers, success_rate, total_count
            FROM strata_learning.learned_rewrites
            WHERE success_rate >= $1
              AND (
                  pattern_type = 'exact' AND lower(pattern) = lower($2)
                  OR pattern_type = 'keyword' AND similarity(pattern, $2) > 0.3
              )
            ORDER BY success_rate DESC, total_count DESC
            LIMIT 1
            """,
            min_success_rate,
            query,
        )
        if not rows:
            _rewrite_cache[cache_key] = None
            return None

        row = rows[0]
        result = LearnedRewrite(
            pattern=row["pattern"],
            optimized_rewrite=row["optimized_rewrite"],
            layers=row["layers"] or [],
            success_rate=row["success_rate"],
            total_count=row["total_count"],
        )
        _rewrite_cache[cache_key] = result
        return result

    async def upsert_learned_rewrite(
        self,
        pattern: str,
        pattern_type: str,
        optimized_rewrite: str,
        layers: list[str] | None = None,
        success_count: int = 0,
        total_count: int = 0,
    ) -> None:
        """Insert or update a learned rewrite."""
        _rewrite_cache.clear()
        success_rate = success_count / total_count if total_count > 0 else 0.0
        await self._execute(
            """
            INSERT INTO strata_learning.learned_rewrites
                (pattern, pattern_type, optimized_rewrite, layers,
                 success_count, total_count, success_rate, last_used, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, now(), now())
            ON CONFLICT (pattern, pattern_type) DO UPDATE SET
                optimized_rewrite = EXCLUDED.optimized_rewrite,
                success_count = EXCLUDED.success_count,
                total_count = EXCLUDED.total_count,
                success_rate = EXCLUDED.success_rate,
                last_used = now(),
                updated_at = now()
            """,
            pattern,
            pattern_type,
            optimized_rewrite,
            layers,
            success_count,
            total_count,
            success_rate,
        )

    # ── Layer Affinities ──────────────────────────────────────

    async def find_layer_affinities(
        self, query: str, min_affinity: float = 0.6
    ) -> list[str]:
        """Find layers with learned affinity for a query pattern."""
        rows = await self._fetch(
            """
            SELECT layer_name, affinity_score
            FROM strata_learning.layer_affinity_scores
            WHERE similarity(query_pattern, $1) > 0.3
              AND affinity_score >= $2
            ORDER BY affinity_score DESC
            """,
            query,
            min_affinity,
        )
        return [r["layer_name"] for r in rows]

    async def upsert_layer_affinity(
        self,
        query_pattern: str,
        layer_name: str,
        affinity_score: float,
        sample_count: int,
    ) -> None:
        """Insert or update how strongly a query pattern belongs to a layer."""
        await self._execute(
            """
            INSERT INTO strata_learning.layer_affinity_scores
                (query_pattern, layer_name, affinity_score, sample_count, last_updated)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (query_pattern, layer_name) DO UPDATE SET
                affinity_score = EXCLUDED.affinity_score,
                sample_count = EXCLUDED.sample_count,
                last_updated = now()
            """,
            query_pattern,
            layer_name,
            affinity_score,
            sample_count,
        )

    # ── Stats ─────────────────────────────────────────────────

    async def get_stats(self, layer: str | None = None) -> dict:
        """Get learning system statistics. Optionally filter by layer."""
        stats: dict = {}

        if layer:
            rows = await self._fetch(
                "SELECT count(*) as n FROM strata_learning.interaction_events WHERE $1 = ANY(layers_searched)",
                layer,
            )
        else:
            rows = await self._fetch(
                "SELECT count(*) as n FROM strata_learning.interaction_events"
            )
        stats["total_interactions"] = rows[0]["n"] if rows else 0

        if layer:
            rows = await self._fetch(
                "SELECT count(*) as n FROM strata_learning.learned_rewrites WHERE success_rate >= 0.5 AND $1 = ANY(layers)",
                layer,
            )
        else:
            rows = await self._fetch(
                "SELECT count(*) as n FROM strata_learning.learned_rewrites WHERE success_rate >= 0.5"
            )
        stats["active_rewrites"] = rows[0]["n"] if rows else 0

        if layer:
            rows = await self._fetch(
                "SELECT count(*) as n FROM strata_learning.query_synonyms WHERE layer = $1",
                layer,
            )
        else:
            rows = await self._fetch(
                "SELECT count(*) as n FROM strata_learning.query_synonyms"
            )
        stats["synonyms"] = rows[0]["n"] if rows else 0

        if layer:
            rows = await self._fetch(
                """
                SELECT term, synonym, layer, confidence, usage_count
                FROM strata_learning.query_synonyms
                WHERE layer = $1
                ORDER BY usage_count DESC LIMIT 10
                """,
                layer,
            )
        else:
            rows = await self._fetch(
                """
                SELECT term, synonym, layer, confidence, usage_count
                FROM strata_learning.query_synonyms
                ORDER BY usage_count DESC LIMIT 10
                """
            )
        stats["top_synonyms"] = rows

        return stats

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def apply_synonyms(query: str, synonyms: list[dict]) -> str:
        """Enrich a query by appending synonym expansions.

        Preserves the original query and appends synonym info so
        vector search can match on both original and expanded terms.
        """
        if not synonyms:
            return query

        expansions = []
        for s in synonyms:
            term = s["term"]
            synonym = s["synonym"]
            if (
                term.lower() in query.lower()
                and synonym.lower() not in query.lower()
                and synonym not in expansions  # one pair can be stored under several layers
            ):
                expansions.append(synonym)

        if expansions:
            return f"{query} ({', '.join(expansions)})"
        return query

    async def reset(self, layer: str | None = None) -> dict | None:
        """Reset learning data. If layer is given (even ""), only that layer.

        Runs as one transaction: returns None if it fails, and then nothing
        was deleted. A layer reset also returns how many rows it deleted.
        """
        _rewrite_cache.clear()
        _synonym_cache.clear()
        if layer is None:
            statuses = await self._transaction((
                "TRUNCATE strata_learning.feedback_signals,"
                " strata_learning.interaction_events,"
                " strata_learning.query_sessions,"
                " strata_learning.learned_rewrites,"
                " strata_learning.query_synonyms,"
                " strata_learning.layer_affinity_scores CASCADE",
            ))
            return None if statuses is None else {"reset": "all"}
        statuses = await self._transaction(
            ("DELETE FROM strata_learning.learned_rewrites WHERE $1 = ANY(layers)", layer),
            ("DELETE FROM strata_learning.query_synonyms WHERE layer = $1", layer),
            ("DELETE FROM strata_learning.layer_affinity_scores WHERE layer_name = $1", layer),
        )
        if statuses is None:
            return None
        deleted = sum(int(status.split()[-1]) for status in statuses)
        return {"reset": "layer", "layer": layer, "deleted": deleted}
