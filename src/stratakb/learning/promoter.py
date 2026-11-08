"""Pattern mining and promotion engine.

Analyzes interaction history and feedback signals to promote
successful query patterns into learned_rewrites, synonyms and
layer affinities.

Rewrites, synonyms and affinities are recomputed from the lookback
window on every run, so running the promoter twice never counts the
same evidence twice.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import structlog

from stratakb.learning.signals import REFORMULATION_MIN_SIMILARITY, content_similarity
from stratakb.learning.store import MINED_SYNONYM_CONFIDENCE, LearningStore
from stratakb.learning.synonyms import extract_synonym_candidates

logger = structlog.get_logger()

# One reformulation can be a coincidence; a pair is stored from the second on
MIN_SYNONYM_SIGHTINGS = 2

# The interaction just before ie in the same session (the query ie reformulated)
_PREVIOUS_INTERACTION = """
    JOIN LATERAL (
        SELECT p.query_text, p.confidence_level
        FROM strata_learning.interaction_events p
        WHERE p.session_id = ie.session_id AND p.created_at < ie.created_at
        ORDER BY p.created_at DESC
        LIMIT 1
    ) prev ON true
"""


def _normalize(query: str) -> str:
    """The pattern key for a question: trimmed and lowercased."""
    return query.strip().lower()


def _decode_queries(value) -> list[str] | None:
    """Decode rewritten_queries (JSONB, which asyncpg returns as text)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, list) and value and all(isinstance(q, str) for q in value):
        return value
    return None


class Promoter:
    """Mines feedback data and promotes successful patterns."""

    def __init__(self, store: LearningStore):
        self.store = store

    async def run(self, lookback_days: int = 30) -> dict:
        """Run the full promotion cycle.

        Returns summary of what was promoted.
        """
        results = {
            "rewrites_promoted": 0,
            "synonyms_mined": 0,
            "affinities_updated": 0,
            "rewrites_decayed": 0,
        }

        results["rewrites_promoted"] = await self.promote_rewrites(lookback_days)
        results["synonyms_mined"] = await self.mine_synonyms(lookback_days)
        results["affinities_updated"] = await self.update_layer_affinities(lookback_days)
        results["rewrites_decayed"] = await self.decay_stale_rewrites()

        logger.info("promotion_complete", **results)
        return results

    async def promote_rewrites(self, lookback_days: int = 30) -> int:
        """Promote query rewrites that led to good results.

        Evidence for a question:
        1. An ask marked helpful: its rewritten queries worked.
        2. A reformulation that got a high-confidence result right after the
           question did not: the reformulation's queries work for the question.

        success_count is the evidence for the question's most common rewrite;
        total_count is how often the question was asked, so asks that led
        nowhere lower the success rate.
        """
        rows = await self.store._fetch(
            f"""
            SELECT ie.query_text AS question, ie.rewritten_queries,
                   NULL AS fallback_query, ie.layers_searched AS layers, ie.created_at
            FROM strata_learning.interaction_events ie
            WHERE EXISTS (
                SELECT 1 FROM strata_learning.feedback_signals fs
                WHERE fs.interaction_id = ie.id AND fs.signal_name = 'helpful'
              )
              AND ie.tool_used = 'strata_ask'
              AND ie.created_at > now() - interval '1 day' * $1
            UNION ALL
            SELECT prev.query_text, ie.rewritten_queries,
                   ie.query_text, ie.layers_searched, ie.created_at
            FROM strata_learning.feedback_signals fs
            JOIN strata_learning.interaction_events ie ON ie.id = fs.interaction_id
            {_PREVIOUS_INTERACTION}
            WHERE fs.signal_name = 'reformulation'
              AND ie.confidence_level = 'high'
              AND prev.confidence_level IS DISTINCT FROM 'high'
              AND ie.created_at > now() - interval '1 day' * $1
            ORDER BY created_at
            """,
            lookback_days,
        )

        evidence: dict[str, Counter] = defaultdict(Counter)
        layers: dict[tuple[str, str], list[str] | None] = {}
        for row in rows:
            pattern = _normalize(row["question"])
            queries = _decode_queries(row["rewritten_queries"])
            if not queries and row["fallback_query"]:
                queries = [row["fallback_query"]]
            # A "rewrite" that is the question itself teaches nothing
            if not queries or [_normalize(q) for q in queries] == [pattern]:
                continue
            rewrite = json.dumps(queries)
            evidence[pattern][rewrite] += 1
            layers[(pattern, rewrite)] = row["layers"]  # rows are oldest first: keep the latest

        if not evidence:
            return 0

        asked = await self.store._fetch(
            """
            SELECT lower(btrim(query_text)) AS pattern, count(*) AS n
            FROM strata_learning.interaction_events
            WHERE lower(btrim(query_text)) = ANY($1::text[])
              AND created_at > now() - interval '1 day' * $2
            GROUP BY 1
            """,
            list(evidence),
            lookback_days,
        )
        asked_count = {r["pattern"]: r["n"] for r in asked}

        for pattern, rewrites in evidence.items():
            rewrite, success = rewrites.most_common(1)[0]
            await self.store.upsert_learned_rewrite(
                pattern=pattern,
                pattern_type="exact",
                optimized_rewrite=rewrite,
                layers=layers[(pattern, rewrite)],
                success_count=success,
                total_count=max(asked_count.get(pattern, 0), success),
            )

        return len(evidence)

    async def mine_synonyms(self, lookback_days: int = 30) -> int:
        """Mine synonym pairs from reformulation chains.

        Pairs each interaction detected as a reformulation with the
        interaction just before it in the same session, and extracts term
        mappings from the query differences. A pair seen fewer than
        MIN_SYNONYM_SIGHTINGS times in the window is not stored.

        Returns how many pairs were written.
        """
        await self.store.purge_weak_mined_synonyms()

        rows = await self.store._fetch(
            f"""
            SELECT
                prev.query_text as query_a,
                ie.query_text as query_b,
                ie.layers_searched as layers
            FROM strata_learning.feedback_signals fs
            JOIN strata_learning.interaction_events ie ON fs.interaction_id = ie.id
            {_PREVIOUS_INTERACTION}
            WHERE fs.signal_name = 'reformulation'
              AND ie.created_at > now() - interval '1 day' * $1
            """,
            lookback_days,
        )

        sightings: Counter = Counter()
        for row in rows:
            # Signals stored under the older, looser rule stay in the window
            if (
                content_similarity(row["query_a"], row["query_b"])
                < REFORMULATION_MIN_SIMILARITY
            ):
                continue
            layer = row["layers"][0] if row.get("layers") else None
            for c in extract_synonym_candidates(row["query_a"], row["query_b"]):
                sightings[(c["term"], c["synonym"], layer)] += 1

        count = 0
        for (term, synonym, layer), seen in sightings.items():
            if seen < MIN_SYNONYM_SIGHTINGS:
                continue
            await self.store.upsert_synonym(
                term=term,
                synonym=synonym,
                layer=layer,
                source="pattern_mining",
                confidence=MINED_SYNONYM_CONFIDENCE,
                usage_count=seen,
            )
            count += 1

        return count

    async def update_layer_affinities(self, lookback_days: int = 30) -> int:
        """Learn which layers a question belongs to from routing corrections.

        Evidence:
        1. Explicit correct_layer feedback: the named layer is right, and the
           other layers searched were wrong.
        2. A layer_switch signal: the user named layers right after routing
           picked others for the previous question.

        The affinity score is (right + 1) / (right + wrong + 2), so a single
        correction gives 0.67, enough to steer routing (threshold 0.6).
        """
        rows = await self.store._fetch(
            f"""
            SELECT ie.query_text AS question,
                   ARRAY[fs.metadata->>'correct_layer'] AS right_layers,
                   ie.layers_searched AS wrong_layers
            FROM strata_learning.feedback_signals fs
            JOIN strata_learning.interaction_events ie ON ie.id = fs.interaction_id
            WHERE fs.signal_name = 'correct_layer'
              AND fs.created_at > now() - interval '1 day' * $1
            UNION ALL
            SELECT prev.query_text,
                   ARRAY(SELECT jsonb_array_elements_text(fs.metadata->'user_specified')),
                   ARRAY(SELECT jsonb_array_elements_text(fs.metadata->'auto_detected'))
            FROM strata_learning.feedback_signals fs
            JOIN strata_learning.interaction_events ie ON ie.id = fs.interaction_id
            {_PREVIOUS_INTERACTION}
            WHERE fs.signal_name = 'layer_switch'
              AND fs.created_at > now() - interval '1 day' * $1
            """,
            lookback_days,
        )

        right: Counter = Counter()
        wrong: Counter = Counter()
        for row in rows:
            pattern = _normalize(row["question"])
            right_layers = {layer for layer in row["right_layers"] or [] if layer}
            for layer in right_layers:
                right[(pattern, layer)] += 1
            for layer in set(row["wrong_layers"] or []) - right_layers:
                wrong[(pattern, layer)] += 1

        keys = set(right) | set(wrong)
        for pattern, layer in keys:
            r, w = right[(pattern, layer)], wrong[(pattern, layer)]
            await self.store.upsert_layer_affinity(
                query_pattern=pattern,
                layer_name=layer,
                affinity_score=(r + 1) / (r + w + 2),
                sample_count=r + w,
            )

        return len(keys)

    async def decay_stale_rewrites(
        self, unused_days: int = 60, decay_factor: float = 0.9
    ) -> int:
        """Decay success_rate on rewrites not used recently. Returns how many decayed."""
        status = await self.store._execute(
            """
            UPDATE strata_learning.learned_rewrites
            SET success_rate = success_rate * $1,
                updated_at = now()
            WHERE last_used < now() - interval '1 day' * $2
              AND success_rate > 0.1
            """,
            decay_factor,
            unused_days,
        )
        # The status string is "UPDATE <rows>"
        try:
            return int(status.split()[-1]) if status else 0
        except ValueError:
            return 0
