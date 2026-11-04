"""Interaction tracker — wraps RAG tools with learning observation.

Transparently records every query, its results, and detected
feedback signals into strata_learning tables.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from stratakb.config import get_settings
from stratakb.learning.pii import scrub_query
from stratakb.learning.signals import detect_reformulation, detect_layer_switch, detect_follow_up
from stratakb.learning.store import LearningStore

logger = structlog.get_logger()


class Tracker:
    """Wraps RAG tool calls with interaction recording."""

    def __init__(self, store: LearningStore):
        self.store = store
        self._settings = get_settings()
        self._interaction_count = 0

    async def track_ask(
        self,
        original_fn,
        question: str,
        layers: list[str] | None = None,
        include_sources: bool = True,
        top_k: int = 10,
    ) -> dict:
        """Wrap strata_ask with tracking; adds interaction_id to the result."""
        if not self._settings.learning_enabled:
            return await original_fn(question, layers, include_sources, top_k)

        start = time.monotonic()
        result = await original_fn(question, layers, include_sources, top_k)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.get("status") != "ok":
            return result

        detected = result.get("auto_detected_layers")
        await self._record_safely(
            result,
            tool_name="strata_ask",
            query_text=question,
            layers_searched=layers or detected,
            layers_detected=detected,
            rewritten_queries=result.get("queries_used"),
            chunks=result.get("retrieved_chunks", []),
            confidence_level=result.get("confidence", {}).get("level"),
            confidence_score=result.get("confidence", {}).get("top_score"),
            response_time_ms=elapsed_ms,
        )

        return result

    async def track_search(
        self,
        original_fn,
        query: str,
        layers: list[str] | None = None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> dict:
        """Wrap strata_search with tracking; adds interaction_id to the result."""
        if not self._settings.learning_enabled:
            return await original_fn(query, layers, top_k, metadata_filter)

        start = time.monotonic()
        result = await original_fn(query, layers, top_k, metadata_filter)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.get("status") != "ok":
            return result

        chunks = result.get("results", [])
        top_score = max((c.get("score", 0) for c in chunks), default=0)

        detected = result.get("auto_detected_layers")
        await self._record_safely(
            result,
            tool_name="strata_search",
            query_text=query,
            layers_searched=layers or detected,
            layers_detected=detected,
            rewritten_queries=None,
            chunks=chunks,
            confidence_level=None,
            confidence_score=top_score,
            response_time_ms=elapsed_ms,
        )

        return result

    async def _record_safely(self, result: dict, **kwargs) -> None:
        """Record the interaction and add its id to the result; tracking never fails the tool."""
        try:
            interaction_id = await self._record(**kwargs)
        except Exception as e:
            logger.warning("learning_record_failed", error=str(e))
            return
        if interaction_id:
            result["interaction_id"] = interaction_id

    async def _record(
        self,
        tool_name: str,
        query_text: str,
        layers_searched: list[str] | None,
        chunks: list[dict],
        confidence_level: str | None,
        confidence_score: float | None,
        response_time_ms: int,
        rewritten_queries: list[str] | None = None,
        layers_detected: list[str] | None = None,
    ) -> str | None:
        """Record interaction and detect signals. Returns the interaction id.

        layers_searched are the layers actually searched; layers_detected is
        set when they were picked by routing rather than by the caller.
        """
        try:
            from stratakb.core.auth import get_current_user
            user_id, _ = get_current_user()
        except Exception:
            user_id = None

        # Get or create session
        session = await self.store.get_or_create_session(
            user_id=user_id,
            timeout_minutes=self._settings.learning_session_timeout_minutes,
        )
        if not session:
            return None

        session_id = session["id"]

        # Scrub PII from query if configured
        stored_query = query_text
        if not self._settings.learning_store_raw_queries:
            stored_query = scrub_query(query_text)

        # Store the interaction
        result_summaries = [
            {
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score"),
                "layer": c.get("layer"),
                "source_file": c.get("source_file"),
            }
            for c in chunks[:20]  # Cap stored results
        ]

        interaction = await self.store.store_interaction(
            session_id=str(session_id),
            query_text=stored_query,
            rewritten_queries=rewritten_queries,
            layers_detected=layers_detected,
            layers_searched=layers_searched,
            results=result_summaries,
            result_count=len(chunks),
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            tool_used=tool_name,
            response_time_ms=response_time_ms,
        )

        if not interaction:
            return None

        interaction_id = str(interaction["id"])

        # Update session
        await self.store.update_session(str(session_id))

        # Detect implicit signals
        recent = await self.store.get_recent_interactions(str(session_id), limit=2)
        if len(recent) >= 2:
            prev = recent[1]  # Previous interaction (recent[0] is current)
            time_delta = (
                interaction["created_at"] - prev["created_at"]
            ).total_seconds()

            signals = detect_reformulation(
                current_query=query_text,
                previous_query=prev["query_text"],
                previous_confidence=prev.get("confidence_level"),
                time_delta_seconds=time_delta,
            )
            signals.extend(
                detect_layer_switch(
                    current_layers_explicit=None if layers_detected else layers_searched,
                    previous_layers_auto=prev.get("layers_detected"),
                )
            )
            signals.extend(
                detect_follow_up(
                    current_query=query_text,
                    previous_results=prev.get("results"),
                )
            )

            for sig in signals:
                await self.store.store_feedback(
                    interaction_id=interaction_id,
                    signal_type=sig["signal_type"],
                    signal_name=sig["signal_name"],
                    signal_value=sig["signal_value"],
                    metadata=sig.get("metadata"),
                )

            if signals:
                logger.info(
                    "learning_signals_detected",
                    count=len(signals),
                    names=[s["signal_name"] for s in signals],
                )

        # Auto-run promoter every N interactions
        self._interaction_count += 1
        if self._interaction_count % self._settings.learning_promote_every_n_queries == 0:
            asyncio.create_task(self._run_promoter())

        logger.debug("learning_interaction_recorded", interaction_id=interaction_id)
        return interaction_id

    async def _run_promoter(self) -> None:
        """Run the promoter engine to promote/decay learned patterns."""
        try:
            from stratakb.learning.promoter import Promoter
            promoter = Promoter(self.store)
            stats = await promoter.run()
            logger.info("auto_promoter_complete", **stats)
        except Exception as e:
            logger.debug("auto_promoter_failed", error=str(e))
