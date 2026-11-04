"""Confidence scoring for RAG search results.

Assigns high/medium/low confidence based on cosine similarity scores
and number of relevant sources.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceThresholds:
    """Configurable thresholds for confidence scoring."""

    high_score: float = 0.8
    medium_score: float = 0.6
    high_min_sources: int = 3
    medium_min_sources: int = 1


def score_confidence(
    scores: list[float],
    *,
    thresholds: ConfidenceThresholds | None = None,
) -> dict:
    """Compute confidence level from search result scores.

    Args:
        scores: List of cosine similarity scores (0-1) from search results.
        thresholds: Optional custom thresholds.

    Returns:
        Dict with level (high/medium/low), top_score, relevant_sources count, and warning text if low.
    """
    if not thresholds:
        thresholds = ConfidenceThresholds()

    if not scores:
        return {
            "level": "low",
            "top_score": 0.0,
            "relevant_sources": 0,
            "warning": "No results found. Try broadening your search or using different terms.",
        }

    top_score = max(scores)
    # Count sources above medium threshold as "relevant"
    relevant = sum(1 for s in scores if s >= thresholds.medium_score)

    if top_score >= thresholds.high_score and relevant >= thresholds.high_min_sources:
        return {
            "level": "high",
            "top_score": round(top_score, 4),
            "relevant_sources": relevant,
        }
    elif top_score >= thresholds.medium_score and relevant >= thresholds.medium_min_sources:
        return {
            "level": "medium",
            "top_score": round(top_score, 4),
            "relevant_sources": relevant,
        }
    else:
        return {
            "level": "low",
            "top_score": round(top_score, 4),
            "relevant_sources": relevant,
            "warning": "Low confidence results. Consider verifying with authoritative sources or refining your query.",
        }
