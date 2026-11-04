"""Confidence scoring for RAG search results.

Assigns high/medium/low confidence based on how relevant the best results
are and how many relevant sources there are. Scores come on one of two
scales: the query/chunk cosine similarity ("similarity") or the
cross-encoder's relevance probability ("rerank").
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


# Provisional thresholds, one set per score scale. They are not measured yet:
# the evaluation suite's abstention tier sets the final values.
# Cosine similarity between query and chunk embeddings (all-mpnet-base-v2).
SIMILARITY_THRESHOLDS = ConfidenceThresholds(high_score=0.6, medium_score=0.45)
# Cross-encoder relevance probability (sigmoid over the logits; 0.5 is logit 0).
RERANK_THRESHOLDS = ConfidenceThresholds(high_score=0.8, medium_score=0.5)

_BASIS_THRESHOLDS = {
    "similarity": SIMILARITY_THRESHOLDS,
    "rerank": RERANK_THRESHOLDS,
}


def score_confidence(
    scores: list[float],
    *,
    basis: str = "similarity",
    thresholds: ConfidenceThresholds | None = None,
) -> dict:
    """Compute confidence level from search result scores.

    Args:
        scores: Scores of the search results, all on the scale named by basis.
        basis: "similarity" (query/chunk cosine similarity) or "rerank"
            (cross-encoder relevance probability, 0-1). Picks the thresholds.
        thresholds: Optional custom thresholds, overriding the basis default.

    Returns:
        Dict with level (high/medium/low), top_score, relevant_sources count,
        basis, and warning text if low.

    Raises:
        ValueError: If basis is not "similarity" or "rerank".
    """
    if basis not in _BASIS_THRESHOLDS:
        raise ValueError(f"Unknown confidence basis: {basis!r}")
    if not thresholds:
        thresholds = _BASIS_THRESHOLDS[basis]

    if not scores:
        return {
            "level": "low",
            "top_score": 0.0,
            "relevant_sources": 0,
            "basis": basis,
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
            "basis": basis,
        }
    elif top_score >= thresholds.medium_score and relevant >= thresholds.medium_min_sources:
        return {
            "level": "medium",
            "top_score": round(top_score, 4),
            "relevant_sources": relevant,
            "basis": basis,
        }
    else:
        return {
            "level": "low",
            "top_score": round(top_score, 4),
            "relevant_sources": relevant,
            "basis": basis,
            "warning": "Low confidence results. Consider verifying with authoritative sources or refining your query.",
        }
