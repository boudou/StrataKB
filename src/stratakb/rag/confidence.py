"""Confidence scoring for RAG search results.

Assigns high/medium/low confidence based on how relevant the best results
are and how many relevant sources there are. Scores come on one of two
scales: the query/chunk cosine similarity ("similarity") or the
cross-encoder's relevance probability ("rerank").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Thresholds for one score scale; the score cut-offs have no default."""

    high_score: float
    medium_score: float
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

_NO_RESULTS_WARNING = "No results found. Try broadening your search or using different terms."
_LOW_CONFIDENCE_WARNING = (
    "Low confidence results. Consider verifying with authoritative sources"
    " or refining your query."
)


def similarity_or_zero(similarity: float | None) -> float:
    """A chunk without an embedding has no similarity; it counts as 0.0."""
    return 0.0 if similarity is None else similarity


def score_confidence(scores: list[float], *, basis: str) -> dict:
    """Compute confidence level from search result scores.

    Args:
        scores: Scores of the search results, all on the scale named by basis.
        basis: "similarity" (query/chunk cosine similarity) or "rerank"
            (cross-encoder relevance probability, 0-1). Required: it picks the
            thresholds, and the result reports it.

    Returns:
        Dict with level (high/medium/low), top_score, relevant_sources count,
        basis, and warning text if low.

    Raises:
        ValueError: If basis is not "similarity" or "rerank".
    """
    if basis not in _BASIS_THRESHOLDS:
        raise ValueError(f"Unknown confidence basis: {basis!r}")
    thresholds = _BASIS_THRESHOLDS[basis]

    if not scores:
        level, top_score, relevant = "low", 0.0, 0
    else:
        top_score = max(scores)
        # Count sources above medium threshold as "relevant"
        relevant = sum(1 for s in scores if s >= thresholds.medium_score)
        if top_score >= thresholds.high_score and relevant >= thresholds.high_min_sources:
            level = "high"
        elif top_score >= thresholds.medium_score and relevant >= thresholds.medium_min_sources:
            level = "medium"
        else:
            level = "low"

    result = {
        "level": level,
        "top_score": round(top_score, 4),
        "relevant_sources": relevant,
        "basis": basis,
    }
    if level == "low":
        result["warning"] = _LOW_CONFIDENCE_WARNING if scores else _NO_RESULTS_WARNING
    return result


def answer_status(confidence: dict, passage_count: int) -> str:
    """found, partly_found or not_found, judged on the best passage's score.

    The best passage clearing the high threshold is found, the medium one
    partly found; anything below, or no passage at all, is not found.
    """
    if passage_count == 0:
        return "not_found"
    thresholds = _BASIS_THRESHOLDS[confidence["basis"]]
    top_score = confidence["top_score"]
    if top_score >= thresholds.high_score:
        return "found"
    if top_score >= thresholds.medium_score:
        return "partly_found"
    return "not_found"
