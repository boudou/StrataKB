"""Cross-encoder reranking for improved retrieval precision.

Over-retrieves from vector search, then reranks with a cross-encoder
to push the most relevant results to the top.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from stratakb.knowledge.embeddings import ModelKind, _hub_repo, load_cached_first
from stratakb.rag.confidence import (
    RERANK_THRESHOLDS,
    SIMILARITY_THRESHOLDS,
    ConfidenceThresholds,
)

if TYPE_CHECKING:
    from stratakb.config import Settings

logger = structlog.get_logger()

# CrossEncoder puts "cross-encoder/" before a bare model name
_RERANKING_MODEL_KIND = ModelKind(
    "reranking", "RERANKING_MODEL", "cross-encoder", "set RERANKING_ENABLED=false"
)

DEFAULT_RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankerProfile:
    """What one reranker model was measured with.

    Attributes:
        thresholds: Its confidence thresholds, on its relevance probability.
        pool_multiplier: strata_ask reranks top_k times this many candidates.
        max_length: The input length it loads with; None keeps the model's own.
    """

    thresholds: ConfidenceThresholds
    pool_multiplier: int = 1
    max_length: int | None = None


_DEFAULT_PROFILE = RerankerProfile(RERANK_THRESHOLDS)

# Measured on the PEP gold set (165 answerable and 50 must-refuse questions).
# The default gains little from a larger pool, so it keeps top_k; gte ranks
# better with a pool of 3 x top_k, at about 2 passages a second on a laptop CPU,
# and was measured at 512 tokens (its tokenizer allows 8192).
_PROFILES = {
    DEFAULT_RERANKING_MODEL: _DEFAULT_PROFILE,
    "Alibaba-NLP/gte-reranker-modernbert-base": RerankerProfile(
        ConfidenceThresholds(high_score=0.9, medium_score=0.86),
        pool_multiplier=3,
        max_length=512,
    ),
}

_uncalibrated_warned: set[str] = set()


def reranker_profile(model_name: str) -> RerankerProfile:
    """The profile of a reranker model; any other model gets the default's.

    A bare name is looked up as CrossEncoder loads it, with "cross-encoder/"
    before it. A model without a profile, a folder path included, logs
    uncalibrated_reranker once per name.
    """
    profile = _PROFILES.get(_hub_repo(model_name, _RERANKING_MODEL_KIND.organization))
    if profile is not None:
        return profile
    if model_name not in _uncalibrated_warned:
        _uncalibrated_warned.add(model_name)
        logger.warning("uncalibrated_reranker", model=model_name)
    return _DEFAULT_PROFILE


def ranking_settings(settings: Settings) -> tuple[ConfidenceThresholds, int | None]:
    """The thresholds strata_ask judges by and its rerank pool multiplier.

    With reranking on: the model's profile thresholds and its pool multiplier,
    unless RERANKING_CANDIDATE_MULTIPLIER overrides it. With reranking off: the
    similarity thresholds and no multiplier.
    """
    if not settings.reranking_enabled:
        return SIMILARITY_THRESHOLDS, None
    profile = reranker_profile(settings.reranking_model)
    return (
        profile.thresholds,
        settings.reranking_candidate_multiplier or profile.pool_multiplier,
    )


_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker(model_name: str = DEFAULT_RERANKING_MODEL):
    """Lazy-load the cross-encoder model (singleton, thread-safe)."""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        from sentence_transformers import CrossEncoder

        logger.info("loading_reranker_model", model=model_name)
        max_length = reranker_profile(model_name).max_length
        model_kwargs = {} if max_length is None else {"max_length": max_length}
        _reranker = load_cached_first(
            CrossEncoder, model_name, _RERANKING_MODEL_KIND, **model_kwargs
        )
        logger.info("reranker_model_loaded", model=model_name)
    return _reranker


def score_pairs(
    query: str,
    texts: list[str],
    model_name: str = DEFAULT_RERANKING_MODEL,
) -> list[float]:
    """Score each text against the query as a relevance probability (0-1).

    The sigmoid is set at predict time, so it replaces the model's own default
    activation: the result is one sigmoid over the logits for any model.
    """
    from torch import nn

    reranker = _get_reranker(model_name)
    scores = reranker.predict([(query, t) for t in texts], activation_fn=nn.Sigmoid())
    return [float(s) for s in scores]


def rerank_results(
    query: str,
    chunks: list[dict],
    top_k: int = 10,
    model_name: str = DEFAULT_RERANKING_MODEL,
) -> list[dict]:
    """Rerank chunks using a cross-encoder model.

    Args:
        query: The original search query.
        chunks: List of chunk dicts with at least "content" key.
        top_k: Number of results to return after reranking.
        model_name: Cross-encoder model to use.

    Returns:
        Top-k chunks sorted by cross-encoder relevance, each with a
        "rerank_score": the relevance probability (0-1).
    """
    if not chunks:
        return []

    scores = score_pairs(query, [c["content"] for c in chunks], model_name)

    # Pair chunks with new scores and sort
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        reranked = dict(chunk)
        reranked["rerank_score"] = round(float(score), 4)
        results.append(reranked)

    logger.info(
        "reranking_complete",
        input_chunks=len(chunks),
        output_chunks=len(results),
        top_rerank_score=results[0]["rerank_score"] if results else 0,
    )
    return results
