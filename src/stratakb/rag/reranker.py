"""Cross-encoder reranking for improved retrieval precision.

Over-retrieves from vector search, then reranks with a cross-encoder
to push the most relevant results to the top.
"""
from __future__ import annotations

import threading

import structlog

logger = structlog.get_logger()

_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load the cross-encoder model (singleton, thread-safe)."""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        from sentence_transformers import CrossEncoder

        from stratakb.knowledge.embeddings import load_cached_first

        logger.info("loading_reranker_model", model=model_name)
        _reranker = load_cached_first(
            CrossEncoder,
            model_name,
            f"STRATA_LOCAL_ONLY is on and the reranking model {model_name} is "
            f"not installed on this machine, so it cannot be downloaded. On a "
            f"machine that may download, run `uv run hf download {model_name}`, "
            f"set RERANKING_MODEL to a folder that holds the model, or set "
            f"RERANKING_ENABLED=false.",
        )
        logger.info("reranker_model_loaded", model=model_name)
    return _reranker


def score_pairs(
    query: str,
    texts: list[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
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
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
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
