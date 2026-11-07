import pytest
from unittest.mock import patch, MagicMock

from tests.helpers import LogitModel


def test_rerank_sorts_by_cross_encoder_score():
    """Reranker should sort results by cross-encoder score, not original score."""
    from stratakb.rag.reranker import rerank_results

    chunks = [
        {"content": "Irrelevant chunk", "score": 0.95},
        {"content": "The SEPA rejection code AC01 means incorrect account number", "score": 0.70},
        {"content": "Another irrelevant chunk", "score": 0.90},
    ]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.95, 0.2]

    with patch("stratakb.rag.reranker._get_reranker", return_value=mock_model):
        reranked = rerank_results("SEPA rejection AC01", chunks, top_k=2)

    assert len(reranked) == 2
    assert "AC01" in reranked[0]["content"]
    assert reranked[0]["rerank_score"] == 0.95
    assert reranked[1]["rerank_score"] == 0.2


def test_rerank_returns_all_when_fewer_than_top_k():
    """When fewer chunks than top_k, return all of them."""
    from stratakb.rag.reranker import rerank_results

    chunks = [{"content": "Only chunk", "score": 0.8}]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9]

    with patch("stratakb.rag.reranker._get_reranker", return_value=mock_model):
        reranked = rerank_results("query", chunks, top_k=5)

    assert len(reranked) == 1
    assert reranked[0]["rerank_score"] == 0.9


def test_rerank_empty_input():
    """Empty input should return empty list without calling model."""
    from stratakb.rag.reranker import rerank_results
    assert rerank_results("query", [], top_k=5) == []


def test_rerank_preserves_chunk_fields():
    """Reranker should preserve all original chunk fields."""
    from stratakb.rag.reranker import rerank_results

    chunks = [
        {
            "content": "Some content",
            "score": 0.8,
            "chunk_id": "abc-123",
            "layer": "sepa",
            "metadata": {"key": "value"},
        }
    ]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.75]

    with patch("stratakb.rag.reranker._get_reranker", return_value=mock_model):
        reranked = rerank_results("query", chunks, top_k=5)

    assert reranked[0]["chunk_id"] == "abc-123"
    assert reranked[0]["layer"] == "sepa"
    assert reranked[0]["metadata"] == {"key": "value"}
    assert reranked[0]["score"] == 0.8  # original score preserved
    assert reranked[0]["rerank_score"] == 0.75  # new score added


def test_rerank_scores_are_probabilities():
    """Raw logits become relevance probabilities through one sigmoid."""
    from torch import nn

    from stratakb.rag.reranker import rerank_results

    chunks = [{"content": "relevant", "score": 0.5}, {"content": "unrelated", "score": 0.9}]
    model = LogitModel([4.0, -4.0])

    with patch("stratakb.rag.reranker._get_reranker", return_value=model):
        reranked = rerank_results("query", chunks, top_k=2)

    assert [c["rerank_score"] for c in reranked] == [0.982, 0.018]
    assert isinstance(model.activation_fn, nn.Sigmoid)


def test_score_pairs_keeps_input_order():
    """score_pairs returns one float per text, in the order the texts came in."""
    from stratakb.rag.reranker import score_pairs

    model = LogitModel([-1.0, 2.0, 0.0])

    with patch("stratakb.rag.reranker._get_reranker", return_value=model):
        scores = score_pairs("query", ["a", "b", "c"])

    assert all(isinstance(s, float) for s in scores)
    assert [round(s, 4) for s in scores] == [0.2689, 0.8808, 0.5]


def test_local_only_message_text():
    """The local-only message names the cross-encoder repository and the switch."""
    from stratakb.knowledge.embeddings import _local_only_error
    from stratakb.rag.reranker import _RERANKING_MODEL_KIND

    assert _local_only_error("scorer", _RERANKING_MODEL_KIND) == (
        "STRATA_LOCAL_ONLY is on and the reranking model scorer is not installed on this "
        "machine, so it cannot be downloaded. On a machine that may download, run "
        "`uv run hf download cross-encoder/scorer`, set RERANKING_MODEL to a folder that "
        "holds the model, or set RERANKING_ENABLED=false."
    )


def test_offline_message_text():
    """The offline message lists the ways out with one "or", like the local-only one."""
    from stratakb.knowledge.embeddings import _offline_error
    from stratakb.rag.reranker import _RERANKING_MODEL_KIND

    assert _offline_error("scorer", _RERANKING_MODEL_KIND) == (
        "HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set and the reranking model scorer is "
        "not in the Hugging Face cache, so it cannot be downloaded. Unset both to let "
        "StrataKB download it, run `uv run hf download cross-encoder/scorer` on a machine "
        "that may download, set RERANKING_MODEL to a folder that holds the model, or set "
        "RERANKING_ENABLED=false."
    )
