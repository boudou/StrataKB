import pytest
from unittest.mock import patch, MagicMock


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
