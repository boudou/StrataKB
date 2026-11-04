"""Regression tests for implicit filters and automatic layer fallback."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stratakb.config import Settings
from stratakb.rag.retriever import SearchResult, _hybrid_search
from stratakb.rag.tools import register_rag_tools


class _ToolCollector:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn

        return register


@pytest.fixture
def tools(monkeypatch):
    settings = Settings(
        _env_file=None,
        learning_enabled=False,
        embedding_backend="local",
        anthropic_api_key="test-key",
        reranking_enabled=False,
        reranking_candidate_multiplier=3,
        reranking_model="test-model",
        strata_user="test-user",
        strata_role="viewer",
    )
    monkeypatch.setattr("stratakb.config.get_settings", lambda: settings)
    monkeypatch.setattr("stratakb.rag.tools.write_audit_log", AsyncMock())
    detect = AsyncMock(return_value=[])
    monkeypatch.setattr("stratakb.rag.tools._detect_layers", detect)
    collector = _ToolCollector()
    register_rag_tools(collector, "postgresql://test")
    return collector.tools, settings, detect


def _result(name, layer, *, score=1.0, similarity=0.9, doc_type="markdown"):
    result = SearchResult(
        chunk_id=name,
        content=f"content for {name}",
        score=score,
        layer_name=layer,
        source_file=f"{name}.md",
        source_section="section",
        doc_type=doc_type,
    )
    result.similarity_score = similarity
    return result


@pytest.mark.asyncio
async def test_search_does_not_turn_hint_words_into_hard_filters(tools, monkeypatch):
    registered, _, _ = tools
    search = AsyncMock(return_value=[_result("guide", "docs")])
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", search)

    response = await registered["strata_search"]("accounting guide", top_k=1)

    assert response["status"] == "ok"
    assert response["results"][0]["doc_type"] == "markdown"
    assert search.await_args.kwargs["metadata_filter"] is None


@pytest.mark.asyncio
async def test_search_underfilled_route_retries_all_layers_with_explicit_filter(tools, monkeypatch):
    registered, _, detect = tools
    detect.return_value = ["routed"]
    search = AsyncMock(
        side_effect=[
            [_result("first", "routed", doc_type="pdf")],
            [
                _result("first", "routed", doc_type="pdf"),
                _result("second", "other", doc_type="pdf"),
            ],
        ]
    )
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", search)
    explicit_filter = {"doc_type": "pdf"}

    response = await registered["strata_search"]("guide", top_k=2, metadata_filter=explicit_filter)

    assert [r["layer"] for r in response["results"]] == ["routed", "other"]
    assert response["auto_detected_layers"] == ["routed"]
    assert response["routing_fallback"] == {
        "reason": "insufficient_results",
        "searched_layers": "all",
    }
    assert [call.kwargs["layers"] for call in search.await_args_list] == [["routed"], None]
    assert all(call.kwargs["metadata_filter"] == explicit_filter for call in search.await_args_list)


@pytest.mark.asyncio
async def test_search_uses_similarity_not_normalized_rank_to_detect_low_confidence(
    tools, monkeypatch
):
    registered, _, detect = tools
    detect.return_value = ["routed"]
    search = AsyncMock(
        side_effect=[
            [
                _result("weak1", "routed", similarity=0.2),
                _result("weak2", "routed", similarity=0.1),
            ],
            [_result("strong", "other"), _result("weak1", "routed", similarity=0.2)],
        ]
    )
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", search)

    response = await registered["strata_search"]("question", top_k=2)

    assert response["routing_fallback"]["reason"] == "low_confidence"
    assert response["results"][0]["layer"] == "other"
    assert search.await_count == 2


@pytest.mark.asyncio
async def test_explicit_layers_are_never_broadened(tools, monkeypatch):
    registered, _, detect = tools
    search = AsyncMock(return_value=[_result("weak", "chosen", similarity=0.1)])
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", search)

    response = await registered["strata_search"]("question", layers=["chosen"], top_k=2)

    assert response["status"] == "ok"
    assert "routing_fallback" not in response
    assert search.await_count == 1
    assert search.await_args.kwargs["layers"] == ["chosen"]
    detect.assert_not_awaited()


@pytest.mark.asyncio
async def test_ask_explicit_layers_are_never_broadened(tools, monkeypatch):
    registered, _, detect = tools
    monkeypatch.setattr(
        "stratakb.rag.query_rewriter.rewrite_query", AsyncMock(return_value=["question"])
    )
    search = AsyncMock(return_value=[_result("weak", "chosen", similarity=0.1)])
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", search)

    response = await registered["strata_ask"]("question", layers=["chosen"], top_k=2)

    assert response["status"] == "ok"
    assert "routing_fallback" not in response
    assert search.await_count == 1
    assert search.await_args.kwargs["layers"] == ["chosen"]
    detect.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_reports_fallback_failure(tools, monkeypatch):
    registered, _, detect = tools
    detect.return_value = ["routed"]
    search = AsyncMock(side_effect=[[], RuntimeError("database unavailable")])
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", search)

    response = await registered["strata_search"]("question", top_k=2)

    assert response == {"status": "error", "error": "database unavailable"}
    assert search.await_count == 2


@pytest.mark.asyncio
async def test_ask_falls_back_after_deduplicating_rewritten_queries(tools, monkeypatch):
    registered, _, detect = tools
    detect.return_value = ["routed"]
    monkeypatch.setattr(
        "stratakb.rag.query_rewriter.rewrite_query", AsyncMock(return_value=["one", "two"])
    )

    async def search(_db, _key, _query, *, layers, top_k):
        if layers:
            return [_result("first", "routed")]
        return [_result("first", "routed"), _result("second", "other", score=0.9)]

    search_mock = AsyncMock(side_effect=search)
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", search_mock)

    response = await registered["strata_ask"]("question", top_k=2)

    assert response["routing_fallback"]["reason"] == "insufficient_results"
    assert [c["chunk_id"] for c in response["retrieved_chunks"]] == ["first", "second"]
    assert {s["layer"] for s in response["sources"]} == {"routed", "other"}
    assert search_mock.await_count == 4
    assert [call.kwargs["layers"] for call in search_mock.await_args_list] == [
        ["routed"],
        ["routed"],
        None,
        None,
    ]


@pytest.mark.asyncio
async def test_ask_uses_reranked_confidence_for_fallback(tools, monkeypatch):
    registered, settings, detect = tools
    settings.reranking_enabled = True
    detect.return_value = ["routed"]
    monkeypatch.setattr(
        "stratakb.rag.query_rewriter.rewrite_query", AsyncMock(return_value=["question"])
    )
    search = AsyncMock(
        side_effect=[
            [_result("a", "routed"), _result("b", "routed")],
            [_result("c", "other"), _result("d", "other")],
        ]
    )
    monkeypatch.setattr("stratakb.rag.retriever.search_chunks", search)

    def rerank(_question, chunks, _top_k, _model):
        score = 0.2 if chunks[0]["layer"] == "routed" else 0.9
        return [{**chunk, "rerank_score": score} for chunk in chunks]

    monkeypatch.setattr("stratakb.rag.reranker.rerank_results", rerank)
    response = await registered["strata_ask"]("question", top_k=2)

    assert response["routing_fallback"]["reason"] == "low_confidence"
    assert response["confidence"]["level"] == "medium"
    assert all(chunk["layer"] == "other" for chunk in response["retrieved_chunks"])
    assert search.await_count == 2


@pytest.mark.asyncio
async def test_hybrid_search_keeps_cosine_similarity_separate_from_rrf_rank():
    chunk_id = "00000000-0000-0000-0000-000000000001"
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [{"source": "semantic", "id": chunk_id, "rank": 1}],
        [
            {
                "id": chunk_id,
                "content": "weak match",
                "layer_name": "routed",
                "source_file": "a.md",
                "source_section": "",
                "doc_type": "markdown",
                "metadata": {},
                "similarity_score": 0.2,
            }
        ],
    ]
    transaction = AsyncMock()
    conn.transaction = MagicMock(return_value=transaction)
    acquired = AsyncMock()
    acquired.__aenter__.return_value = conn
    pool = MagicMock()
    pool.acquire.return_value = acquired
    settings = SimpleNamespace(
        hnsw_ef_search=40,
        hybrid_rrf_k=60,
        hybrid_semantic_weight=0.7,
        hybrid_keyword_weight=0.3,
    )

    results = await _hybrid_search(pool, "question", "[1,0]", ["routed"], 1, None, settings)

    assert results[0].score == 1.0
    assert results[0].similarity_score == 0.2
    assert conn.fetch.await_args_list[1].args[2] == "[1,0]"
