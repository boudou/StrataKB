"""Tests for the interaction tracker."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stratakb.learning.store import LearningStore
from stratakb.learning.tracker import Tracker
from stratakb.rag.confidence import Confidence
from stratakb.rag.passages import AskResult, SearchResponse, ToolFailure


@pytest.mark.asyncio
async def test_tracker_calls_pii_scrub():
    """Tracker should scrub PII from query before storing."""
    settings = MagicMock(
        learning_enabled=True,
        learning_store_raw_queries=False,
        learning_session_timeout_minutes=30,
    )

    store = MagicMock(spec=LearningStore)
    store.get_or_create_session = AsyncMock(
        return_value={"id": "sess-1", "session_start": None, "query_count": 0}
    )
    store.store_interaction = AsyncMock(return_value={"id": "int-1", "created_at": None})
    store.update_session = AsyncMock()
    store.get_recent_interactions = AsyncMock(return_value=[])
    store.store_feedback = AsyncMock()

    tracker = Tracker(store, settings)

    with patch("stratakb.learning.tracker.scrub_query", return_value="scrubbed") as mock_scrub:
        await tracker._record(
            tool_name="strata_search",
            query_text="check IBAN GR1601101250000000012300695",
            layers_searched=["sepa"],
            chunks=[],
            confidence_level=None,
            confidence_score=0.5,
            response_time_ms=100,
        )
        mock_scrub.assert_called_once()

    # Verify scrubbed query was stored, not the raw one
    call_kwargs = store.store_interaction.call_args[1]
    assert call_kwargs["query_text"] == "scrubbed"


@pytest.mark.asyncio
async def test_tracker_skips_when_disabled():
    """Tracker should pass-through when learning is disabled."""
    settings = MagicMock(learning_enabled=False)

    store = MagicMock()
    tracker = Tracker(store, settings)

    response = _search_response()
    original_fn = AsyncMock(return_value=response)
    result = await tracker.track_search(original_fn, "test query")
    original_fn.assert_called_once()
    assert result is response


@pytest.mark.asyncio
async def test_tracker_detects_signals_on_reformulation():
    """Tracker should detect reformulation signals between sequential queries."""
    settings = MagicMock(
        learning_enabled=True,
        learning_store_raw_queries=True,
        learning_session_timeout_minutes=30,
    )

    now = datetime.now(UTC)
    store = MagicMock(spec=LearningStore)
    store.get_or_create_session = AsyncMock(return_value={"id": "sess-1"})
    store.store_interaction = AsyncMock(return_value={"id": "int-2", "created_at": now})
    store.update_session = AsyncMock()
    store.get_recent_interactions = AsyncMock(return_value=[
        {
            "id": "int-2",
            "query_text": "SEPA credit transfer rejection codes",
            "created_at": now,
            "confidence_level": None,
            "layers_searched": None,
            "results": None,
        },
        {
            "id": "int-1",
            "query_text": "SEPA credit transfer errors",
            "created_at": now - timedelta(seconds=15),
            "confidence_level": "low",
            "layers_searched": None,
            "results": None,
        },
    ])
    store.store_feedback = AsyncMock()

    tracker = Tracker(store, settings)
    await tracker._record(
        tool_name="strata_ask",
        query_text="SEPA credit transfer rejection codes",
        layers_searched=None,
        chunks=[],
        confidence_level=None,
        confidence_score=0.5,
        response_time_ms=200,
    )

    # Should have detected reformulation signal and stored feedback
    assert store.store_feedback.called
    feedback_call = store.store_feedback.call_args[1]
    assert feedback_call["signal_name"] == "reformulation"


@pytest.mark.asyncio
async def test_promoter_fires_after_n_queries():
    """Promoter should run after learning_promote_every_n_queries interactions."""
    mock_settings = MagicMock()
    mock_settings.learning_enabled = True
    mock_settings.learning_promote_every_n_queries = 3
    mock_settings.learning_session_timeout_minutes = 30
    mock_settings.learning_store_raw_queries = False

    mock_store = AsyncMock()
    mock_store.get_or_create_session.return_value = {"id": "s1", "query_count": 0}
    mock_store.store_interaction.return_value = {"id": "i1", "created_at": None}
    mock_store.get_recent_interactions.return_value = []

    tracker = Tracker(mock_store, mock_settings)
    tracker._interaction_count = 2  # Next call = 3rd

    with patch.object(tracker, "_run_promoter", new_callable=AsyncMock) as mock_promoter:
        await tracker._record(
            tool_name="strata_search",
            query_text="test",
            layers_searched=None,
            chunks=[],
            confidence_level=None,
            confidence_score=0.5,
            response_time_ms=100,
        )
        mock_promoter.assert_called_once()


def _mock_store(recent=None):
    store = MagicMock(spec=LearningStore)
    store.get_or_create_session = AsyncMock(return_value={"id": "sess-1"})
    store.store_interaction = AsyncMock(return_value={"id": "int-9", "created_at": None})
    store.update_session = AsyncMock()
    store.get_recent_interactions = AsyncMock(return_value=recent or [])
    store.store_feedback = AsyncMock(return_value=True)
    return store


def _confidence(level="medium", top_score=0.5):
    return Confidence(level, top_score, 1, "similarity")


def _ask_result(**fields):
    defaults = {
        "question": "question",
        "queries_used": ["question"],
        "confidence": _confidence(),
        "chunks": [{"chunk_id": "c1"}],
        "passages": [],
        "status": "found",
    }
    return AskResult(**{**defaults, **fields})


def _search_response(**fields):
    defaults = {"query": "query", "confidence": _confidence(), "results": []}
    return SearchResponse(**{**defaults, **fields})


def _enabled_settings():
    return MagicMock(
        learning_enabled=True,
        learning_store_raw_queries=True,
        learning_session_timeout_minutes=30,
        learning_promote_every_n_queries=1000,
    )


@pytest.mark.asyncio
async def test_track_search_returns_interaction_id_and_timing():
    store = _mock_store()
    tracker = Tracker(store, _enabled_settings())

    search = AsyncMock(return_value=_search_response(
        results=[{"chunk_id": "c1", "score": 0.7}],
        auto_detected_layers=["sepa"],
    ))

    with patch("stratakb.learning.tracker.time.monotonic", side_effect=[10.0, 10.25]):
        result = await tracker.track_search(search, "SEPA reject codes", None, 5, None)

    assert result.interaction_id == "int-9"
    assert result.to_dict()["interaction_id"] == "int-9"
    kwargs = store.store_interaction.call_args.kwargs
    assert kwargs["response_time_ms"] == 250
    assert kwargs["layers_searched"] == ["sepa"]
    assert kwargs["layers_detected"] == ["sepa"]


@pytest.mark.asyncio
async def test_track_ask_survives_recording_failure():
    store = _mock_store()
    store.store_interaction = AsyncMock(side_effect=RuntimeError("db down"))
    tracker = Tracker(store, _enabled_settings())

    ask = AsyncMock(return_value=_ask_result(chunks=[]))
    result = await tracker.track_ask(ask, "question", ["sepa"], True, 10)

    assert result.status == "found"
    assert result.interaction_id is None
    assert "interaction_id" not in result.concise()


@pytest.mark.asyncio
async def test_layer_switch_compares_with_previous_auto_detection():
    now = datetime.now(UTC)
    recent = [
        {"id": "int-9", "query_text": "cut-off times", "created_at": now,
         "confidence_level": None, "layers_detected": None, "layers_searched": ["sepa"],
         "results": []},
        {"id": "int-8", "query_text": "settlement cut-off",
         "created_at": now - timedelta(seconds=300), "confidence_level": "low",
         "layers_detected": ["iso20022"], "layers_searched": ["iso20022"],
         "results": [{"chunk_id": "c1", "score": 0.4, "layer": "iso20022"}]},
    ]
    store = _mock_store(recent)
    store.store_interaction = AsyncMock(return_value={"id": "int-9", "created_at": now})
    tracker = Tracker(store, _enabled_settings())
    await tracker._record(
        tool_name="strata_search",
        query_text="cut-off times",
        layers_searched=["sepa"],  # named by the caller: layers_detected is None
        chunks=[],
        confidence_level=None,
        confidence_score=0.5,
        response_time_ms=10,
    )

    names = [c.kwargs["signal_name"] for c in store.store_feedback.call_args_list]
    assert names == ["layer_switch"]


@pytest.mark.asyncio
async def test_track_search_stores_the_confidence_the_tool_returned():
    store = _mock_store()
    tracker = Tracker(store, _enabled_settings())

    # The similarities alone would score high: the tracker must not rescore.
    search = AsyncMock(return_value=_search_response(
        results=[
            {"chunk_id": f"c{i}", "score": 1.0 - i * 0.1, "similarity": 0.9}
            for i in range(3)
        ],
    ))
    await tracker.track_search(search, "SEPA reject codes", ["sepa"], 5, None)

    kwargs = store.store_interaction.call_args.kwargs
    assert kwargs["confidence_level"] == "medium"
    assert kwargs["confidence_score"] == 0.5


@pytest.mark.asyncio
async def test_track_search_stores_result_similarity():
    store = _mock_store()
    tracker = Tracker(store, _enabled_settings())

    search = AsyncMock(return_value=_search_response(results=[
        {"chunk_id": "c1", "content": "text", "score": 1.0, "similarity": 0.3,
         "layer": "sepa", "source_file": "rules.md"},
    ]))
    await tracker.track_search(search, "SEPA reject codes", ["sepa"], 5, None)

    kwargs = store.store_interaction.call_args.kwargs
    assert kwargs["results"][0] == {
        "chunk_id": "c1", "score": 1.0, "similarity": 0.3,
        "layer": "sepa", "source_file": "rules.md",
    }


@pytest.mark.asyncio
async def test_track_ask_records_rewritten_queries_and_chunks():
    store = _mock_store()
    tracker = Tracker(store, _enabled_settings())

    ask = AsyncMock(return_value=_ask_result(
        queries_used=["a", "b"],
        chunks=[{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        confidence=_confidence("high", 0.8),
    ))
    result = await tracker.track_ask(ask, "question", ["sepa"], True, 10)

    ask.assert_awaited_once_with("question", ["sepa"], True, 10, None)
    assert result.interaction_id == "int-9"
    assert result.concise()["interaction_id"] == "int-9"
    kwargs = store.store_interaction.call_args.kwargs
    assert kwargs["tool_used"] == "strata_ask"
    assert kwargs["rewritten_queries"] == ["a", "b"]
    assert kwargs["result_count"] == 2
    assert kwargs["confidence_level"] == "high"
    assert kwargs["confidence_score"] == 0.8
    # Named by the caller: nothing was detected
    assert kwargs["layers_searched"] == ["sepa"]
    assert kwargs["layers_detected"] is None


@pytest.mark.asyncio
async def test_track_ask_records_detected_layers():
    store = _mock_store()
    tracker = Tracker(store, _enabled_settings())

    ask = AsyncMock(return_value=_ask_result(auto_detected_layers=["iso20022"]))
    await tracker.track_ask(ask, "question", None, True, 10)

    kwargs = store.store_interaction.call_args.kwargs
    assert kwargs["layers_searched"] == ["iso20022"]
    assert kwargs["layers_detected"] == ["iso20022"]


@pytest.mark.asyncio
async def test_track_search_records_no_rewritten_queries():
    store = _mock_store()
    tracker = Tracker(store, _enabled_settings())

    search = AsyncMock(return_value=_search_response(results=[{"chunk_id": "c1"}]))
    await tracker.track_search(search, "query", ["sepa"], 5, {"doc_type": "pdf"})

    search.assert_awaited_once_with("query", ["sepa"], 5, {"doc_type": "pdf"})
    kwargs = store.store_interaction.call_args.kwargs
    assert kwargs["tool_used"] == "strata_search"
    assert kwargs["rewritten_queries"] is None
    assert kwargs["result_count"] == 1
    assert kwargs["results"][0]["chunk_id"] == "c1"
    assert kwargs["confidence_level"] == "medium"
    assert kwargs["confidence_score"] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["track_ask", "track_search"])
async def test_track_returns_error_result_without_recording(tool):
    store = _mock_store()
    tracker = Tracker(store, _enabled_settings())

    error = ToolFailure("boom")
    result = await getattr(tracker, tool)(AsyncMock(return_value=error), "query")

    assert result is error
    store.store_interaction.assert_not_awaited()
