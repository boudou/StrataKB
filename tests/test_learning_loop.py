"""End-to-end tests of the learning loop (require DB).

Each test records interactions through the tracker, adds feedback, runs the
promoter, and checks that routing and query rewriting use what was learned.
Questions carry a unique tag so tests don't see each other's data.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from stratakb.learning.promoter import Promoter
from stratakb.learning.store import LearningStore
from stratakb.rag.confidence import Confidence
from stratakb.rag.passages import AskResult, SearchResponse


@pytest.fixture
def tag():
    return f"probe{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def store(tag):
    from stratakb.config import get_settings

    store = LearningStore(get_settings().database_url)
    yield store
    await store._execute(
        "DELETE FROM strata_learning.interaction_events WHERE query_text LIKE $1", f"%{tag}%"
    )
    await store._execute(
        "DELETE FROM strata_learning.learned_rewrites WHERE pattern LIKE $1", f"%{tag}%"
    )
    await store._execute(
        "DELETE FROM strata_learning.layer_affinity_scores WHERE query_pattern LIKE $1", f"%{tag}%"
    )
    await store._execute(
        "DELETE FROM strata_learning.query_synonyms WHERE term LIKE $1 OR synonym LIKE $1",
        f"%{tag}%",
    )


def _tracker(store):
    from stratakb.learning.tracker import Tracker

    settings = MagicMock(
        learning_enabled=True,
        learning_store_raw_queries=True,
        learning_session_timeout_minutes=30,
        learning_promote_every_n_queries=1000,
    )
    return Tracker(store, settings)


def _ask_returning(queries, level, layers=None):
    async def ask(question, layers_arg, include_sources, top_k, mode=None):
        return AskResult(
            question=question,
            queries_used=queries,
            confidence=Confidence(level, 0.9 if level == "high" else 0.3, 1, "similarity"),
            chunks=[{"chunk_id": "c1", "score": 0.5, "layer": "sepa", "source_file": "a.md"}],
            passages=[],
            status="found" if level == "high" else "not_found",
            auto_detected_layers=layers or [],
        )

    return ask


def _search_returning(layers=None):
    async def search(query, layers_arg, top_k, metadata_filter):
        return SearchResponse(
            query=query,
            confidence=Confidence("low", 0.3, 0, "similarity"),
            results=[{"chunk_id": "c1", "score": 0.4, "layer": "x", "source_file": "a.md"}],
            auto_detected_layers=layers or [],
        )

    return search


@pytest.mark.asyncio
async def test_helpful_asks_promote_a_rewrite_that_bypasses_the_llm(store, tag):
    from stratakb.rag.query_rewriter import rewrite_query

    tracker = _tracker(store)
    question = f"why was my payment rejected {tag}"
    rewrite = [f"sepa reject reason codes {tag}", f"pacs.002 rejected status {tag}"]

    ask = _ask_returning(rewrite, "medium")
    for _ in range(5):
        result = await tracker.track_ask(ask, question, None, True, 10)
        assert result.interaction_id
        assert await store.store_feedback(result.interaction_id, "explicit", "helpful", 1.0)

    assert await Promoter(store).promote_rewrites() >= 1

    learned = await store.find_rewrites(question)
    assert learned is not None
    assert learned.queries == rewrite
    assert learned.total_count == 5
    assert learned.success_rate == pytest.approx(1.0)

    # Enough samples at a high success rate: the rewriter skips the API call
    queries = await rewrite_query(question, anthropic_api_key="unused", learning_store=store)
    assert queries == rewrite


@pytest.mark.asyncio
async def test_rescued_reformulation_promotes_a_rewrite(store, tag):
    tracker = _tracker(store)
    first = f"payment bounced back {tag}"
    second = f"payment bounced back reject reason code {tag}"

    await tracker.track_ask(_ask_returning([first], "low"), first, None, True, 10)
    await tracker.track_ask(_ask_returning([second], "high"), second, None, True, 10)

    await Promoter(store).promote_rewrites()

    learned = await store.find_rewrites(first, min_success_rate=0.5)
    assert learned is not None
    assert learned.queries == [second]


@pytest.mark.asyncio
async def test_correct_layer_feedback_steers_routing(store, tag):
    tracker = _tracker(store)
    question = f"settlement cut-off times {tag}"

    result = await tracker.track_search(_search_returning(["iso20022"]), question, None, 5, None)
    await store.store_feedback(
        result.interaction_id, "explicit", "correct_layer", -0.3, {"correct_layer": "sepa"}
    )

    await Promoter(store).update_layer_affinities()

    assert await store.find_layer_affinities(question) == ["sepa"]


async def _synonym_rows(store, tag):
    return await store._fetch(
        """
        SELECT term, synonym, layer, confidence, usage_count, source
        FROM strata_learning.query_synonyms WHERE term LIKE $1 OR synonym LIKE $1
        ORDER BY term
        """,
        f"%{tag}%",
    )


async def _ask(tracker, question):
    await tracker.track_ask(_ask_returning([question], "low"), question, None, True, 10)


@pytest.mark.asyncio
async def test_one_sighting_is_not_applied_and_two_are(store, tag):
    from stratakb.config import get_settings
    from stratakb.rag.query_rewriter import rewrite_query

    tracker = _tracker(store)
    first = f"rotate service keys{tag}"
    second = f"rotate service credentials{tag}"
    term, synonym = f"keys{tag}", f"credentials{tag}"
    settings = get_settings().model_copy(update={"query_rewrite_provider": "anthropic"})

    async def rewritten():
        return await rewrite_query(
            first, anthropic_api_key="", learning_store=store, settings=settings
        )

    await _ask(tracker, first)
    await _ask(tracker, second)
    await Promoter(store).mine_synonyms()

    assert await _synonym_rows(store, tag) == []
    assert await rewritten() == [first]

    # The step back from second to first is one sighting of the reverse pair
    await _ask(tracker, first)
    await _ask(tracker, second)
    await Promoter(store).mine_synonyms()

    rows = await _synonym_rows(store, tag)
    assert [(r["term"], r["synonym"]) for r in rows] == [(term, synonym)]
    assert rows[0]["confidence"] == pytest.approx(0.7)
    assert rows[0]["usage_count"] == 2
    assert [r["synonym"] for r in await store.find_synonyms(first)] == [synonym]
    assert await rewritten() == [f"{first} ({synonym})"]


@pytest.mark.asyncio
async def test_promoter_rerun_keeps_one_row_with_no_layer(store, tag):
    tracker = _tracker(store)
    first = f"renew tls certificate{tag}"
    second = f"renew tls cert{tag}"
    for question in (first, second, first, second):
        await _ask(tracker, question)

    for _ in range(3):
        await Promoter(store).mine_synonyms()

    rows = await _synonym_rows(store, tag)
    assert len(rows) == 1
    assert rows[0]["layer"] is None
    assert rows[0]["usage_count"] == 2


@pytest.mark.asyncio
async def test_old_mined_rows_are_ignored_then_purged(store, tag):
    mined, manual = f"oldmined{tag}", f"oldmanual{tag}"
    for term, source in ((mined, "pattern_mining"), (manual, "manual")):
        await store._execute(
            "INSERT INTO strata_learning.query_synonyms (term, synonym, source, confidence)"
            " VALUES ($1, 'x', $2, 0.6)",
            term,
            source,
        )

    assert await store.find_synonyms(f"{mined} {manual}") == []

    await Promoter(store).mine_synonyms()

    assert [r["term"] for r in await _synonym_rows(store, tag)] == [manual]


@pytest.mark.asyncio
async def test_old_reformulation_signals_do_not_feed_mining(store, tag):
    """A signal stored under the old rule is re-checked before it teaches anything."""
    tracker = _tracker(store)
    first = f"How do I bake{tag}?"
    second = f"How do I configure{tag}?"
    for question in (first, second, first, second):
        await _ask(tracker, question)
    # Today's detector stays silent here, so store the old rule's signals by hand
    rows = await store._fetch(
        "SELECT id FROM strata_learning.interaction_events WHERE query_text = $1", second
    )
    for row in rows:
        await store.store_feedback(str(row["id"]), "implicit", "reformulation", -0.5)

    await Promoter(store).mine_synonyms()

    assert await _synonym_rows(store, tag) == []


@pytest.mark.asyncio
async def test_top_synonyms_show_layer_and_usage(store, tag):
    layer = f"layer{tag}"
    await store.upsert_synonym(
        f"term{tag}", "x", layer=layer, confidence=0.7, usage_count=10_000_000
    )

    for stats in (await store.get_stats(layer=layer), await store.get_stats()):
        row = next(r for r in stats["top_synonyms"] if r["term"] == f"term{tag}")
        assert row["layer"] == layer
        assert row["usage_count"] == 10_000_000


@pytest.mark.asyncio
async def test_layer_switch_steers_routing(store, tag):
    tracker = _tracker(store)
    question = f"instant payment timeout {tag}"

    # Routing picked iso20022; the user then named sepa for the same question
    await tracker.track_search(_search_returning(["iso20022"]), question, None, 5, None)
    await tracker.track_search(_search_returning(), question, ["sepa"], 5, None)

    signals = await store._fetch(
        """
        SELECT fs.signal_name FROM strata_learning.feedback_signals fs
        JOIN strata_learning.interaction_events ie ON ie.id = fs.interaction_id
        WHERE ie.query_text = $1
        """,
        question,
    )
    assert [s["signal_name"] for s in signals] == ["layer_switch"]

    await Promoter(store).update_layer_affinities()

    assert await store.find_layer_affinities(question) == ["sepa"]
