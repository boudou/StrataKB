"""The gold set scorer in extractive mode: sentence scores, quoted sentences, marks.

Only the test named so uses the test database.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.gold_helpers import SCORER, _load, _score

EXTRACTIVE_KEYS = {"top_sentence_score", "named_clause", "quoted"}


def test_extractive_mode_is_asked_for(scorer_input, tmp_path):
    asked, plain = [], []
    _score(scorer_input, "peps_small", tmp_path / "a.json", "--mode", "extractive", seen=asked)
    _score(scorer_input, "peps_small", tmp_path / "b.json", seen=plain)

    assert asked
    assert all(arguments["mode"] == "extractive" for arguments in asked)
    assert plain
    assert not any("mode" in arguments for arguments in plain)


def _sentence(text, exact=False):
    from stratakb.rag.extractive import Sentence

    return Sentence(1, 0, len(text), text, 0, exact)


def _scored_reply(candidates):
    """A strata_ask that scores candidates as the server does, then quotes none of them."""

    def reply(_arguments):
        import stratakb.rag.tools

        kept = stratakb.rag.tools.select_sentences("?", candidates, 0.5, "model")
        assert not [s for s in kept if not s.exact]
        return {
            "status": "ok", "mode": "extractive", "answer_status": "not_found",
            "answer": [], "passages": [],
        }

    return reply


@pytest.mark.parametrize("exact", [False, True], ids=["scored", "named-clause"])
def test_extractive_rows_record_the_top_sentence_score(scorer_input, tmp_path, exact):
    import stratakb.rag.reranker
    import stratakb.rag.tools

    select, scoring = stratakb.rag.tools.select_sentences, stratakb.rag.reranker.score_pairs
    candidates = [_sentence("first one."), _sentence("second one.")]
    if exact:
        candidates.append(_sentence("7.2 the named clause.", exact=True))
    out = tmp_path / "out.json"

    with patch("stratakb.rag.reranker.score_pairs", return_value=[0.42, 0.31]) as score_pairs:
        _score(
            scorer_input, "peps_small", out, "--mode", "extractive",
            reply=_scored_reply(candidates),
        )
        # The server's own call, once per question: no second pass
        assert score_pairs.call_count == 3

    assert stratakb.rag.tools.select_sentences is select
    assert stratakb.rag.reranker.score_pairs is scoring
    result = json.loads(out.read_text(encoding="utf-8"))
    refusal = next(row for row in result["rows"] if row["id"] == "r1")
    assert refusal["mode"] == "extractive"
    assert refusal["status"] == "not_found"
    assert refusal["top_sentence_score"] == 0.42
    assert refusal["named_clause"] is exact
    assert refusal["quoted"] == []
    marks = {m["mark"]: m for m in result["summary"]["marks"]}
    if exact:
        # A named clause is quoted without a score, so at every mark
        assert {m["refused"]["not_in_corpus"] for m in marks.values()} == {0.0}
        assert {m["answered"]["all"] for m in marks.values()} == {100.0}
    else:
        assert marks[0.4]["refused"]["not_in_corpus"] == 0.0
        assert marks[0.45]["refused"]["not_in_corpus"] == 100.0
        assert marks[0.45]["refused"]["other_layer"] == 100.0
        assert marks[0.45]["refused"]["off_domain"] == 100.0
        assert (marks[0.4]["answered"]["all"], marks[0.45]["answered"]["all"]) == (100.0, 0.0)


def test_the_hook_is_put_back_when_the_run_stops(scorer_input, tmp_path):
    import stratakb.rag.reranker
    import stratakb.rag.tools

    select, scoring = stratakb.rag.tools.select_sentences, stratakb.rag.reranker.score_pairs

    def reply(arguments):
        import stratakb.rag.tools

        assert stratakb.rag.tools.select_sentences is not select
        return {"status": "error", "error": "the model is broken"}

    with pytest.raises(SystemExit):
        _score(scorer_input, "peps_small", tmp_path / "out.json", "--mode", "extractive",
               reply=reply)

    assert stratakb.rag.tools.select_sentences is select
    assert stratakb.rag.reranker.score_pairs is scoring


def test_extractive_rows_keep_what_was_quoted(scorer_input, tmp_path):
    def reply(_arguments):
        return {
            "status": "ok", "mode": "extractive", "answer_status": "found", "passages": [],
            "answer": [
                {"text": f"sentence {i} " + "x" * 400, "score": 0.9,
                 "citation": {"file": "a.md"}}
                for i in range(4)
            ],
        }

    out = tmp_path / "out.json"
    _score(scorer_input, "peps_small", out, "--mode", "extractive", reply=reply)

    row = json.loads(out.read_text(encoding="utf-8"))["rows"][0]
    # Nothing was scored by the stub, so no score and no named clause
    assert (row["top_sentence_score"], row["named_clause"]) == (None, False)
    assert len(row["quoted"]) == 3
    assert row["quoted"][0]["file"] == "a.md"
    assert row["quoted"][0]["score"] == 0.9
    assert len(row["quoted"][0]["text"]) == 300


def test_summary_marks_sweep_the_candidates_and_the_run_mark(scorer_input, tmp_path, monkeypatch):
    monkeypatch.setenv("RERANKING_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
    out = tmp_path / "out.json"

    with patch("stratakb.rag.reranker.score_pairs", return_value=[0.42, 0.31]):
        _score(
            scorer_input, "peps_small", out, "--mode", "extractive",
            reply=_scored_reply([_sentence("first one."), _sentence("second one.")]),
        )

    marks = [m["mark"] for m in json.loads(out.read_text(encoding="utf-8"))["summary"]["marks"]]
    assert marks == sorted({*(round(0.3 + 0.05 * i, 2) for i in range(14)), 0.86})
    assert (marks[0], marks[-1]) == (0.3, 0.95)


def test_grounded_results_have_no_extractive_keys(scorer_input, tmp_path):
    out = tmp_path / "out.json"
    _score(scorer_input, "peps_small", out, "--mode", "grounded")

    result = json.loads(out.read_text(encoding="utf-8"))
    assert "marks" not in result["summary"]
    assert {row["mode"] for row in result["rows"]} == {"grounded"}
    assert not any(EXTRACTIVE_KEYS & row.keys() for row in result["rows"])


@pytest.mark.parametrize(
    ("model", "mark"),
    [
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", 0.5),
        ("Alibaba-NLP/gte-reranker-modernbert-base", 0.86),
    ],
    ids=["default", "gte"],
)
@pytest.mark.parametrize("mode", [None, "extractive"])
def test_run_block_names_the_extractive_mark(
    scorer_input, tmp_path, monkeypatch, model, mark, mode
):
    monkeypatch.setenv("RERANKING_MODEL", model)
    out = tmp_path / "out.json"
    _score(scorer_input, "peps_small", out, *(["--mode", mode] if mode else []))

    answer = json.loads(out.read_text(encoding="utf-8"))["run"]["answer"]
    assert answer["mode"] == mode
    assert answer["extractive"] == {"reranking_model": model, "mark": mark}


async def test_database_extractive_scoring_records_sentence_scores(
    db_url, filled_layer, tmp_path, monkeypatch, no_sdk_env
):
    """The real strata_ask: a refusal whose best sentence scores under the mark."""
    from stratakb.core.database import get_pool
    from stratakb.knowledge.documents import store_document_version
    from stratakb.rag.retriever import SearchResult
    from tests.helpers import DOC_TEXT

    monkeypatch.setenv("RERANKING_ENABLED", "true")
    monkeypatch.setenv("RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    pool = await get_pool(db_url)
    layer_id = await pool.fetchval(
        "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", filled_layer
    )
    async with pool.acquire() as conn, conn.transaction():
        document_id, _ = await store_document_version(
            conn, layer_id=layer_id, file_id=None, source_root="/docs",
            relative_path="contracts/acme-msa.md", text=DOC_TEXT, page_map=None,
            text_origin="text",
        )
    hit = SearchResult(
        chunk_id=f"hit-{uuid4().hex[:6]}", content="content", score=1.0,
        layer_name=filled_layer, source_file="hit.md", source_section="section",
        doc_type="markdown", metadata={}, similarity=0.9,
        document_id=document_id, char_start=0, char_end=len(DOC_TEXT),
    )
    gold = [{"id": "r1", "kind": "refusal", "group": "off_domain", "question": "refuse?"}]
    out = tmp_path / "out.json"

    with (
        patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[hit])),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(side_effect=lambda question, **_: [question]),
        ),
        patch(
            "stratakb.rag.reranker.score_pairs",
            side_effect=lambda _q, texts, _m: [0.9 if t == "content" else 0.42 for t in texts],
        ),
    ):
        await _load(SCORER).score(gold, filled_layer, set(), out, mode="extractive")

    [row] = json.loads(out.read_text(encoding="utf-8"))["rows"]
    assert (row["mode"], row["status"]) == ("extractive", "not_found")
    assert row["top_sentence_score"] == 0.42
    assert row["named_clause"] is False
    # The scorer closed its pool on the way out
    pool = await get_pool(db_url)
    assert not await pool.fetchval(
        "SELECT count(*) FROM strata_core.receipt_documents WHERE document_id = $1", document_id
    )
