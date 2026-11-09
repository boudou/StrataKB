"""Exact lookup of clause and section references (stratakb.rag.clause_lookup).

The reference parsing and trimming tests are pure. The tool tests ingest the
fixture contract through the real pipeline (embeddings mocked) into the test
database, then call strata_ask and strata_search in-process with the search
patched.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from stratakb.rag.clause_lookup import (
    ClauseMatches,
    Reference,
    _exact_chunk,
    cap_note,
    find_references,
    fit_clauses,
    fit_to_budget,
    max_exact_chars,
    question_pattern,
    unanchored,
    without_exact_spans,
)
from stratakb.rag.passages import pack_passages, passage_cost
from stratakb.rag.retriever import SearchResult
from tests.helpers import ingest

FIXTURES = Path(__file__).parent / "fixtures"
CLAUSE_PATH = "2 Termination > 2.2 Termination for cause"


def _labels(question: str, extra=()) -> list[tuple[str, ...]]:
    return [ref.labels for ref in find_references(question, extra)]


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


class TestFindReferences:
    def test_clause_number(self):
        assert find_references("What does clause 7.2 say?") == [
            Reference("clause 7.2", ("7.2", "clause 7.2"))
        ]

    def test_section_sign(self):
        assert _labels("show § 4") == [("4", "section 4")]
        assert _labels("show §4(2)") == [("4(2)", "section 4(2)")]

    def test_article_and_annex(self):
        assert _labels("Article 12 and Annex II") == [
            ("12", "article 12"),
            ("annex ii",),
        ]
        assert _labels("see appendix A") == [("appendix a",)]

    def test_statute_style_number(self):
        assert _labels("what is 7(2)(b)") == [("7(2)(b)",)]
        assert _labels("section 7(2)(b) applies") == [("7(2)(b)", "section 7(2)(b)")]

    def test_bare_dotted_number(self):
        assert _labels("explain 4.3.2 please") == [("4.3.2",)]

    def test_preset_pattern(self):
        assert _labels("show AT-44") == []
        assert _labels("show AT-44", [r"(?P<label>AT-\d+)\b"]) == [("at-44",)]

    def test_preset_pattern_anchored_to_a_line_start(self):
        # Ingestion matches the pattern at a line start; in a question the id
        # stands anywhere, but still at a word start
        anchored = r"^(?P<label>AT-R?\d{1,3})\b"
        assert _labels("show AT-44 and AT-R1", [anchored]) == [("at-44",), ("at-r1",)]
        assert _labels("see FLAT-12", [anchored]) == []
        assert _labels("show AT-44", [r"(?i)^(?P<label>at-\d+)"]) == [("at-44",)]

    def test_preset_pattern_with_context_looks_for_its_label(self):
        # The SEPA pattern needs 'Identification:' at ingest; a question
        # names the attribute alone
        sepa = r"^Identification:\s*(?P<label>AT-[A-Z]?\d{1,3})\b"
        assert _labels("show AT-R051 and AT-44", [sepa]) == [("at-r051",), ("at-44",)]
        assert _labels("see FLAT-12 or AT-R0512x", [sepa]) == []
        iso = r"^(?P<label>(?:pain|pacs|camt)\.\d{3}\.\d{3}\.\d{2})\b"
        assert _labels("what is in pacs.008.001.08?", [iso]) == [("pacs.008.001.08",)]

    def test_question_pattern(self):
        assert question_pattern(r"^Id:\s*(?P<label>AT-\d+)\b") == r"(?<!\w)(?P<label>AT-\d+)(?!\w)"
        assert question_pattern(r"(?i)^x (?P<label>[^)\]]+(a|b))") == (
            r"(?i)(?<!\w)(?P<label>[^)\]]+(a|b))(?!\w)"
        )
        assert question_pattern(r"^AT-\d+") == r"(?<!\w)AT-\d+"
        # A label group that refers to another group falls back to the whole pattern
        assert _labels("show AT-1AT", [r"^(AT)-(?P<label>\d+\1)"]) == [("1at",)]

    def test_unanchored(self):
        assert unanchored(r"^AT-\d+") == r"(?<!\w)AT-\d+"
        assert unanchored(r"(?i)^at") == r"(?i)(?<!\w)at"
        assert unanchored(r"AT-\d+") == r"AT-\d+"
        assert unanchored(r"a^b") == r"a^b"

    def test_reference_past_the_cap_is_not_read(self):
        pattern = [r"^(?P<label>AT-\d+)\b"]
        assert _labels("filler " * 400 + "show AT-44", pattern) == []
        assert _labels("show AT-44 " + "filler " * 400, pattern) == [("at-44",)]
        # Characters 1998-2002 hold AT-44: the cut must not leave AT-4
        question = "x" * 1997 + " AT-44 and more"
        assert _labels(question, pattern) == []
        assert _labels("clause 7.2 " + "filler " * 15_000) == [("7.2", "clause 7.2")]

    def test_patterns_read_at_most_the_cap(self):
        from stratakb.rag.layer_detect import MAX_PATTERN_QUESTION_CHARS

        class Recorder:
            lengths: list[int] = []

            def finditer(self, text):
                self.lengths.append(len(text))
                return iter(())

        recorder = Recorder()
        assert find_references("word " * 20_000, [recorder]) == []
        assert recorder.lengths and max(recorder.lengths) <= MAX_PATTERN_QUESTION_CHARS

    def test_year_is_not_a_reference(self):
        assert find_references("what changed in 2026") == []
        assert find_references("the 05.11.2026 release") == []
        assert find_references("the 7 days notice") == []

    def test_words_are_not_numerals(self):
        assert find_references("which section is about payments?") == []
        assert find_references("the part in question") == []

    def test_repeats_are_dropped(self):
        assert _labels("clause 2.1, then clause 2.1 again, and 2.2") == [
            ("2.1", "clause 2.1"),
            ("2.2",),
        ]


# ---------------------------------------------------------------------------
# Trimming and deduplication
# ---------------------------------------------------------------------------


def _exact(text: str, start: int = 100) -> dict:
    return {
        "chunk_id": None, "content": text, "score": 1.0, "layer": "docs",
        "source_file": "a.md", "metadata": {}, "document_id": 1,
        "char_start": start, "char_end": start + len(text), "clause_path": "7 X",
        "match": "exact",
    }


def test_fit_to_budget_keeps_offsets():
    text = " ".join(f"word{i}" for i in range(2000))
    chunk = _exact(text)
    budget = 300

    trimmed = fit_to_budget(chunk, budget)

    assert passage_cost(trimmed) <= budget
    assert text.startswith(trimmed["content"])
    assert trimmed["char_end"] - trimmed["char_start"] == len(trimmed["content"])
    assert not trimmed["content"].endswith((" ", "wor"))
    assert trimmed["cut"] is True
    assert fit_to_budget(_exact("short"), budget)["content"] == "short"


def test_max_exact_chars():
    assert max_exact_chars(1500) == 6000
    assert max_exact_chars(10) == 200


def test_fit_clauses_shares_what_a_floor_leaves():
    text = "word " * 400
    chunks = [_exact(text) | {"clause_path": "x" * 400}, _exact(text, 3000), _exact(text, 6000)]
    budget = 600

    even = [fit_to_budget(c, budget // 3) for c in chunks]
    assert passage_cost(even[0]) > budget // 3
    assert len(pack_passages(even, budget)) == 2

    assert len(pack_passages(fit_clauses(chunks, budget), budget)) == 3


def test_fit_clauses_drops_nothing():
    text = "word " * 400
    chunks = [_exact(text, start) for start in (0, 3000, 6000)]

    fitted = fit_clauses(chunks, 10)

    assert [c["char_start"] for c in fitted] == [0, 3000, 6000]
    assert all(len(c["content"]) <= 200 and c["cut"] for c in fitted)


def test_fit_to_budget_floor_is_its_cheapest_cut():
    chunk = _exact("termination " * 200)

    floor = passage_cost(fit_to_budget(chunk, 1))

    assert floor == min(passage_cost(fit_to_budget(chunk, b)) for b in range(1, 300))


def test_fit_clauses_keeps_what_an_even_split_keeps():
    chunks = [
        _exact("word " * 60, 100) | {"clause_path": "x" * 40},
        _exact("word " * 200, 5000) | {"clause_path": "x" * 3},
        _exact("word " * 200, 10000) | {"clause_path": "x" * 120},
    ]
    budget = 500

    even = [fit_to_budget(c, budget // 3) for c in chunks]
    assert len(pack_passages(even, budget)) == 3

    assert len(pack_passages(fit_clauses(chunks, budget), budget)) == 3


def test_short_clause_over_its_share_is_not_marked_cut():
    chunk = fit_to_budget(_exact("short"), 1)

    assert chunk["content"] == "short"
    assert "cut" not in chunk


class TestCapNote:
    def test_nothing_left_out(self):
        assert cap_note(ClauseMatches([]), layer_named=False) is None

    def test_candidate_limit_without_a_count(self):
        matches = ClauseMatches([], 0, True, references=1)

        assert cap_note(matches, layer_named=True) == "More sections may match than were read."

    def test_fewer_clauses_only_for_several_references(self):
        matches = ClauseMatches([], unread=3, references=2)

        assert cap_note(matches, layer_named=True) == (
            "3 more matching sections not read; ask about fewer clauses at a time."
        )
        assert cap_note(matches._replace(references=1), layer_named=True) == (
            "3 more matching sections not read."
        )

    def test_layer_advice_only_when_none_was_named(self):
        matches = ClauseMatches([], unread=2, unread_is_minimum=True, references=1)

        assert cap_note(matches, layer_named=False) == (
            "At least 2 more matching sections not read; name a layer to narrow the question."
        )

    def test_both_pieces_of_advice(self):
        note = cap_note(ClauseMatches([], unread=1, references=2), layer_named=False)

        assert note == (
            "1 more matching section not read; ask about fewer clauses at a time, "
            "or name a layer to narrow the question."
        )

    def test_references_past_the_cap_are_named(self):
        one = ClauseMatches([], references=10, skipped_references=("Annex II",))
        assert cap_note(one, layer_named=True) == (
            "1 more reference not looked up: Annex II. Ask about at most 10 clauses at a time."
        )

        both = ClauseMatches(
            [], unread=2, references=10, skipped_references=("4.1", "4.2")
        )
        assert cap_note(both, layer_named=True) == (
            "2 more matching sections not read; ask about fewer clauses at a time. "
            "2 more references not looked up: 4.1, 4.2. Ask about at most 10 clauses at a time."
        )


async def test_lookup_names_references_past_the_cap():
    from stratakb.rag import clause_lookup

    question = ", ".join(f"3.{i}" for i in range(1, 12))
    resolve = AsyncMock(return_value=ClauseMatches([]))

    with patch.object(clause_lookup, "resolve", resolve):
        matches = await clause_lookup.lookup("postgresql://unused", question, None, 1500)

    assert len(resolve.call_args.args[1]) == clause_lookup.MAX_REFERENCES
    assert matches.chunks == []
    assert matches.references == 10
    assert matches.skipped_references == ("3.11",)


def test_chunks_inside_an_exact_span_are_dropped():
    exact = [_exact("x" * 50, start=100)]
    chunks = [
        {"document_id": 1, "char_start": 110, "char_end": 140},
        {"document_id": 1, "char_start": 140, "char_end": 200},
        {"document_id": 2, "char_start": 110, "char_end": 140},
        {"document_id": None, "char_start": None, "char_end": None},
    ]
    assert without_exact_spans(chunks, exact) == chunks[1:]


# ---------------------------------------------------------------------------
# strata_ask and strata_search (database)
# ---------------------------------------------------------------------------


def _payload(result) -> dict:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return result
    return json.loads(result[0].text)


@pytest.fixture
def settings():
    from stratakb.config import Settings, get_settings

    return Settings(
        _env_file=None,
        database_url=get_settings().database_url,
        learning_enabled=False,
        reranking_enabled=False,
        strata_user="test-user",
        strata_role="viewer",
    )


@pytest.fixture
def audit():
    return AsyncMock()


@pytest.fixture
def server(settings, audit):
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.rag.tools.write_audit_log", audit),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(side_effect=lambda question, **_: [question]),
        ),
    ):
        yield build_server(settings)


async def _make_layer(db_url: str) -> str:
    from stratakb.knowledge.layers import create_layer

    name = f"test_clause_{uuid4().hex[:10]}"
    await create_layer(
        db_url, name=name, display_name="Test Clause",
        chunk_config={"chunk_size": 200, "chunk_overlap": 40},
    )
    return name


async def _ingest_ok(db_url: str, layer: str, source: Path) -> None:
    result = await ingest(db_url, layer, source)
    assert result.docs_failed == 0


@pytest.fixture
async def layers(settings, tmp_path):
    """Two layers: one holding contract.md, one holding it twice under other names."""
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer

    db_url = settings.database_url
    one, two = await _make_layer(db_url), await _make_layer(db_url)
    single = tmp_path / "single"
    single.mkdir()
    shutil.copy(FIXTURES / "contract.md", single / "contract.md")
    double = tmp_path / "double"
    double.mkdir()
    shutil.copy(FIXTURES / "contract.md", double / "acme.md")
    shutil.copy(FIXTURES / "contract.md", double / "globex.md")
    await _ingest_ok(db_url, one, single)
    await _ingest_ok(db_url, two, double)
    yield one, two
    for name in (one, two):
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


async def _document(db_url: str, layer: str, relative_path: str) -> dict:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    row = await pool.fetchrow(
        """
        SELECT d.id, d.text FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1 AND d.relative_path = $2
        ORDER BY d.version DESC LIMIT 1
        """,
        layer,
        relative_path,
    )
    return dict(row)


def _retrieved(layer: str, document_id: int | None = None, span=None) -> SearchResult:
    start, end = span if span else (None, None)
    return SearchResult(
        chunk_id=str(uuid4()), content="retrieved text", score=1.0, layer_name=layer,
        source_file="other.md", source_section="", doc_type="markdown",
        similarity=0.5, document_id=document_id, char_start=start, char_end=end,
    )


async def _call(server, tool: str, results, **arguments):
    search = AsyncMock(return_value=results)
    with patch("stratakb.rag.retriever.search_chunks", search):
        response = _payload(await server.call_tool(tool, arguments))
    return response, search


class TestAskClause:
    async def test_clause_22_first(self, server, settings, layers, audit):
        layer, _ = layers
        doc = await _document(settings.database_url, layer, "contract.md")
        expected = doc["text"][doc["text"].index("2.2 Termination for cause"):
                               doc["text"].index("\nAnnex II\n")].rstrip()

        response, search = await _call(
            server, "strata_ask", [_retrieved(layer)],
            question="what does clause 2.2 say", layers=[layer],
        )

        assert response["status"] == "found"
        first = response["passages"][0]
        start = doc["text"].index(expected)
        assert first["ref"] == f"d{doc['id']}:{start}-{start + len(expected)}"
        assert first["text"] == expected == doc["text"][start:start + len(expected)]
        citation = first["citation"]
        assert citation["file"] == "contract.md"
        assert citation["clause_path"] == CLAUSE_PATH
        assert citation["layer"] == layer
        assert citation["lines"] == "15-28"
        # Retrieval still adds its passages after the exact one
        assert search.await_count == 1
        assert [p["text"] for p in response["passages"][1:]] == ["retrieved text"]
        assert audit.await_args.kwargs["parameters"]["exact_matches"] == 1

    async def test_detailed_marks_the_exact_chunk(self, server, settings, layers):
        layer, _ = layers
        response, _ = await _call(
            server, "strata_ask", [],
            question="show clause 2.2", layers=[layer], response_format="detailed",
        )

        assert response["answer_status"] == "found"
        chunk = response["retrieved_chunks"][0]
        assert chunk["match"] == "exact"
        assert chunk["chunk_id"] is None
        assert chunk["clause_path"] == CLAUSE_PATH
        assert response["sources"][0]["relevance_score"] == 1.0

    async def test_same_clause_two_documents(self, server, settings, layers):
        _, layer = layers

        response, _ = await _call(
            server, "strata_ask", [],
            question="what does clause 2.2 say", layers=[layer],
        )

        exact = response["passages"][:2]
        assert sorted(p["citation"]["file"] for p in exact) == ["acme.md", "globex.md"]
        assert {p["citation"]["clause_path"] for p in exact} == {CLAUSE_PATH}
        assert exact[0]["text"] == exact[1]["text"]
        assert len(response["passages"]) == 2

    async def test_lookup_stays_in_the_layers_searched(self, server, settings, layers):
        one, _ = layers

        response, _ = await _call(
            server, "strata_ask", [], question="clause 2.2", layers=[one],
        )

        assert [p["citation"]["file"] for p in response["passages"]] == ["contract.md"]

    async def test_retrieved_chunk_inside_the_clause_is_dropped(
        self, server, settings, layers
    ):
        layer, _ = layers
        doc = await _document(settings.database_url, layer, "contract.md")
        start = doc["text"].index("Either party may terminate this agreement with")
        inside = _retrieved(layer, doc["id"], (start, start + 40))

        response, _ = await _call(
            server, "strata_ask", [inside], question="clause 2.2", layers=[layer],
        )

        assert len(response["passages"]) == 1
        assert response["passages"][0]["citation"]["clause_path"] == CLAUSE_PATH

    async def test_unknown_clause_falls_back(self, server, settings, layers, audit):
        layer, _ = layers

        response, search = await _call(
            server, "strata_ask", [_retrieved(layer)],
            question="what does clause 9.4 say", layers=[layer], response_format="detailed",
        )

        assert search.await_count == 1
        assert [c.get("match") for c in response["retrieved_chunks"]] == [None]
        assert [p["text"] for p in response["passages"]] == ["retrieved text"]
        assert audit.await_args.kwargs["parameters"]["exact_matches"] == 0

    async def test_long_clause_is_trimmed_to_the_budget(self, server, settings, layers):
        layer, _ = layers
        settings.answer_token_budget = 120
        doc = await _document(settings.database_url, layer, "contract.md")

        response, _ = await _call(
            server, "strata_ask", [], question="clause 2", layers=[layer],
        )

        first = response["passages"][0]
        start = doc["text"].index("2. Termination")
        assert first["text"] == doc["text"][start:start + len(first["text"])]
        assert first["ref"] == f"d{doc['id']}:{start}-{start + len(first['text'])}"
        assert len(first["text"]) < len("2. Termination\n\n2.1 Notice") + 200

    async def test_cut_clause_is_partly_found(self, server, settings, layers):
        layer, _ = layers
        settings.answer_token_budget = 120

        response, _ = await _call(
            server, "strata_ask", [], question="clause 2", layers=[layer],
        )

        assert response["status"] == "partly_found"
        assert response["mode"] == "grounded"
        assert response["note"].startswith("2 Termination is only partly returned")
        assert "contract.md" in response["note"]
        assert response["next_step"]
        assert "cut" not in response["passages"][0]

        detailed, _ = await _call(
            server, "strata_ask", [], question="clause 2", layers=[layer],
            response_format="detailed",
        )

        assert detailed["status"] == "ok"
        assert detailed["answer_status"] == "partly_found"
        assert detailed["note"] == response["note"]

    async def test_whole_clause_stays_found(self, server, settings, layers):
        layer, _ = layers

        response, _ = await _call(
            server, "strata_ask", [], question="clause 2.2", layers=[layer],
        )

        assert response["status"] == "found"
        assert "note" not in response

    async def test_long_clause_ref_verifies(self, server, settings, layers):
        # Clause 2.2 is wider than twice the layer's chunk_size (400)
        layer, _ = layers

        response, _ = await _call(
            server, "strata_ask", [], question="clause 2.2", layers=[layer],
        )

        ref = response["passages"][0]["ref"]
        start, end = (int(n) for n in ref.split(":")[1].split("-"))
        assert end - start > 400
        draft = '"Either party may terminate this agreement with immediate effect" [S1].'
        result = _payload(
            await server.call_tool("strata_verify", {"draft": draft, "passages": [ref]})
        )
        assert result["result"] == "verified", result
        assert result["receipt_id"]

    async def test_cut_clause_ref_verifies(self, server, settings, layers):
        layer, _ = layers
        settings.answer_token_budget = 120

        response, _ = await _call(
            server, "strata_ask", [], question="clause 2", layers=[layer],
        )

        ref = response["passages"][0]["ref"]
        draft = '"Either party may terminate this agreement by written notice" [S1].'
        result = _payload(
            await server.call_tool("strata_verify", {"draft": draft, "passages": [ref]})
        )
        assert result["result"] == "verified", result
        assert result["receipt_id"]

    async def test_each_cut_clause_names_its_file(self, server, settings, layers):
        _, layer = layers
        settings.answer_token_budget = 400

        response, _ = await _call(
            server, "strata_ask", [], question="clause 2", layers=[layer],
        )

        assert response["status"] == "partly_found"
        assert len(response["passages"]) == 2
        assert "acme.md" in response["note"]
        assert "globex.md" in response["note"]

    async def test_clause_over_the_budget_is_named(self, server, settings, layers):
        _, layer = layers
        settings.answer_token_budget = 150

        response, _ = await _call(
            server, "strata_ask", [], question="clause 2", layers=[layer],
        )

        assert response["status"] == "partly_found"
        assert [p["citation"]["file"] for p in response["passages"]] == ["acme.md"]
        assert "1 named clause left out" in response["note"]
        assert "globex.md" in response["note"]

        detailed, _ = await _call(
            server, "strata_ask", [], question="clause 2", layers=[layer],
            response_format="detailed",
        )

        assert detailed["answer_status"] == "partly_found"
        assert detailed["note"] == response["note"]

    async def test_matches_past_the_cap_are_noted(self, server, settings, layers):
        layer, _ = layers
        doc = await _document(settings.database_url, layer, "contract.md")
        chunk = _exact_chunk(
            {
                "layer_name": layer, "source_file": "contract.md", "path": "7 X",
                "doc_type": "markdown", "document_id": doc["id"], "char_start": 0,
            },
            doc["text"][:40],
        )

        with patch(
            "stratakb.rag.tools._exact_clauses",
            AsyncMock(return_value=ClauseMatches([chunk], unread=3)),
        ):
            response, _ = await _call(
                server, "strata_ask", [], question="clause 2", layers=[layer],
            )
            detailed, _ = await _call(
                server, "strata_ask", [], question="clause 2", layers=[layer],
                response_format="detailed",
            )

        assert response["status"] == "partly_found"
        assert response["note"].endswith("3 more matching sections not read.")
        assert detailed["answer_status"] == "partly_found"
        assert detailed["note"] == response["note"]

    async def test_candidate_limit_is_noted(self, server, settings, layers):
        layer, _ = layers
        doc = await _document(settings.database_url, layer, "contract.md")
        chunk = _exact_chunk(
            {
                "layer_name": layer, "source_file": "contract.md", "path": "7 X",
                "doc_type": "markdown", "document_id": doc["id"], "char_start": 0,
            },
            doc["text"][:40],
        )

        with patch(
            "stratakb.rag.tools._exact_clauses",
            AsyncMock(return_value=ClauseMatches([chunk], 0, True, references=1)),
        ):
            response, _ = await _call(
                server, "strata_ask", [], question="clause 2", layers=[layer],
            )

        assert response["status"] == "partly_found"
        assert response["note"].endswith("More sections may match than were read.")

    async def test_eleventh_reference_is_named(self, server, settings, layers):
        layer, _ = layers
        question = (
            "clauses 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10 and Annex II"
        )

        response, _ = await _call(
            server, "strata_ask", [], question=question, layers=[layer],
        )

        assert (
            "1 more reference not looked up: Annex II. Ask about at most 10 clauses at a time."
            in response["note"]
        )

    async def test_search_keeps_a_clause_over_the_budget(self, server, settings, layers):
        _, layer = layers
        settings.answer_token_budget = 150

        response, _ = await _call(
            server, "strata_search", [], query="clause 2", layers=[layer],
        )

        exact = [r for r in response["results"] if r.get("match") == "exact"]
        assert [r["source_file"] for r in exact] == ["acme.md", "globex.md"]


async def test_cut_clause_rule_in_the_contract(server):
    assert "cut true" in server.instructions
    assert "strata_search returns a note" in server.instructions
    assert "or not at all" in server.instructions
    assert len(server.instructions) <= 4000
    prompt = await server.get_prompt("answer_from_documents", {"question": "q"})
    assert "cut true" in prompt.messages[0].content.text


async def test_resolve_caps_a_label_many_documents_share(settings, tmp_path):
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer
    from stratakb.rag.clause_lookup import MAX_EXACT_MATCHES, resolve

    db_url = settings.database_url
    layer = await _make_layer(db_url)
    names = [f"doc{i:02d}.md" for i in range(MAX_EXACT_MATCHES + 3)]
    for name in names:
        shutil.copy(FIXTURES / "contract.md", tmp_path / name)
    try:
        await _ingest_ok(db_url, layer, tmp_path)

        matches = await resolve(db_url, find_references("clause 2.2"), [layer])
        chunks = matches.chunks

        assert len(chunks) == MAX_EXACT_MATCHES
        assert matches.unread == 3
        assert matches.unread_is_minimum is False
        assert [c["source_file"] for c in chunks] == names[:MAX_EXACT_MATCHES]
        assert {c["clause_path"] for c in chunks} == {CLAUSE_PATH}
        assert all(c["match"] == "exact" and c["content"] for c in chunks)
    finally:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=layer)


async def test_match_cap_reaches_ask_and_search(server, settings, tmp_path):
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer
    from stratakb.rag.clause_lookup import MAX_EXACT_MATCHES

    db_url = settings.database_url
    layer = await _make_layer(db_url)
    for i in range(MAX_EXACT_MATCHES + 3):
        shutil.copy(FIXTURES / "contract.md", tmp_path / f"doc{i:02d}.md")
    try:
        await _ingest_ok(db_url, layer, tmp_path)

        asked, _ = await _call(
            server, "strata_ask", [], question="clause 2.2", layers=[layer],
        )
        searched, _ = await _call(
            server, "strata_search", [], query="clause 2.2", layers=[layer],
        )

        assert asked["status"] == "partly_found"
        assert "3 more matching sections not read." in asked["note"]
        exact = [r for r in searched["results"] if r.get("match") == "exact"]
        assert len(exact) == MAX_EXACT_MATCHES
        assert searched["status"] == "ok"
        assert searched["note"] == "3 more matching sections not read."
    finally:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=layer)


async def test_resolve_says_when_the_count_stopped(settings, tmp_path, monkeypatch):
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer
    from stratakb.rag import clause_lookup

    monkeypatch.setattr(clause_lookup, "_MAX_CANDIDATE_SECTIONS", 12)
    db_url = settings.database_url
    layer = await _make_layer(db_url)
    for i in range(13):
        shutil.copy(FIXTURES / "contract.md", tmp_path / f"doc{i:02d}.md")
    try:
        await _ingest_ok(db_url, layer, tmp_path)

        matches = await clause_lookup.resolve(db_url, find_references("clause 2.2"), [layer])

        assert len(matches.chunks) == clause_lookup.MAX_EXACT_MATCHES
        assert matches.unread == 2
        assert matches.unread_is_minimum is True
    finally:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=layer)


def test_exact_chunk_has_the_ask_chunk_shape():
    from stratakb.rag.tools import _ask_chunk

    row = {
        "layer_name": "docs", "source_file": "a.md", "path": "7 X", "doc_type": "markdown",
        "document_id": 1, "char_start": 10,
    }
    exact = _exact_chunk(row, "text")
    retrieved = _ask_chunk(_retrieved("docs", 1, (10, 14)), 0.5)

    assert list(exact) == [*retrieved, "match"]
    assert exact["char_end"] == 14


_RULEBOOK = (
    "Attributes\n==========\n\n"
    "Identification: AT-44\n"
    "Name: The amount of the credit transfer in euro\n"
    "Description: The amount is given in euro, with two decimals.\n\n"
    "Identification: AT-45\n"
    "Name: The requested execution date\n"
    "Description: The date on which the bank debits the account.\n"
)


@pytest.fixture
async def rulebook_layer(settings, tmp_path):
    """A layer with the SEPA preset's attribute pattern, holding one rulebook text."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer
    from stratakb.knowledge.presets import load_preset

    db_url = settings.database_url
    name = f"test_clause_{uuid4().hex[:10]}"
    sepa = Path(__file__).parent.parent / "examples" / "sepa" / "preset.yaml"
    await create_layer(
        db_url, name=name, display_name="Test Rulebook",
        chunk_config={"chunk_size": 200, "chunk_overlap": 40},
        structure_patterns=load_preset(sepa).structure_patterns,
    )
    src = tmp_path / "rulebook"
    src.mkdir()
    (src / "rulebook.txt").write_text(_RULEBOOK, encoding="utf-8")
    await _ingest_ok(db_url, name, src)
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


class TestLayerPatternLookup:
    @pytest.mark.parametrize(
        ("tool", "arguments"),
        [("strata_ask", {"question": "what does AT-44 say"}),
         ("strata_search", {"query": "show AT-44"})],
    )
    async def test_layer_pattern_finds_the_attribute(
        self, server, rulebook_layer, tool, arguments
    ):
        response, _ = await _call(server, tool, [], layers=[rulebook_layer], **arguments)

        start = _RULEBOOK.index("Identification: AT-44")
        expected = _RULEBOOK[start:_RULEBOOK.index("Identification: AT-45")].rstrip()
        if tool == "strata_ask":
            first = response["passages"][0]
            assert first["text"] == expected
            assert first["citation"]["clause_path"] == "Attributes > Identification: AT-44"
        else:
            first = response["results"][0]
            assert first["match"] == "exact"
            assert first["content"] == expected

    async def test_other_layers_patterns_are_not_used(self, server, rulebook_layer, layers):
        # The contract layer has no patterns, so AT-44 names nothing there
        contract_layer, _ = layers
        response, _ = await _call(
            server, "strata_search", [_retrieved(contract_layer)],
            query="show AT-44", layers=[contract_layer],
        )

        assert [r["content"] for r in response["results"]] == ["retrieved text"]


class TestSearchClause:
    async def test_exact_result_first(self, server, settings, layers, audit):
        layer, _ = layers

        response, search = await _call(
            server, "strata_search", [_retrieved(layer)],
            query="clause 2.2", layers=[layer],
        )

        assert search.await_count == 1
        first, second = response["results"]
        assert first["match"] == "exact"
        assert first["section"] == CLAUSE_PATH
        assert first["content"].startswith("2.2 Termination for cause")
        assert first["chunk_id"] is None
        assert second["content"] == "retrieved text"
        assert "match" not in second
        assert response["total"] == 2
        assert audit.await_args.kwargs["parameters"]["exact_matches"] == 1

    async def test_cut_clause_is_marked(self, server, settings, layers):
        layer, _ = layers
        settings.answer_token_budget = 120

        response, _ = await _call(
            server, "strata_search", [_retrieved(layer)],
            query="clause 2", layers=[layer],
        )

        assert response["status"] == "ok"
        first, second = response["results"]
        assert first["cut"] is True
        assert "cut" not in second

        settings.answer_token_budget = 1500
        response, _ = await _call(
            server, "strata_search", [], query="clause 2.2", layers=[layer],
        )

        assert "cut" not in response["results"][0]

    async def test_unknown_reference_is_plain_search(self, server, settings, layers):
        layer, _ = layers

        response, _ = await _call(
            server, "strata_search", [_retrieved(layer)],
            query="version 7.25", layers=[layer],
        )

        assert [r["content"] for r in response["results"]] == ["retrieved text"]
