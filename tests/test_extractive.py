"""Extractive answers: the document's own sentences, chosen without a language model.

The splitting, mode and selection tests are pure. The tool tests call
strata_ask in-process with the search and the cross-encoder patched; layers,
documents and receipts use the test database.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from stratakb.knowledge.layers import ANSWER_MODES
from stratakb.rag.confidence import RERANK_THRESHOLDS
from stratakb.rag.extractive import (
    GROUNDED_REFUSED_NOTE,
    LEGACY_ONLY_NOTE,
    MAX_SENTENCE_CHARS,
    MAX_SENTENCES,
    OCR_WARNING,
    Sentence,
    answer_item,
    candidate_sentences,
    effective_mode,
    fit_to_answer_budget,
    select_sentences,
    split_sentences,
)
from stratakb.rag.passages import AnswerMode, AskResult, ExtractiveAnswer, ask_chunk
from stratakb.rag.retriever import SearchResult
from tests.helpers import as_admin, call_tool, ingest

FIXTURES = Path(__file__).parent / "fixtures"

DOC_TEXT = (
    "# Master agreement\n"
    "\n"
    "Intro line.\n"
    "\n"
    "## 7 Termination\n"
    "\n"
    "7.2 Either party may end this agreement\n"
    "with ninety days' written notice. Notice goes by registered letter, e.g. by courier.\n"
    "\n"
    "## 8 Payment\n"
    "\n"
    "Invoices are due within 30 days. Late payment carries interest.\n"
)
NINETY = "7.2 Either party may end this agreement\nwith ninety days' written notice."
LETTER = "Notice goes by registered letter, e.g. by courier."
INVOICES = "Invoices are due within 30 days."
INTEREST = "Late payment carries interest."
# Lines without end marks: one sentence of 100 KB, the part a chunk hits at its end
BLOCK_HIT = "the fee for the annex is ninety euros "
LONG_BLOCK = "item without an end mark\n" * 4200 + BLOCK_HIT * 21


def _texts(text: str) -> list[str]:
    return [text[start:end] for start, end in split_sentences(text)]


class TestSplitSentences:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("One here. Two there! Three now?", ["One here.", "Two there!", "Three now?"]),
            ("Payment is due; interest runs.", ["Payment is due;", "interest runs."]),
            ("Use a form, e.g. the blue one. Done.", ["Use a form, e.g. the blue one.", "Done."]),
            ("Use a form, i.e. Form Blue. Done.", ["Use a form, i.e. Form Blue.", "Done."]),
            # A single letter reads as an initial: too long a quote, never half a one
            ("J. Smith signs. Done.", ["J. Smith signs.", "Done."]),
            ("See Annex A. It applies.", ["See Annex A. It applies."]),
            ("See No. 7 here. Next.", ["See No. 7 here.", "Next."]),
            ("See Art. 5 of the law. Next.", ["See Art. 5 of the law.", "Next."]),
            ("Under clause 7.2 it ends. Next.", ["Under clause 7.2 it ends.", "Next."]),
            ("7. Payment is due. It is final.", ["7. Payment is due.", "It is final."]),
            ('He said "stop." Then he left.', ['He said "stop."', "Then he left."]),
            ("No end mark", ["No end mark"]),
        ],
    )
    def test_sentences(self, text, expected):
        assert _texts(text) == expected

    def test_wrapped_line_stays_in_its_sentence(self):
        text = "Either party may end\nwith notice.\n"
        assert _texts(text) == ["Either party may end\nwith notice."]

    def test_heading_and_blank_lines_are_not_sentences(self):
        assert _texts("# Title\n\nBody text.\n\n## Next\n\nMore.\n") == ["Body text.", "More."]

    def test_blank_line_ends_a_sentence_without_a_mark(self):
        assert _texts("First part\n\nSecond part\n") == ["First part", "Second part"]

    def test_list_items_and_numbered_clauses_stand_alone(self):
        assert _texts("- one item\n- two item\n") == ["- one item", "- two item"]
        assert _texts("7.1 First rule\n7.2 Second rule\n") == ["7.1 First rule", "7.2 Second rule"]

    def test_nothing_without_a_letter_or_digit(self):
        assert _texts("...\n\n---\n\n") == []

    def test_offsets_are_into_the_text(self):
        text = "  Leading space. Next one.  "
        for start, end in split_sentences(text):
            assert text[start:end] == text[start:end].strip()
        assert split_sentences("One. Two.", start=10) == [(10, 14), (15, 19)]

    @pytest.mark.parametrize(
        "hostile",
        [
            "A. " * 20000,
            "a" * 60000 + " b. Next",
            "." * 60000 + "x",
            '."' * 30000 + "x",
            "1. " * 20000,
        ],
        ids=["initials", "unbroken-word", "dot-run", "dot-quote-run", "numbers"],
    )
    def test_hostile_text_splits_in_linear_time(self, hostile):
        began = time.perf_counter()

        split_sentences(hostile)

        # A quadratic splitter needs over ten seconds for the first three
        assert time.perf_counter() - began < 1.0

    def test_long_token_before_a_dot_still_ends_the_sentence(self):
        word = "a" * 200
        assert _texts(f"{word}. Next one.") == [f"{word}.", "Next one."]


class TestEffectiveMode:
    @pytest.mark.parametrize(
        ("requested", "layer_modes", "expected"),
        [
            (None, ["grounded"], ("grounded", None)),
            ("grounded", ["grounded"], ("grounded", None)),
            ("extractive", ["grounded"], ("extractive", None)),
            (None, ["extractive"], ("extractive", None)),
            ("extractive", ["extractive"], ("extractive", None)),
            ("grounded", ["extractive"], ("extractive", GROUNDED_REFUSED_NOTE)),
            (None, ["grounded", "extractive"], ("extractive", None)),
            (None, [], ("grounded", None)),
        ],
    )
    def test_strictest_wins(self, requested, layer_modes, expected):
        assert effective_mode(requested, layer_modes) == expected

    def test_answer_mode_type_matches_the_layer_modes(self):
        assert set(get_args(AnswerMode)) == set(ANSWER_MODES)


def _chunk(start: int, end: int, *, document_id: int = 1, **fields) -> dict:
    return {"document_id": document_id, "char_start": start, "char_end": end, **fields}


def _candidates(chunks, text=DOC_TEXT) -> list[Sentence]:
    return candidate_sentences(chunks, {1: {"text": text}}, {1: len(text) + 1000})


class TestCandidateSentences:
    def test_overlapping_chunks_give_each_sentence_once(self):
        start = DOC_TEXT.index(NINETY)
        end = DOC_TEXT.index(LETTER) + len(LETTER)

        sentences = _candidates([_chunk(start, end), _chunk(start + 10, end)])

        assert [s.text for s in sentences] == [NINETY, LETTER]
        assert {s.chunk_index for s in sentences} == {0}

    def test_chunk_cut_inside_a_sentence_gives_the_whole_sentence(self):
        inside = DOC_TEXT.index("ninety")

        sentences = _candidates([_chunk(inside, inside + 6)])

        assert [s.text for s in sentences] == [NINETY]

    def test_chunk_ending_on_a_sentence_start_leaves_it_out(self):
        start = DOC_TEXT.index(INVOICES)
        assert [s.text for s in _candidates([_chunk(0, start)])] == [
            "Intro line.", NINETY, LETTER,
        ]

    def test_sentence_cut_by_the_read_is_dropped(self):
        text = "First one here. Second one is cut"

        sentences = candidate_sentences(
            [_chunk(0, len(text))], {1: {"text": text}}, {1: len(text)}
        )

        assert [s.text for s in sentences] == ["First one here."]

    @pytest.mark.parametrize("cut", [" ", "\n", " \n  "])
    def test_sentence_cut_on_whitespace_by_the_read_is_dropped(self, cut):
        text = "First one here. Neither party shall be liable" + cut

        sentences = candidate_sentences(
            [_chunk(0, len(text))], {1: {"text": text}}, {1: len(text)}
        )

        assert [s.text for s in sentences] == ["First one here."]

    def test_same_text_ending_before_the_read_keeps_the_sentence(self):
        text = "First one here. Neither party shall be liable "

        sentences = candidate_sentences(
            [_chunk(0, len(text))], {1: {"text": text}}, {1: len(text) + 1}
        )

        assert [s.text for s in sentences] == [
            "First one here.", "Neither party shall be liable",
        ]

    def test_short_document_keeps_its_last_sentence(self):
        text = "Only one sentence without a mark"

        sentences = candidate_sentences(
            [_chunk(0, len(text))], {1: {"text": text}}, {1: len(text) + 1000}
        )

        assert [s.text for s in sentences] == [text]

    def test_chunk_without_a_document_gives_nothing(self):
        chunks = [
            {"document_id": None, "char_start": None, "char_end": None},
            _chunk(0, 10, document_id=2),
        ]
        assert _candidates(chunks) == []

    @pytest.mark.parametrize("fields", [{}, {"match": "exact"}], ids=["scored", "exact"])
    def test_block_over_the_limit_is_not_quoted(self, fields):
        text = f"{LONG_BLOCK}\n\nA short sentence.\n"
        hit_start = text.index(BLOCK_HIT)

        sentences = _candidates([_chunk(hit_start, len(text), **fields)], text)

        assert [s.text for s in sentences] == ["A short sentence."]

    def test_span_of_exactly_the_limit_is_quoted(self):
        text = "a" * MAX_SENTENCE_CHARS + "\n\nNext.\n"
        assert _candidates([_chunk(0, 10)], text)[0].end == MAX_SENTENCE_CHARS

    def test_exact_chunk_marks_its_sentences(self):
        sentences = _candidates(
            [_chunk(DOC_TEXT.index(NINETY), DOC_TEXT.index(LETTER), match="exact")]
        )
        assert [(s.text, s.exact) for s in sentences] == [(NINETY, True)]


def _sentence(text: str, *, exact: bool = False, start: int = 0) -> Sentence:
    return Sentence(1, start, start + len(text), text, 0, exact)


class TestSelectSentences:
    def test_below_threshold_dropped_and_order_kept(self):
        candidates = [_sentence("a"), _sentence("b", start=5), _sentence("c", start=9)]
        with patch("stratakb.rag.reranker.score_pairs", return_value=[0.6, 0.2, 0.9]):
            chosen = select_sentences("q", candidates, 0.5, "model")

        assert [(s.text, s.score) for s in chosen] == [("a", 0.6), ("c", 0.9)]

    def test_score_on_the_threshold_stays(self):
        with patch("stratakb.rag.reranker.score_pairs", return_value=[0.5]):
            assert select_sentences("q", [_sentence("a")], 0.5, "model")

    def test_capped_at_the_best(self):
        candidates = [_sentence(f"s{i}", start=i * 5) for i in range(MAX_SENTENCES + 2)]
        scores = [0.6, 0.99, 0.7, 0.71, 0.72, 0.73, 0.74]
        with patch("stratakb.rag.reranker.score_pairs", return_value=scores):
            chosen = select_sentences("q", candidates, 0.5, "model")

        assert [s.text for s in chosen] == ["s1", "s3", "s4", "s5", "s6"]

    def test_exact_sentences_need_no_model(self):
        with patch("stratakb.rag.reranker.score_pairs") as score:
            chosen = select_sentences("q", [_sentence("a", exact=True)], 0.5, "model")

        score.assert_not_called()
        assert [(s.text, s.score) for s in chosen] == [("a", 1.0)]

    def test_budget_keeps_the_best_in_the_order_given(self):
        low, high = _sentence("x" * 400, start=0), _sentence("y" * 400, start=500)
        sentences = [replace(low, score=0.6), replace(high, score=0.9)]
        chunks, documents = [_chunk(0, 1000)], _BUDGET_DOCUMENTS

        assert fit_to_answer_budget(sentences, chunks, documents, 10_000) == sentences
        # The best is kept even when alone it passes the budget
        assert fit_to_answer_budget(sentences, chunks, documents, 1) == sentences[1:]

    def test_budget_keeps_the_start_of_an_exact_clause(self):
        sentences = [_sentence(f"{i}" * 400, exact=True, start=i * 500) for i in range(4)]

        kept = fit_to_answer_budget(sentences, [_chunk(0, 2000)], _BUDGET_DOCUMENTS, 300)

        assert kept == sentences[: len(kept)]
        assert 0 < len(kept) < len(sentences)

    def test_exact_sentences_stay_beside_scored_ones(self):
        candidates = [_sentence("a", exact=True), _sentence("b", start=5)]
        with patch("stratakb.rag.reranker.score_pairs", return_value=[0.1]) as score:
            chosen = select_sentences("q", candidates, 0.5, "model")

        assert score.call_args.args[1] == ["b"]
        assert [s.text for s in chosen] == ["a"]


_BUDGET_DOCUMENTS = {
    1: {"relative_path": "a.md", "page_map": None, "text": "x" * 3000, "text_origin": "text"}
}


class TestAnswerItem:
    def _item(self, origin: str) -> dict:
        chunk = {
            "chunk_id": "c", "content": "x", "layer": "l", "doc_type": "pdf", "metadata": {},
            "document_id": 1, "char_start": 0, "char_end": 9,
        }
        document = {
            "relative_path": "a.pdf", "page_map": None, "text": "One here.", "text_origin": origin,
        }
        return answer_item(Sentence(1, 0, 9, "One here.", 0, False, 0.7), [chunk], {1: document})

    def test_fields(self):
        item = self._item("text")
        assert item["text"] == "One here."
        assert item["ref"] == "d1:0-9"
        assert item["passage"] == "S1"
        assert item["citation"]["file"] == "a.pdf"
        assert item["text_origin"] == "text"
        assert item["score"] == 0.7
        assert "warning" not in item

    def test_ocr_text_warns(self):
        item = self._item("ocr")
        assert item["text_origin"] == "ocr"
        assert item["warning"] == OCR_WARNING


class TestAskResultExtractive:
    def _answer(self, **fields) -> AskResult:
        from stratakb.rag.confidence import Confidence

        defaults = {
            "question": "q",
            "queries_used": ["q"],
            "confidence": Confidence("high", 0.9, 3, "similarity"),
            "chunks": [],
            "passages": [{"id": "S1"}],
            "status": "found",
            "extraction": self._extraction(),
        }
        return AskResult(**{**defaults, **fields})

    def _extraction(self, note: str | None = None) -> ExtractiveAnswer:
        return ExtractiveAnswer([{"text": "t", "score": 0.9}], "rcpt_1", "found", note)

    def test_mode_follows_the_extraction(self):
        assert self._answer().mode == "extractive"
        assert self._answer(extraction=None).mode == "grounded"

    def test_not_found_is_still_extractive(self):
        result = self._answer(
            status="not_found",
            extraction=ExtractiveAnswer([], None, "not_found", None),
        )

        assert result.mode == "extractive"
        assert result.concise()["answer"] == []
        assert "passages" not in result.concise()

    def test_concise_has_answer_instead_of_passages_and_no_score(self):
        response = self._answer().concise()

        assert list(response) == ["status", "mode", "answer", "receipt_id"]
        assert response["answer"] == [{"text": "t"}]

    def test_detailed_has_scores_and_passages(self):
        response = self._answer(extraction=self._extraction(note="why")).detailed()

        assert response["answer"] == [{"text": "t", "score": 0.9}]
        assert response["receipt_id"] == "rcpt_1"
        assert response["note"] == "why"
        assert response["passages"] == [{"id": "S1"}]

    def test_grounded_has_no_extractive_field(self):
        result = self._answer(extraction=None)

        assert list(result.concise()) == ["status", "mode", "passages"]
        assert not {"answer", "receipt_id", "note"} & set(result.detailed())

    def test_partly_found_step_speaks_of_quotes(self):
        assert "quotes" in self._answer(status="partly_found").concise()["next_step"]
        grounded = self._answer(status="partly_found", extraction=None)
        assert "quotes" not in grounded.concise()["next_step"]


# ---------------------------------------------------------------------------
# strata_ask (database)
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


@pytest.fixture
async def make_layer(settings):
    """Make layers in a given mode, each holding DOC_TEXT; yields (layer, document_id)."""
    from stratakb.core.database import get_pool
    from stratakb.knowledge.documents import store_document_version
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    db_url = settings.database_url
    names = []

    async def make(
        answer_mode: str, *, text_origin: str = "text", text: str = DOC_TEXT
    ) -> tuple[str, int]:
        name = f"test_extract_{uuid4().hex[:10]}"
        names.append(name)
        await create_layer(db_url, name=name, display_name="Extract", answer_mode=answer_mode)
        pool = await get_pool(db_url)
        layer_id = await pool.fetchval(
            "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", name
        )
        async with pool.acquire() as conn, conn.transaction():
            document_id, _ = await store_document_version(
                conn, layer_id=layer_id, file_id=None, source_root="/docs",
                relative_path="contracts/acme-msa.md", text=text, page_map=None,
                text_origin=text_origin,
            )
        return name, document_id

    yield make
    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


def _hit(layer, document_id=None, span=None, *, similarity=0.9, name="hit") -> SearchResult:
    start, end = span if span else (0, len(DOC_TEXT)) if document_id else (None, None)
    return SearchResult(
        chunk_id=f"{name}-{uuid4().hex[:6]}",
        content="content",
        score=1.0,
        layer_name=layer,
        source_file=f"{name}.md",
        source_section="section",
        doc_type="markdown",
        metadata={"section_path": ["Master agreement", "7 Termination"]},
        similarity=similarity,
        document_id=document_id,
        char_start=start,
        char_end=end,
    )


def _relevant(*words: str, score: float = 0.9):
    """A cross-encoder that finds the sentences holding any of words relevant."""

    def score_pairs(_question, texts, _model):
        return [score if any(w in t for w in words) else 0.05 for t in texts]

    return score_pairs


async def _ask(server, results, score_pairs=None, **arguments) -> dict:
    scorer = score_pairs or _relevant("ninety")
    with (
        patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=results)),
        patch("stratakb.rag.reranker.score_pairs", side_effect=scorer),
    ):
        return _payload(
            await server.call_tool("strata_ask", {"question": "how to end it?", **arguments})
        )


class TestAskExtractive:
    async def test_answer_is_verbatim(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        response = await _ask(
            server, [_hit(layer, document_id)], _relevant("ninety", "registered"), layers=[layer]
        )

        assert response["status"] == "found"
        assert response["mode"] == "extractive"
        assert "passages" not in response
        assert [item["text"] for item in response["answer"]] == [NINETY, LETTER]
        for item in response["answer"]:
            document, span = item["ref"].split(":")
            start, end = map(int, span.split("-"))
            assert document == f"d{document_id}"
            assert item["text"] == DOC_TEXT[start:end]
            assert item["citation"]["file"] == "contracts/acme-msa.md"
            assert item["citation"]["layer"] == layer
            assert item["passage"] == "S1"
            assert item["text_origin"] == "text"
            assert "warning" not in item
            assert "score" not in item
        assert response["answer"][0]["citation"]["lines"] == "7-8"
        assert response["receipt_id"].startswith("rcpt_")

    async def test_receipt_opens_on_the_quoted_sentences(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        response = await _ask(server, [_hit(layer, document_id)], layers=[layer])
        opened = _payload(
            await server.call_tool("strata_get_receipt", {"receipt_id": response["receipt_id"]})
        )

        receipt = opened["receipt"]
        assert receipt["kind"] == "extractive"
        assert receipt["mode"] == "extractive"
        assert receipt["layers"] == [layer]
        assert [q["quote"] for q in receipt["quotes"]] == [NINETY]
        assert receipt["quotes"][0]["document_id"] == document_id
        assert receipt["quotes"][0]["clause_path"] == "Master agreement > 7 Termination"

    async def test_ocr_sentence_warns(self, server, make_layer):
        layer, document_id = await make_layer("extractive", text_origin="ocr")

        response = await _ask(server, [_hit(layer, document_id)], layers=[layer])

        assert response["answer"][0]["text_origin"] == "ocr"
        assert response["answer"][0]["warning"] == OCR_WARNING

    async def test_grounded_request_on_extractive_layer_stays_extractive(
        self, server, make_layer
    ):
        layer, document_id = await make_layer("extractive")

        response = await _ask(server, [_hit(layer, document_id)], layers=[layer], mode="grounded")

        assert response["mode"] == "extractive"
        assert response["note"] == GROUNDED_REFUSED_NOTE
        assert response["answer"][0]["text"] == NINETY

    async def test_extractive_request_on_grounded_layer(self, server, make_layer):
        layer, document_id = await make_layer("grounded")

        response = await _ask(
            server, [_hit(layer, document_id)], layers=[layer], mode="extractive"
        )

        assert response["mode"] == "extractive"
        assert "note" not in response
        assert response["answer"][0]["text"] == NINETY

    async def test_grounded_layer_unchanged(self, server, make_layer):
        layer, document_id = await make_layer("grounded")

        with patch("stratakb.rag.tools.issue_receipt", AsyncMock()) as issue:
            response = await _ask(server, [_hit(layer, document_id)], layers=[layer])

        issue.assert_not_called()
        assert response["mode"] == "grounded"
        assert set(response) == {"status", "mode", "passages"}

    async def test_mixed_layers_extractive(self, server, make_layer):
        grounded, _ = await make_layer("grounded")
        extractive, document_id = await make_layer("extractive")

        response = await _ask(
            server, [_hit(extractive, document_id)], layers=[grounded, extractive]
        )

        assert response["mode"] == "extractive"
        assert "note" not in response

    async def test_routing_fallback_to_all_layers_finds_the_extractive_one(
        self, server, make_layer
    ):
        grounded, _ = await make_layer("grounded")
        extractive, document_id = await make_layer("extractive")

        with patch(
            "stratakb.rag.tools._detect_layers", AsyncMock(return_value=[grounded])
        ):
            # A low-confidence result sends the question to every layer
            response = await _ask(
                server, [_hit(extractive, document_id, similarity=0.1)], top_k=1
            )

        assert response["mode"] == "extractive"

    async def test_layer_made_extractive_is_read_at_once(self, server, settings, make_layer):
        from stratakb.knowledge.layers import set_layer_answer_mode

        layer, document_id = await make_layer("grounded")
        hits = [_hit(layer, document_id)]
        assert (await _ask(server, hits, layers=[layer]))["mode"] == "grounded"

        await set_layer_answer_mode(settings.database_url, layer, "extractive")

        assert (await _ask(server, hits, layers=[layer]))["mode"] == "extractive"

    async def test_below_threshold_not_found(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        response = await _ask(
            server,
            [_hit(layer, document_id)],
            _relevant("ninety", score=RERANK_THRESHOLDS.medium_score - 0.01),
            layers=[layer],
        )

        assert response["status"] == "not_found"
        assert response["answer"] == []
        assert "receipt_id" not in response
        assert "passages" not in response
        assert response["next_step"]

    async def test_long_block_is_not_found_and_issues_no_receipt(self, server, make_layer):
        text = f"{LONG_BLOCK}\n\nA short line on something else.\n"
        layer, document_id = await make_layer("extractive", text=text)
        hit_start = text.index(BLOCK_HIT)
        span = (hit_start, hit_start + len(BLOCK_HIT) * 21)

        with patch("stratakb.rag.tools.issue_receipt", AsyncMock()) as issue:
            response = await _ask(server, [_hit(layer, document_id, span)], layers=[layer])

        assert response["status"] == "not_found"
        assert response["answer"] == []
        issue.assert_not_called()

    async def test_named_clause_in_a_long_block_is_not_found(self, server, make_layer):
        text = f"{LONG_BLOCK}\n\nA short line on something else.\n"
        layer, document_id = await make_layer("extractive", text=text)
        hit_start = text.index(BLOCK_HIT)
        chunk = ask_chunk(
            chunk_id="clause", content=text[hit_start:hit_start + 200], score=1.0,
            similarity=1.0, layer=layer, source_file="contracts/acme-msa.md", section=None,
            doc_type="markdown", metadata={}, document_id=document_id, char_start=hit_start,
            char_end=hit_start + 200, clause_path=None,
        )
        chunk["match"] = "exact"

        with patch("stratakb.rag.tools._exact_clauses", AsyncMock(return_value=[chunk])):
            response = await _ask(server, [], layers=[layer])

        assert response["status"] == "not_found"
        assert response["answer"] == []

    async def test_answer_stays_within_the_budget(self, server, settings, make_layer):
        sentences = [f"Clause {i} says ninety {'word ' * 370}here." for i in range(5)]
        text = " ".join(sentences) + " Tail.\n"
        layer, document_id = await make_layer("extractive", text=text)
        scores = {s: 0.9 - i / 100 for i, s in enumerate(sentences)}

        def score_pairs(_question, texts, _model):
            return [scores.get(t, 0.05) for t in texts]

        response = await _ask(
            server,
            [_hit(layer, document_id, (0, len(text) - 7))],
            score_pairs,
            layers=[layer],
        )

        quoted = [a["text"] for a in response["answer"]]
        assert quoted == sentences[: len(quoted)]
        assert 0 < len(quoted) < len(sentences)
        assert len(json.dumps(response["answer"])) / 4 <= settings.answer_token_budget

    async def test_status_follows_the_sentence_score(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        response = await _ask(
            server, [_hit(layer, document_id)], _relevant("ninety", score=0.6), layers=[layer]
        )

        assert response["status"] == "partly_found"
        assert "quotes" in response["next_step"]

    async def test_no_results_not_found(self, server, make_layer):
        layer, _ = await make_layer("extractive")

        response = await _ask(server, [], layers=[layer])

        assert response["status"] == "not_found"
        assert response["answer"] == []
        assert "note" not in response

    async def test_legacy_chunks_ask_for_a_reingest(self, server, make_layer):
        layer, _ = await make_layer("extractive")

        response = await _ask(server, [_hit(layer)], layers=[layer])

        assert response["status"] == "not_found"
        assert response["note"] == LEGACY_ONLY_NOTE

    async def test_refused_grounded_and_legacy_notes_join(self, server, make_layer):
        layer, _ = await make_layer("extractive")

        response = await _ask(server, [_hit(layer)], layers=[layer], mode="grounded")

        assert response["note"] == f"{GROUNDED_REFUSED_NOTE} {LEGACY_ONLY_NOTE}"

    async def test_audit_records_mode(self, server, make_layer, audit):
        extractive, document_id = await make_layer("extractive")
        grounded, _ = await make_layer("grounded")

        await _ask(server, [_hit(extractive, document_id)], layers=[extractive])
        assert audit.await_args.kwargs["parameters"]["mode"] == "extractive"

        await _ask(server, [_hit(grounded, document_id)], layers=[grounded])
        assert audit.await_args.kwargs["parameters"]["mode"] == "grounded"

    async def test_missing_model_is_an_error_not_an_empty_answer(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        def no_model(*_):
            raise OSError("model not found")

        response = await _ask(server, [_hit(layer, document_id)], no_model, layers=[layer])

        assert response["status"] == "error"
        assert "Extractive mode needs the reranker model" in response["error"]

    async def test_invalid_mode_refused(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        with pytest.raises(ToolError):
            await _ask(server, [_hit(layer, document_id)], layers=[layer], mode="free")

    async def test_named_clause_quoted_without_the_model(self, server, settings, tmp_path):
        from stratakb.knowledge.layers import create_layer, delete_layer

        db_url = settings.database_url
        layer = f"test_extract_{uuid4().hex[:10]}"
        await create_layer(
            db_url, name=layer, display_name="Extract", answer_mode="extractive",
            chunk_config={"chunk_size": 200, "chunk_overlap": 40},
        )
        try:
            shutil.copy(FIXTURES / "contract.md", tmp_path / "contract.md")
            assert (await ingest(db_url, layer, tmp_path)).docs_failed == 0

            with (
                patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[])),
                patch("stratakb.rag.reranker.score_pairs") as score,
            ):
                response = _payload(
                    await server.call_tool(
                        "strata_ask", {"question": "what does clause 2.2 say", "layers": [layer]}
                    )
                )

            score.assert_not_called()
            assert response["status"] == "found"
            assert response["receipt_id"].startswith("rcpt_")
            assert response["answer"]
            for item in response["answer"]:
                assert item["citation"]["clause_path"].endswith("2.2 Termination for cause")
        finally:
            await delete_layer(db_url, name=layer)


class TestSettingTheMode:
    async def test_preset_layer_reports_extractive(self, server, settings, tmp_path):
        from stratakb.knowledge.layers import delete_layer
        from stratakb.knowledge.presets import create_preset_layer, load_preset

        db_url = settings.database_url
        name = f"test_extract_{uuid4().hex[:10]}"
        path = tmp_path / "preset.yaml"
        path.write_text(
            f"name: {name}\ndisplay_name: Extract\nanswer_mode: extractive\n"
            "sources:\n  - name: docs\n    path: docs\n",
            encoding="utf-8",
        )
        try:
            created = await create_preset_layer(db_url, load_preset(path))
            assert created["layer"]["answer_mode"] == "extractive"

            listed = await call_tool(server, "strata_list_knowledge_layers")

            layer = next(row for row in listed["layers"] if row["name"] == name)
            assert layer["answer_mode"] == "extractive"
        finally:
            await delete_layer(db_url, name=name)

    async def test_create_tool_accepts_the_mode(self, server, settings):
        from stratakb.knowledge.layers import delete_layer, list_layers

        db_url = settings.database_url
        name = f"test_extract_{uuid4().hex[:10]}"
        try:
            with as_admin():
                created = await call_tool(
                    server, "strata_create_knowledge_layer",
                    name=name, display_name="Extract", answer_mode="extractive",
                )
                refused = await call_tool(
                    server, "strata_create_knowledge_layer",
                    name=f"{name}_bad", display_name="Bad", answer_mode="free",
                )

            assert created["status"] == "created"
            assert refused["status"] == "error"
            assert "answer_mode" in refused["error"]
            rows = await list_layers(db_url)
            assert next(r for r in rows if r["name"] == name)["answer_mode"] == "extractive"
        finally:
            await delete_layer(db_url, name=name)
