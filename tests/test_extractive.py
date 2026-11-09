"""Extractive answers: the document's own sentences, chosen without a language model.

The splitting, mode and selection tests here are pure. test_extractive_ask.py
calls strata_ask itself.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import get_args
from unittest.mock import patch

import pydantic_core
import pytest

from stratakb.knowledge.layers import ANSWER_MODES
from stratakb.rag.extractive import (
    GROUNDED_REFUSED_NOTE,
    MAX_SENTENCE_CHARS,
    MAX_SENTENCES,
    OCR_WARNING,
    Sentence,
    answer_item,
    candidate_sentences,
    clause_gap_note,
    effective_mode,
    fit_to_answer_budget,
    select_sentences,
    split_sentences,
)
from stratakb.rag.passages import (
    AnswerMode,
    AskResult,
    ExtractiveAnswer,
    estimate_tokens,
    listed_cost,
)
from tests.helpers import BLOCK_HIT, DOC_TEXT, LETTER, LONG_BLOCK, NINETY

INVOICES = "Invoices are due within 30 days."
INTEREST = "Late payment carries interest."


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


class TestClauseGapNote:
    TEXT = f"{INVOICES} {'word ' * 450}end. {INTEREST}\n"
    DOCUMENTS = {1: {"text": TEXT, "relative_path": "terms.md"}}
    ENDS = {1: len(TEXT) + 1000}

    def _note(self, chunk, kept=None):
        kept = _candidates([chunk], self.TEXT) if kept is None else kept
        return clause_gap_note([chunk], self.DOCUMENTS, self.ENDS, kept)

    def test_whole_clause_needs_no_note(self):
        short = "One here. Two there.\n"
        chunk = _chunk(0, len(short), match="exact", clause_path="7 Pay")
        documents = {1: {"text": short, "relative_path": "terms.md"}}
        kept = candidate_sentences([chunk], documents, {1: len(short) + 1000})

        assert clause_gap_note([chunk], documents, {1: len(short) + 1000}, kept) is None

    def test_sentence_too_long_to_quote_is_named(self):
        note = self._note(_chunk(0, len(self.TEXT), match="exact", clause_path="7 Pay"))

        assert note.startswith("7 Pay is only partly quoted: a sentence too long to quote")
        assert "does not fit" not in note
        assert note.endswith("Read the whole clause in terms.md.")

    def test_sentence_past_the_budget_is_named(self):
        short = "One here. Two there.\n"
        documents = {1: {"text": short, "relative_path": "terms.md"}}
        whole = _chunk(0, len(short), match="exact")
        kept = candidate_sentences([whole], documents, {1: len(short) + 1000})[:1]

        note = clause_gap_note([whole], documents, {1: len(short) + 1000}, kept)

        assert note.startswith("The clause the question names is only partly quoted: the rest")
        assert "too long" not in note

    def test_sentence_cut_by_the_read_is_named(self):
        text = "One here. Two is cut"
        documents = {1: {"text": text, "relative_path": "terms.md"}}
        chunk = _chunk(0, len(text), match="exact")
        kept = candidate_sentences([chunk], documents, {1: len(text)})

        assert [s.text for s in kept] == ["One here."]
        note = clause_gap_note([chunk], documents, {1: len(text)}, kept)
        assert "a sentence too long to quote" in note

    def test_clause_cut_by_the_lookup_is_named(self):
        chunk = _chunk(0, len(INVOICES), match="exact", cut=True)

        assert "the rest does not fit" in self._note(chunk)

    def test_scored_chunk_needs_no_note(self):
        assert self._note(_chunk(0, len(self.TEXT)), kept=[]) is None


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

    def test_budget_is_used_to_within_one_sentence(self):
        def tokens(items: list[dict]) -> int:
            return estimate_tokens(pydantic_core.to_json({"answer": items}, indent=2).decode())

        sentences = [
            replace(_sentence(f"{i}" * 300, start=i * 300), score=1 - i / 100)
            for i in range(10)
        ]
        chunks, documents, budget = [_chunk(0, 3000)], _BUDGET_DOCUMENTS, 1000

        kept = fit_to_answer_budget(sentences, chunks, documents, budget)
        items = [answer_item(s, chunks, documents) for s in kept]
        added = tokens(items) - tokens([])
        following = answer_item(sentences[len(kept)], chunks, documents)

        assert 1 < len(kept) < len(sentences)
        assert budget - listed_cost(following) < added <= budget

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
