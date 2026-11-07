"""Extractive answers: the document's own sentences, chosen without a language model.

The stored text is split into sentences and clauses, the cross-encoder scores
the ones the returned passages touch against the question, and the best are
returned word for word: each is text[start:end] of a stored document. A
passage may start or stop inside a sentence; the quote is always the whole
sentence, read from the document, because half a contract sentence can say the
opposite of the whole one.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from stratakb.rag.passages import (
    RESPONSE_SLACK_TOKENS,
    cite,
    estimate_tokens,
    passage_id,
    passage_ref,
)

# Characters read past a document's last cited span, to finish its last sentence
SENTENCE_REACH = 1000

# The most sentences chosen by score; an exact clause's sentences come on top
MAX_SENTENCES = 5

# The longest span quoted: a third of the default answer budget, and about what
# the cross-encoder reads before it truncates. Lines without end marks (a price
# list, a log, YAML) can make one "sentence" of a whole block; it is not cut,
# because half a sentence can say the opposite, so it is not quoted at all
MAX_SENTENCE_CHARS = 2000

# The note when a mode asked for could not loosen a layer's
GROUNDED_REFUSED_NOTE = (
    "A passage returned comes from an extractive layer, so this answer is extractive; "
    "grounded cannot be asked for on that layer."
)
# The note when only chunks stored before documents were kept matched
LEGACY_ONLY_NOTE = "Re-ingest this layer to enable extractive answers."
OCR_WARNING = "from scanned text, check the original page"

_LINE = re.compile(r"[^\n]*(?:\n|$)")
_HEADING = re.compile(r"[ \t]*#{1,6}[ \t]")
# An underline or a horizontal rule ends a block, as a blank line does
_RULE = re.compile(r"[ \t]*[=\-~^*_]{3,}\s*$")
# A list item, a table row, a quote line or a numbered clause starts a new
# block, so a line that merely wraps ("30 days' notice") stays with the last one
_BLOCK_START = re.compile(
    r"[ \t]*(?:[-*•>][ \t]|\||\d+[.)][ \t]|\d+(?:\.\d+)+[.)]?[ \t]|\([a-zA-Z0-9]{1,4}\)[ \t])"
)
# End punctuation, any closing quote or bracket, then whitespace or the end. It
# starts only at the start of a run and never backtracks, so a stored document
# cannot make it quadratic (the re module holds the GIL: that would stall the server)
_TERMINAL = re.compile(r"(?<![.!?;])[.!?;]++[\"'”’)\]]*+(?=\s|$)")
_TOKEN_BEFORE_DOT = re.compile(r"[\w.]*\Z")
# Characters looked back from a dot for its token, longer than any token that
# matters and short enough that each dot costs the same however long the sentence
_TOKEN_REACH = 64
_ABBREVIATIONS = frozenset(
    {
        "approx", "art", "arts", "ca", "cf", "ch", "cl", "co", "corp", "dr", "excl",
        "fig", "figs", "inc", "incl", "ltd", "mr", "mrs", "ms", "no", "nos", "p",
        "para", "paras", "pp", "prof", "ref", "resp", "sec", "secs", "vol", "vs",
    }
)


def _blocks(text: str) -> list[tuple[int, int]]:
    """The spans of text that can hold a sentence, without blank lines and headings."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for match in _LINE.finditer(text):
        line = match.group()
        if not line:
            continue
        skipped = not line.strip() or _HEADING.match(line) or _RULE.match(line)
        if start is not None and (skipped or _BLOCK_START.match(line)):
            blocks.append((start, match.start()))
            start = None
        if start is None and not skipped:
            start = match.start()
    if start is not None:
        blocks.append((start, len(text)))
    return blocks


def _is_sentence_end(text: str, sentence_start: int, match: re.Match[str]) -> bool:
    """False for the dot of an abbreviation, an initial or a clause number."""
    if match.group()[0] != ".":
        return True
    following = text[match.end() : match.end() + 20].lstrip()
    if following[:1].islower():
        return False
    look_from = max(sentence_start, match.start() - _TOKEN_REACH)
    token = _TOKEN_BEFORE_DOT.search(text, look_from, match.start()).group()
    last = token.rsplit(".", 1)[-1].lower()
    if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
        return False
    # "7. Payment": the number of a clause, with nothing before it
    is_number = token[:1].isdigit() and token.replace(".", "").isdigit()
    return not (is_number and not text[sentence_start : match.start() - len(token)].strip())


def split_sentences(text: str, start: int = 0) -> list[tuple[int, int]]:
    """The (start, end) of each sentence or clause in text, offset by start.

    Sentences end at . ! ? ; and blocks at blank lines, headings and list items,
    but a line that only wraps does not end one. Abbreviations (e.g., No., Art.),
    initials and clause numbers (7.2, "7. Payment") stay inside. Surrounding
    whitespace is left out, and so are headings and anything without a letter
    or digit.
    """
    spans: list[tuple[int, int]] = []

    def keep(begin: int, end: int) -> None:
        piece = text[begin:end]
        stripped = piece.strip()
        if any(c.isalnum() for c in stripped):
            first = begin + len(piece) - len(piece.lstrip())
            spans.append((start + first, start + first + len(stripped)))

    for block_start, block_end in _blocks(text):
        position = block_start
        for match in _TERMINAL.finditer(text, block_start, block_end):
            if _is_sentence_end(text, position, match):
                keep(position, match.end())
                position = match.end()
        keep(position, block_end)
    return spans


def _touched_spans(
    spans: Sequence[tuple[int, int]], chunk_start: int, chunk_end: int
) -> list[tuple[int, int]]:
    """The sentence spans that overlap a chunk: spans are sorted and do not overlap."""
    # The first sentence ending after the chunk's start
    first = bisect_right(spans, chunk_start, key=lambda span: span[1])
    touched = []
    for span in spans[first:]:
        if span[0] >= chunk_end:
            break
        touched.append(span)
    return touched


def _cut_by_read(
    span: tuple[int, int], spans: Sequence[tuple[int, int]], text: str, read_end: int
) -> bool:
    """True for the last sentence when the read stopped at read_end: it may be cut."""
    return span == spans[-1] and len(text) >= read_end


def effective_mode(requested: str | None, layer_modes: Sequence[str]) -> tuple[str, str | None]:
    """The mode an answer runs in and a note when the request could not loosen it.

    The strictest wins: an extractive mode among layer_modes, or a request for
    extractive, makes the answer extractive. strata_ask gives the modes of the
    layers its passages come from.
    """
    if requested == "extractive" or "extractive" in layer_modes:
        return "extractive", GROUNDED_REFUSED_NOTE if requested == "grounded" else None
    return "grounded", None


@dataclass(frozen=True)
class Sentence:
    """A sentence of a stored document, and the passage (by index) it was found through.

    An exact sentence belongs to a clause the question names and needs no score.
    """

    document_id: int
    start: int
    end: int
    text: str
    chunk_index: int
    exact: bool
    score: float = 1.0


def candidate_sentences(
    chunks: Sequence[dict[str, Any]],
    documents: dict[int, dict[str, Any]],
    read_ends: dict[int, int],
) -> list[Sentence]:
    """The whole sentences the chunks touch, once each, in chunk then document order.

    documents hold each document's text up to read_ends. Sentences come from
    the document text, not from each chunk, so an overlap between chunks gives
    no duplicates and a chunk's cut gives no fragments. A chunk without a stored
    document or offsets gives none. When the read reached read_ends it may have
    cut the document's last sentence, wherever the cut landed, so that one is
    left out. So is a span longer than MAX_SENTENCE_CHARS.
    """
    spans_by_document: dict[int, list[tuple[int, int]]] = {}
    seen: set[tuple[int, int, int]] = set()
    sentences: list[Sentence] = []
    for index, chunk in enumerate(chunks):
        document_id = chunk.get("document_id")
        document = documents.get(document_id)
        chunk_start, chunk_end = chunk.get("char_start"), chunk.get("char_end")
        if document is None or chunk_start is None or chunk_end is None:
            continue
        text = document["text"]
        if document_id not in spans_by_document:
            spans_by_document[document_id] = split_sentences(text)
        spans = spans_by_document[document_id]
        for start, end in _touched_spans(spans, chunk_start, chunk_end):
            if _cut_by_read((start, end), spans, text, read_ends[document_id]):
                continue
            if end - start > MAX_SENTENCE_CHARS:
                continue
            if (document_id, start, end) in seen:
                continue
            seen.add((document_id, start, end))
            sentences.append(
                Sentence(
                    document_id, start, end, text[start:end], index,
                    chunk.get("match") == "exact",
                )
            )
    return sentences


def select_sentences(
    question: str,
    candidates: Sequence[Sentence],
    threshold: float,
    model_name: str,
    max_sentences: int = MAX_SENTENCES,
) -> list[Sentence]:
    """The sentences that answer question, in the order of candidates.

    Exact sentences all stay. The others are scored with the reranker's
    cross-encoder (relevance probability, 0-1); the best max_sentences of those
    at or above threshold stay.
    """
    scored = [c for c in candidates if not c.exact]
    best: dict[Sentence, float] = {}
    if scored:
        # Imported here: the reranker module loads the model machinery
        from stratakb.rag.reranker import score_pairs

        scores = score_pairs(question, [c.text for c in scored], model_name)
        above = [(c, s) for c, s in zip(scored, scores, strict=True) if s >= threshold]
        above.sort(key=lambda pair: pair[1], reverse=True)
        best = dict(above[:max_sentences])
    return [
        replace(c, score=round(best[c], 4)) if c in best else c
        for c in candidates
        if c.exact or c in best
    ]


def answer_item(
    sentence: Sentence,
    chunks: Sequence[dict[str, Any]],
    documents: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """One sentence of the answer: its text, ref, passage id, citation and text origin."""
    chunk = {
        **chunks[sentence.chunk_index],
        "char_start": sentence.start,
        "char_end": sentence.end,
    }
    document = documents[sentence.document_id]
    item: dict[str, Any] = {
        "text": sentence.text,
        "ref": passage_ref(chunk),
        "passage": passage_id(sentence.chunk_index + 1),
        "citation": cite(chunk, document),
        "text_origin": document["text_origin"],
        "score": sentence.score,
    }
    if document["text_origin"] == "ocr":
        item["warning"] = OCR_WARNING
    return item


def fit_to_answer_budget(
    sentences: Sequence[Sentence],
    chunks: Sequence[dict[str, Any]],
    documents: dict[int, dict[str, Any]],
    budget: int,
) -> list[Sentence]:
    """The best sentences whose answer items fit the token budget, in the order given.

    Exact sentences count as best, in document order, so a clause over the budget
    keeps its start and clause_gap_note says so. As with passages, each costs its
    text, ref and citation, taking stops before the sentence that would pass the
    budget, and the best is always kept (MAX_SENTENCE_CHARS bounds it).
    """
    kept: set[Sentence] = set()
    used = 0
    for sentence in sorted(sentences, key=lambda s: s.score, reverse=True):
        item = answer_item(sentence, chunks, documents)
        cost = estimate_tokens(json.dumps(item, indent=2, ensure_ascii=False))
        cost += RESPONSE_SLACK_TOKENS
        if kept and used + cost > budget:
            break
        kept.add(sentence)
        used += cost
    return [s for s in sentences if s in kept]


def clause_gap_note(
    chunks: Sequence[dict[str, Any]],
    documents: dict[int, dict[str, Any]],
    read_ends: dict[int, int],
    kept: Sequence[Sentence],
) -> str | None:
    """A note naming each clause the question names that kept is missing part of, or None.

    Half a clause can say the opposite of the whole one, so an answer must say
    when it leaves part of one out: a sentence candidate_sentences skips as too
    long or cut by the read, or text past the token budget, whether the clause
    lookup cut it or fit_to_answer_budget did.
    """
    kept_spans = {(s.document_id, s.start, s.end) for s in kept}
    notes = []
    for chunk in chunks:
        document_id = chunk.get("document_id")
        document = documents.get(document_id)
        if chunk.get("match") != "exact" or document is None:
            continue
        text = document["text"]
        # Re-split per clause: only a named clause gets here, and there are few
        spans = split_sentences(text)
        too_long, over_budget = False, bool(chunk.get("cut"))
        for span in _touched_spans(spans, chunk["char_start"], chunk["char_end"]):
            if _cut_by_read(span, spans, text, read_ends[document_id]) or (
                span[1] - span[0] > MAX_SENTENCE_CHARS
            ):
                too_long = True
            elif (document_id, *span) not in kept_spans:
                over_budget = True
        if too_long or over_budget:
            reasons = [
                reason
                for reason, applies in (
                    ("a sentence too long to quote is left out", too_long),
                    ("the rest does not fit in the answer", over_budget),
                )
                if applies
            ]
            name = chunk.get("clause_path") or "The clause the question names"
            notes.append(
                f"{name} is only partly quoted: {' and '.join(reasons)}. "
                f"Read the whole clause in {document['relative_path']}."
            )
    return " ".join(notes) or None
