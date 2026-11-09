"""Exact lookup of clause and section references through the structure map.

A question such as "what does clause 7.2 say" or "show Annex II" names an
address, not a topic. find_references reads such references from the text
('clause 7.2', 'Article 12', '§ 4', 'Annex II', a bare '7.2' or '7(2)(b)',
and ids matched by extra patterns such as 'AT-44'), and resolve looks their
labels up in strata_vectors.document_sections, the structure map stored at
ingest, within the layers searched. Each resolved section comes back as an
exact passage: the stored document's own text over the section's span, with
the section's path. A reference that resolves nowhere is dropped, so a bare
number that is a version or an amount never blocks retrieval.

Lookup reads plain indexed columns (the label index), no full-text search.
At most MAX_REFERENCES references are looked up and MAX_EXACT_MATCHES
sections returned; cap_note tells the caller what those caps left out.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import structlog

from stratakb.knowledge.pattern_safety import pattern_problem
from stratakb.knowledge.structure import normalise_label
from stratakb.rag.layer_detect import pattern_text
from stratakb.rag.passages import ask_chunk, chars_for_tokens, passage_cost

logger = structlog.get_logger()

# The most references read from one question, and sections returned for them
MAX_REFERENCES = 10
MAX_EXACT_MATCHES = 10
# Sections read per lookup before nested ones are dropped: a short label such
# as '1' names a section in nearly every document
_MAX_CANDIDATE_SECTIONS = 5 * MAX_EXACT_MATCHES

# A trimmed exact passage keeps at least this many characters
_MIN_TRIMMED_CHARS = 200

# A clause number: '7', '7.2', '4.3.2', '7(2)(b)'
_NUMBER = r"\d{1,3}(?:\.\d{1,3}){0,3}(?:\(\w{1,3}\))*"

# A keyword, then a number, a Roman numeral or a capital letter ('Annex A')
_KEYWORD_REF_RE = re.compile(
    r"(?:\b(?i:(clause|section|article|annex|appendix|schedule|chapter|part))\b|(§))"
    r"\s*(" + _NUMBER + r"|(?i:[IVXLC]{1,7})|[A-Z])(?![\w(])"
)

# A bare dotted or statute-style number: '7.2', '4.3.2', '7(2)(b)'; '2026'
# and '7' alone are not references
_BARE_REF_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){1,3}|\d{1,3}(?:\(\w{1,3}\))+)(?![\w(]|\.\d)"
)

# Keywords whose numbered sections may also be stored under the bare number
# ('Clause 7.2' as a '7.2 Notice' heading); the others ('Annex', 'Part')
# name their sections only with the keyword
_NUMBER_KEYWORDS = {"clause": "clause", "article": "article", "section": "section", "§": "section"}


class Reference(NamedTuple):
    """A reference in a question and the stored labels it can match."""

    text: str
    labels: tuple[str, ...]


class ClauseMatches(NamedTuple):
    """The chunks a lookup returns, and the matching sections it did not read."""

    chunks: list[dict[str, Any]]
    unread: int = 0
    # The candidate limit stopped the count, so more than unread may match
    unread_is_minimum: bool = False
    # How many references were looked up, and those past MAX_REFERENCES as written
    references: int = 0
    skipped_references: tuple[str, ...] = ()


def _keyword_labels(keyword: str, number: str) -> tuple[str, ...]:
    keyword = keyword.lower()
    number = normalise_label(number)
    name = _NUMBER_KEYWORDS.get(keyword, keyword)
    if keyword in _NUMBER_KEYWORDS and number[:1].isdigit():
        return (number, f"{name} {number}")
    return (f"{name} {number}",)


# A pattern's leading line-start anchor, after any global inline flags
_LEADING_ANCHOR_RE = re.compile(r"^((?:\(\?[aiLmsux]+\))*)\^")


def unanchored(pattern: str) -> str:
    """A layer's structure pattern made to match anywhere in a question.

    At ingest the pattern is matched at line starts ('^AT-\\d+'); in a
    question the id stands mid-sentence ('show AT-44'), so a leading '^'
    becomes a word start.
    """
    return _LEADING_ANCHOR_RE.sub(lambda m: m.group(1) + r"(?<!\w)", pattern, count=1)


_LEADING_FLAGS_RE = re.compile(r"(?:\(\?[aiLmsux]+\))*")
_LABEL_GROUP = "(?P<label>"


def _label_group(pattern: str) -> str | None:
    """The source of pattern's named group 'label', parentheses included."""
    start = pattern.find(_LABEL_GROUP)
    if start < 0:
        return None
    depth = 0
    in_class = False
    i = start
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
            # A ']' right after '[' or '[^' is a literal
            i += 1 if pattern[i + 1:i + 2] == "^" else 0
            i += 1 if pattern[i + 1:i + 2] == "]" else 0
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return pattern[start:i + 1]
        i += 1
    return None


def question_pattern(pattern: str) -> str:
    """A layer's structure pattern made to find its ids in a question.

    At ingest the pattern may need the text around the id
    ('^Identification:\\s*(?P<label>AT-\\d+)'); a question names the id
    alone, anywhere ('show AT-44'). So only the named group 'label' is
    looked for, as a whole word. A pattern without one is used whole, its
    leading '^' read as a word start.
    """
    group = _label_group(pattern)
    if group is None:
        return unanchored(pattern)
    flags = _LEADING_FLAGS_RE.match(pattern).group(0)
    return rf"{flags}(?<!\w){group}(?!\w)"


@functools.lru_cache(maxsize=256)
def _question_regex(pattern: str) -> re.Pattern[str] | None:
    """question_pattern(pattern) compiled, or None when the stored pattern
    must not run (it does not compile or it repeats a repeated part)."""
    # Check the stored pattern and the question form: an unsafe part outside
    # the label group would not show in the label group alone, and text that
    # only looks like the label group ('[(?P<label>(a+)+)]' is a character
    # class) is safe as stored but not once cut out
    problem = pattern_problem(pattern, line_starts=True)
    if problem is None:
        try:
            regex = question_pattern(pattern)
            re.compile(regex)
        except re.error:
            # The label group alone does not compile (it refers to another group)
            regex = unanchored(pattern)
        problem = pattern_problem(regex)
    if problem is not None:
        logger.warning("unsafe_structure_pattern", pattern=pattern, reason=problem)
        return None
    return re.compile(regex)


def find_references(
    question: str, extra_patterns: Iterable[str | re.Pattern[str]] = ()
) -> list[Reference]:
    """The first MAX_REFERENCES clause and section references in question, in
    order, without repeats.

    extra_patterns are a layer's structure patterns (a named group 'label'
    gives the label, as in structure.detect_sections); each match of
    question_pattern anywhere in the question is a reference. A stored
    pattern that does not compile or repeats a repeated part is skipped (it
    reached the layer some other way than its validated writes). Only the
    first MAX_PATTERN_QUESTION_CHARS characters of the question are read.
    """
    return _all_references(question, extra_patterns)[:MAX_REFERENCES]


def _all_references(
    question: str, extra_patterns: Iterable[str | re.Pattern[str]]
) -> list[Reference]:
    """find_references without the MAX_REFERENCES cap."""
    question = pattern_text(question)
    found: list[tuple[int, Reference]] = []
    covered: list[tuple[int, int]] = []

    for m in _KEYWORD_REF_RE.finditer(question):
        keyword = m.group(1) or m.group(2)
        found.append((m.start(), Reference(m.group(0), _keyword_labels(keyword, m.group(3)))))
        covered.append(m.span())

    for m in _BARE_REF_RE.finditer(question):
        if any(a <= m.start() < b for a, b in covered):
            continue
        found.append((m.start(), Reference(m.group(0), (normalise_label(m.group(1)),))))

    for pattern in extra_patterns:
        if isinstance(pattern, str):
            compiled = _question_regex(pattern)
        elif hasattr(pattern, "finditer"):
            compiled = pattern
        else:
            compiled = None
        if compiled is None:
            continue
        for m in compiled.finditer(question):
            if not m.group(0):
                continue
            label = m.groupdict().get("label") or m.group(0)
            found.append((m.start(), Reference(m.group(0), (normalise_label(label),))))

    references: list[Reference] = []
    seen: set[tuple[str, ...]] = set()
    for _, ref in sorted(found, key=lambda item: item[0]):
        if ref.labels not in seen:
            seen.add(ref.labels)
            references.append(ref)
    return references


async def resolve(
    database_url: str, references: Sequence[Reference], layers: list[str] | None
) -> ClauseMatches:
    """The stored sections the references name, as exact chunks.

    Only current documents count (those that still have chunks; an older
    version is kept only for receipts), and with layers only those layers.
    The chunks follow the references' order, then the file path; a section
    inside another one returned is left out, as its text is already there.
    At most _MAX_CANDIDATE_SECTIONS sections are read, and the text of at
    most MAX_EXACT_MATCHES; the sections past that are counted as unread.
    """
    # Each label ranks by the first reference that names it
    ranks: dict[str, int] = {}
    for rank, ref in enumerate(references):
        for label in ref.labels:
            ranks.setdefault(label, rank)
    if not ranks:
        return ClauseMatches([])
    from stratakb.core.database import get_pool

    pool = await get_pool(database_url)
    rows = await pool.fetch(
        """
        SELECT s.id, s.label, s.path, s.char_start, s.char_end,
               d.id AS document_id, d.relative_path,
               c.layer_name, c.source_file, c.doc_type
        FROM strata_vectors.document_sections s
        JOIN strata_vectors.documents d ON d.id = s.document_id
        JOIN LATERAL (
            SELECT ch.layer_name, ch.source_file, ch.doc_type
            FROM strata_vectors.chunks ch
            WHERE ch.document_id = d.id
            LIMIT 1
        ) c ON true
        WHERE s.label = ANY($1::text[])
          AND ($2::text[] IS NULL OR c.layer_name = ANY($2::text[]))
        ORDER BY ($3::int[])[array_position($1::text[], s.label)],
                 d.relative_path, d.id, s.char_start
        LIMIT $4
        """,
        list(ranks),
        layers or None,
        list(ranks.values()),
        _MAX_CANDIDATE_SECTIONS,
    )

    kept: list[Any] = []
    for i, row in enumerate(rows):
        inside = any(
            j != i
            and other["document_id"] == row["document_id"]
            and other["char_start"] <= row["char_start"]
            and row["char_end"] <= other["char_end"]
            # Of two equal spans the first one stays
            and (j < i or (other["char_start"], other["char_end"])
                 != (row["char_start"], row["char_end"]))
            for j, other in enumerate(rows)
        )
        if not inside:
            kept.append(row)
    unread = max(len(kept) - MAX_EXACT_MATCHES, 0)
    unread_is_minimum = len(rows) == _MAX_CANDIDATE_SECTIONS
    kept = kept[:MAX_EXACT_MATCHES]
    if not kept:
        return ClauseMatches([])

    text_rows = await pool.fetch(
        """
        SELECT s.id, substr(d.text, s.char_start + 1, s.char_end - s.char_start) AS text
        FROM strata_vectors.document_sections s
        JOIN strata_vectors.documents d ON d.id = s.document_id
        WHERE s.id = ANY($1::bigint[])
        """,
        [row["id"] for row in kept],
    )
    texts = {r["id"]: r["text"] for r in text_rows}
    chunks: list[dict[str, Any]] = []
    for row in kept:
        text = (texts.get(row["id"]) or "").rstrip()
        if text:
            chunks.append(_exact_chunk(row, text))
    return ClauseMatches(chunks, unread, unread_is_minimum)


def _exact_chunk(row: Any, text: str) -> dict[str, Any]:
    """A resolved section as a strata_ask chunk, marked exact."""
    chunk = ask_chunk(
        chunk_id=None,
        content=text,
        score=1.0,
        similarity=None,
        layer=row["layer_name"],
        source_file=row["source_file"],
        section=row["path"],
        doc_type=row["doc_type"],
        metadata={},
        document_id=row["document_id"],
        char_start=row["char_start"],
        char_end=row["char_start"] + len(text),
        clause_path=row["path"],
    )
    chunk["match"] = "exact"
    return chunk


def fit_to_budget(chunk: dict[str, Any], budget: int) -> dict[str, Any]:
    """chunk, its text cut at a word so its passage fits budget tokens.

    The cut text is still the document's text from char_start, and char_end
    moves with it; "cut" marks a chunk that lost text, so both answer modes can
    say part of the clause is missing. At least _MIN_TRIMMED_CHARS characters are
    kept (cut back to a word break past half of them), so a clause no longer
    than that comes back whole and unmarked; fit_to_budget(chunk, 1) is the
    cheapest cut.
    """
    if passage_cost(chunk) <= budget:
        return chunk
    content = chunk["content"]
    if len(content) <= _MIN_TRIMMED_CHARS:
        return chunk
    cut = len(content)
    trimmed = chunk
    while cut > _MIN_TRIMMED_CHARS:
        cut = max(int(cut * 0.9), _MIN_TRIMMED_CHARS)
        piece = content[:cut]
        space = max(piece.rfind(" "), piece.rfind("\n"))
        if space > cut // 2:
            piece = piece[:space]
        piece = piece.rstrip()
        trimmed = {
            **chunk, "content": piece, "char_end": chunk["char_start"] + len(piece), "cut": True,
        }
        if passage_cost(trimmed) <= budget:
            return trimmed
    # The last pass cut at _MIN_TRIMMED_CHARS: the floor
    return trimmed


def max_exact_chars(budget: int) -> int:
    """The widest exact passage, in characters, that fit_to_budget returns for budget tokens.

    As many characters as estimate_tokens counts as budget tokens (chars_for_tokens).
    """
    return max(chars_for_tokens(budget), _MIN_TRIMMED_CHARS)


def fit_clauses(chunks: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Each chunk fit to the budget left, divided by the chunks still to fit, in order.

    A chunk's share never eats into the floors (fit_to_budget(chunk, 1)) of the
    chunks after it, so a later clause an even split keeps is not pushed out.
    A chunk's own floor can pass its share, and the later ones then share what
    is left. None is dropped, even when nothing is left: each of those comes
    back at its floor, and packing decides what fits.
    """
    floors = [passage_cost(fit_to_budget(chunk, 1)) for chunk in chunks]
    fitted: list[dict[str, Any]] = []
    remaining = budget
    for i, chunk in enumerate(chunks):
        share = min(remaining // (len(chunks) - i), remaining - sum(floors[i + 1:]))
        piece = fit_to_budget(chunk, max(share, 1))
        fitted.append(piece)
        remaining -= passage_cost(piece)
    return fitted


async def lookup(
    database_url: str,
    question: str,
    layers: list[str] | None,
    budget: int,
    extra_patterns: Iterable[str | re.Pattern[str]] = (),
) -> ClauseMatches:
    """The exact chunks for the references in question, each fit to its share of budget.

    The chunks are empty when the question names no section, none resolves, or
    the lookup fails; retrieval then runs as usual.
    """
    found = _all_references(question, extra_patterns)
    if not found:
        return ClauseMatches([])
    references = found[:MAX_REFERENCES]
    try:
        matches = await resolve(database_url, references, layers)
    except Exception as e:
        logger.warning("clause_lookup_failed", error_type=type(e).__name__)
        return ClauseMatches([])
    return matches._replace(
        chunks=fit_clauses(matches.chunks, budget),
        references=len(references),
        skipped_references=tuple(ref.text for ref in found[MAX_REFERENCES:]),
    )


def cap_note(matches: ClauseMatches, layer_named: bool) -> str | None:
    """A note on what the lookup caps left out, or None.

    Matching sections past MAX_EXACT_MATCHES are counted (when the candidate
    limit was hit and none is counted, more may still match), with advice only where it
    helps: fewer clauses when more than one reference was looked up, a layer
    when the caller named none. References past MAX_REFERENCES are named as
    written: they were not looked up.
    """
    sentences = []
    base = None
    if matches.unread:
        n = matches.unread
        base = (
            f"{'At least ' if matches.unread_is_minimum else ''}{n} more matching "
            f"{'section' if n == 1 else 'sections'} not read"
        )
    elif matches.unread_is_minimum:
        base = "More sections may match than were read"
    if base is not None:
        advice = []
        if matches.references > 1:
            advice.append("ask about fewer clauses at a time")
        if not layer_named:
            advice.append("name a layer to narrow the question")
        sentences.append(base + ("; " + ", or ".join(advice) if advice else "") + ".")
    if matches.skipped_references:
        n = len(matches.skipped_references)
        sentences.append(
            f"{n} more {'reference' if n == 1 else 'references'} not looked up: "
            f"{', '.join(matches.skipped_references)}. "
            f"Ask about at most {MAX_REFERENCES} clauses at a time."
        )
    return " ".join(sentences) or None


def in_exact_span(
    document_id: int | None,
    char_start: int | None,
    char_end: int | None,
    exact: Sequence[dict[str, Any]],
) -> bool:
    """True when the span lies inside an exact chunk's span, so its text is there."""
    if document_id is None or char_start is None or char_end is None:
        return False
    return any(
        e["document_id"] == document_id
        and e["char_start"] <= char_start
        and char_end <= e["char_end"]
        for e in exact
    )


def without_exact_spans(
    chunks: Sequence[dict[str, Any]], exact: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """chunks without those whose span lies inside an exact chunk's span."""
    return [
        c
        for c in chunks
        if not in_exact_span(c.get("document_id"), c.get("char_start"), c.get("char_end"), exact)
    ]
