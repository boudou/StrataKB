"""Checks a drafted answer against the passages it cites: quotes and facts.

Two deterministic checks, in Python only (no model call):

- the quote check: every quoted span ("..." [S2]) must appear in the cited
  passage, or in any passage when it cites none, after whitespace, quote
  mark and dash normalisation;
- the fact check: every number, date, amount, percentage, identifier and
  clause reference outside the quotes must appear in a passage cited in the
  same sentence, or in any passage when the sentence cites none. Both sides
  are normalised, so "thirty (30) days" matches "30 days", "1,000.00" matches
  "1000" and "4 November 2026" matches "2026-11-04".

A citation to a passage id that was not given is flagged too, and so is a
draft with no citations at all, even when its quotes and facts match. A draft
with no quote and no fact to check is not verified: there was nothing to
compare. A paraphrase that changes the meaning without changing a fact is not
caught. The sentences with no quote and no fact are listed, not judged: the
verdict covers only what was checked.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import cached_property

from stratakb.rag.passages import passage_id

# At most this many flags are listed; the rest are only counted
MAX_FLAGS = 50
# At most this many unchecked sentences are listed; the rest are only counted
MAX_UNCHECKED_SENTENCES = 50

# Curly and typographic quote marks, dashes and spaces, mapped to plain ones
_TRANSLATE = str.maketrans(
    {
        **dict.fromkeys("“”„‟«»″", '"'),
        **dict.fromkeys("‘’‚‛′", "'"),
        **dict.fromkeys("‐‑‒–—―−", "-"),
        **dict.fromkeys("           ", " "),
        "…": "...",
    }
)


def normalise_text(text: str, *, keep_lines: bool = False) -> str:
    """Plain quote marks, dashes and spaces, and runs of whitespace collapsed.

    With keep_lines, line breaks survive (one per run) and only spaces and
    tabs collapse.
    """
    text = text.translate(_TRANSLATE)
    if keep_lines:
        text = re.sub(r"[^\S\n]+", " ", text)
        return re.sub(r" ?\n[\s]*", "\n", text).strip()
    return re.sub(r"\s+", " ", text).strip()


# A numbered list's markers (1., 2) ...) are layout, not facts
_LIST_MARKER = re.compile(r"^(\d{1,3})[.)] +", re.M)


def _strip_list_markers(lines_text: str) -> str:
    """Text kept line by line (normalise_text with keep_lines), list markers removed."""
    return _LIST_MARKER.sub("", lines_text)


# ---------------------------------------------------------------------------
# Citations and quotes
# ---------------------------------------------------------------------------

# [S1], [S1, S3], [S1-S3] (after normalisation every dash is "-")
_CITATION = re.compile(r"\[\s*S\d+(?:\s*[,;-]\s*S?\d+)*\s*\]")
_CITED_ID = re.compile(r"S?(\d+)(?:\s*-\s*S?(\d+))?")
# A quoted span and the citations right after it
_QUOTE = re.compile(r'"([^"]+)"((?:[ ,.;:]*\[[^\]]*\])*)')


def _cited_ids(marker: str) -> list[str]:
    """The passage ids in one citation marker; a range S1-S3 is expanded."""
    ids: list[str] = []
    for m in _CITED_ID.finditer(marker.strip("[] ")):
        first = int(m.group(1))
        last = int(m.group(2)) if m.group(2) else first
        # A reversed or huge range is read as its two ends only
        span = range(first, last + 1) if 0 <= last - first <= 50 else (first, last)
        ids.extend(passage_id(n) for n in span)
    return ids


# A line break, or end punctuation (with any closing marks) before whitespace
_SENTENCE_BREAK = re.compile(r"\n|[.!?][)\]'\"]*(?=\s|\Z)")
_CITATION_RUN = rf"{_CITATION.pattern}(?:[ ,;]*{_CITATION.pattern})*"
# Citations right after a sentence's end, and a period that goes with them
_TRAILING_CITATIONS = re.compile(rf" *{_CITATION_RUN}\.?(?=\s|\Z)")
# Lines that hold nothing but citations
_CITATION_LINES = re.compile(rf"(?:{_CITATION_RUN}\.?(?:\n|\Z))*")
_NEXT_CHAR = re.compile(r" *(\S?)")


def _sentence_scopes(text: str) -> tuple[list[int], list[tuple[str, ...]]]:
    """Where each sentence of text starts, and the passage ids each one cites.

    text keeps its line breaks. A line break ends a sentence, and so does end
    punctuation, unless a lower-case letter follows, or a digit follows
    anything but a citation ("Art. 12" stays whole). Citations right after the
    end belong to the sentence just ended, unless a lower-case word follows
    them ("[S2] says ..."); after a line break, only when alone on their line.
    """
    starts = [0]
    for m in _SENTENCE_BREAK.finditer(text):
        if m.end() <= starts[-1]:
            continue
        if m.group(0) == "\n":
            starts.append(_CITATION_LINES.match(text, m.end()).end())
            continue
        following = _NEXT_CHAR.match(text, m.end()).group(1)
        if not following:
            continue  # The line break, or the end of the text, closes it
        if following.islower():
            continue
        if following.isdigit() and text[m.start() - 1 : m.start()] != "]":
            continue
        trailing = _TRAILING_CITATIONS.match(text, m.end())
        if trailing is None:
            starts.append(m.end())
            continue
        after = _NEXT_CHAR.match(text, trailing.end()).group(1)
        if after.islower():
            starts.append(m.end())
        elif after:
            starts.append(trailing.end())
    cited: list[list[str]] = [[] for _ in starts]
    for marker in _CITATION.finditer(text):
        cited[bisect_right(starts, marker.start()) - 1].extend(_cited_ids(marker.group(0)))
    return starts, [tuple(dict.fromkeys(ids)) for ids in cited]


# Each part of a quote with an omission must hold at least this many words, so
# fragments such as "a ... e" cannot pass as a quote
_MIN_PART_WORDS = 2


def _find_part(part: str, text: str, pos: int) -> re.Match[str] | None:
    """The first match of part in text from pos, not starting or ending inside a word.

    A part ending in a digit must not stop inside a number ("1" of "1,000.00").
    """
    head = r"(?<!\w)" if part[:1].isalnum() or part[:1] == "_" else ""
    tail = r"(?!\w)" if part[-1:].isalnum() or part[-1:] == "_" else ""
    if part[-1:].isdigit():
        tail = r"(?!\w|[.,]\d)"
    return re.compile(head + re.escape(part) + tail).search(text, pos)


def _quote_found(quote: str, text: str) -> bool:
    """Whether every part of the quote ("..." marks an omission) is in text, in order.

    Parts match whole words only, and with an omission each part needs
    _MIN_PART_WORDS words.
    """
    parts = [p.strip() for p in quote.split("...") if p.strip()]
    if len(parts) > 1 and any(len(p.split()) < _MIN_PART_WORDS for p in parts):
        return False
    pos = 0
    for part in parts:
        m = _find_part(part, text, pos)
        if m is None:
            # Punctuation inside the closing quote mark need not be the source's
            trimmed = part.rstrip(".,;:!?")
            m = _find_part(trimmed, text, pos) if trimmed else None
            if m is None:
                return False
        pos = m.end()
    return bool(parts)


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_WORD = "(?:" + "|".join(sorted([*_SMALL, *_SCALES], key=len, reverse=True)) + ")"
# At most 12 words in one number ("nine hundred and ninety nine thousand ..."),
# so the run stays linear (see the linear-time rules in knowledge/structure.py)
_WORDS = rf"(?<![\w-]){_WORD}(?:(?:\s+and\s+|\s+|-){_WORD}){{0,11}}(?![\w])"
_DIGITS = r"(?<![\w.,])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:st|nd|rd|th)?(?![\w])"
# A number in digits, in words, or both ("thirty (30)", "30 (thirty)")
_NUMBER = (
    rf"(?:{_WORDS}\s*\(\s*{_DIGITS}\s*\)|{_DIGITS}\s*\(\s*{_WORDS}\s*\)|{_DIGITS}|{_WORDS})"
)

# A number on its own: in digits, or in words next to the digits
_BARE_NUMBER = re.compile(
    rf"{_WORDS}\s*\(\s*{_DIGITS}\s*\)|{_DIGITS}\s*\(\s*{_WORDS}\s*\)|{_DIGITS}", re.IGNORECASE
)

_UNIT = (
    r"(?:(?:business|working|banking|calendar)\s+)?"
    r"(?:days?|weeks?|months?|years?|hours?|minutes?|seconds?)"
    r"|per\s?cent|percent"
)
_QUANTITY = re.compile(
    rf"(?P<num>{_NUMBER})(?:\s*-\s*|\s+)(?P<unit>{_UNIT})(?![\w])|(?P<pnum>{_DIGITS})\s?%",
    re.IGNORECASE,
)

_CURRENCIES = {"EUR": "EUR", "€": "EUR", "USD": "USD", "$": "USD", "GBP": "GBP", "£": "GBP",
               "CHF": "CHF", "JPY": "JPY"}
_CURRENCY = r"(?:EUR|USD|GBP|CHF|JPY|€|\$|£)"
# English (1,000.00), European with a decimal comma (1.000,00) or plain
_AMOUNT_NUM = r"(?:\d{1,3}(?:\.\d{3})+,\d{1,2}|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d{1,2})?)"
_AMOUNT = re.compile(
    rf"(?<![\w])(?:(?P<c1>{_CURRENCY})\s?(?P<n1>{_AMOUNT_NUM})"
    rf"|(?P<n2>{_AMOUNT_NUM})\s?(?P<c2>{_CURRENCY}))(?![\d])"
)

_MONTHS = {
    name: n
    for n, names in enumerate(
        [
            ("January", "Jan"), ("February", "Feb"), ("March", "Mar"), ("April", "Apr"),
            ("May",), ("June", "Jun"), ("July", "Jul"), ("August", "Aug"),
            ("September", "Sep", "Sept"), ("October", "Oct"), ("November", "Nov"),
            ("December", "Dec"),
        ],
        start=1,
    )
    for name in names
}
# Capitalised only, so the verb "may" is not a month
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_DATES = [
    ("iso", re.compile(r"(?<![\w-])(\d{4})-(\d{2})-(\d{2})(?![\w-])")),
    ("dmy", re.compile(rf"(?<![\w]){_DAY}\s+(?:of\s+)?({_MONTH}),?\s+(\d{{4}})(?![\w])")),
    ("mdy", re.compile(rf"(?<![\w])({_MONTH})\s+{_DAY},?\s+(\d{{4}})(?![\w])")),
    ("dm", re.compile(rf"(?<![\w]){_DAY}\s+(?:of\s+)?({_MONTH})(?![\w])")),
    ("md", re.compile(rf"(?<![\w])({_MONTH})\s+{_DAY}(?![\w])")),
    ("my", re.compile(rf"(?<![\w])({_MONTH})\s+(\d{{4}})(?![\w])")),
    # 04/11/2026 could be either order: compared only as written
    ("literal", re.compile(r"(?<![\w./])\d{1,2}[/.]\d{1,2}[/.]\d{2,4}(?![\w./])")),
]

# IBAN-shaped, printed in groups of four or compact: account data, not names
ACCOUNT_PATTERNS = [
    re.compile(r"(?<![\w])[A-Z]{2}\d{2}(?: [A-Z0-9]{4}){3,7}(?: [A-Z0-9]{1,3})?(?![\w])"),
    re.compile(r"(?<![\w])[A-Z]{2}\d{2}[A-Z0-9]{10,30}(?![\w])"),
]

IDENTIFIER_PATTERNS = [
    # Message ids such as pacs.008 or pacs.008.001.08
    re.compile(r"(?<![\w.])[a-z]{4}\.\d{3}(?:\.\d{3}\.\d{2})?(?![\w.])"),
    # Codes with a dash, such as AT-02
    re.compile(r"(?<![\w-])[A-Z]{1,5}-\d+[A-Z0-9]*(?![\w-])"),
    # Letters then digits, such as AC01, MS03 or ISO20022
    re.compile(r"(?<![\w])[A-Za-z]+\d+[A-Za-z0-9]*(?![\w])"),
]

_CLAUSE_WORD = (
    r"(?:Article|Art\.|Section|Sec\.|Clause|Paragraph|Para\.|Chapter|Annex|Appendix"
    r"|Schedule|Rule|§)"
)
_CLAUSE_NUM = r"\d+[A-Za-z]?(?:\.\d+)*(?:\(\w{1,4}\))*"
CLAUSE_PATTERNS = [
    re.compile(rf"(?<![\w]){_CLAUSE_WORD}\s*(?P<num>{_CLAUSE_NUM})(?![\w]|\.\d)", re.IGNORECASE),
    # 7(2)(b) or 7.2.1 without a keyword
    re.compile(r"(?<![\w.])(?P<num>\d+(?:\.\d+)*(?:\(\w{1,4}\))+)(?![\w])"),
    re.compile(r"(?<![\w.])(?P<num>\d+(?:\.\d+){2,})(?![\w]|\.\d)"),
]


@dataclass(frozen=True)
class Fact:
    """A checkable value: its kind, the text as written and a canonical key."""

    kind: str
    text: str
    key: tuple


@dataclass
class Flag:
    """One problem in the draft: what kind, the offending text and why."""

    kind: str
    text: str
    message: str
    passages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckResult:
    """The flags (at most MAX_FLAGS), how many more there were, and the counts checked."""

    flags: list[Flag]
    quotes_checked: int
    facts_checked: int
    flags_unlisted: int = 0
    # The sentences with no quote and no fact checked (at most
    # MAX_UNCHECKED_SENTENCES), and how many more there were
    unchecked_sentences: list[str] = field(default_factory=list)
    unchecked_unlisted: int = 0


def _nothing_checked(quotes: int, facts: int) -> bool:
    """Whether the draft had no quote and no fact to compare."""
    return not quotes and not facts


def verdict(result: CheckResult, extra_flags: int = 0) -> str:
    """The verdict on a checked draft: flagged, nothing_checked or verified.

    extra_flags counts the flags raised outside check(), such as a passage
    that could not be used. The verdict covers only what was checked: the
    unchecked sentences do not change it.
    """
    if result.flags or result.flags_unlisted or extra_flags:
        return "flagged"
    if _nothing_checked(result.quotes_checked, result.facts_checked):
        return "nothing_checked"
    return "verified"


def _words_value(text: str) -> int:
    """The value of a number in words, such as "one hundred and twenty"."""
    total = current = 0
    for word in re.split(r"[\s-]+", text.lower()):
        if word in _SMALL:
            current += _SMALL[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word in _SCALES:
            total += max(current, 1) * _SCALES[word]
            current = 0
    return total + current


def _decimal(text: str) -> Decimal | None:
    """A number in digits (English separators, ordinal suffix allowed)."""
    try:
        return Decimal(re.sub(r"(st|nd|rd|th)$", "", text).replace(",", ""))
    except InvalidOperation:
        return None


def _number_value(text: str) -> Decimal | None:
    """The value of a number in digits, words or both; the digits win."""
    digits = re.search(_DIGITS, text)
    if digits:
        return _decimal(digits.group(0))
    return Decimal(_words_value(text))


def _amount_value(text: str) -> Decimal | None:
    """An amount's value: 1.000,00 (European) or 1,000.00 (English)."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{1,2}", text):
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d+,\d{1,2}", text):
        text = text.replace(",", ".")
    return _decimal(text)


def _unit(text: str) -> str:
    """A unit in canonical form: day, business day, month, %."""
    unit = re.sub(r"\s+", " ", text.lower())
    if unit.replace(" ", "") == "percent":
        return "%"
    unit = unit.removeprefix("calendar ")
    unit = re.sub(r"^(working|banking) ", "business ", unit)
    return unit.removesuffix("s")


def _iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _date_fact(kind: str, m: re.Match) -> Fact | None:
    text = m.group(0)
    if kind == "literal":
        return Fact("date", text, ("literal", text))
    if kind == "iso":
        iso = _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return Fact("date", text, ("date", iso)) if iso else None
    if kind == "my":
        return Fact("date", text, ("month", f"{m.group(2)}-{_MONTHS[m.group(1).rstrip('.')]:02d}"))
    if kind in ("dmy", "dm"):
        day, month = int(m.group(1)), _MONTHS[m.group(2).rstrip(".")]
    else:
        month, day = _MONTHS[m.group(1).rstrip(".")], int(m.group(2))
    if kind in ("dmy", "mdy"):
        iso = _iso(int(m.group(3)), month, day)
        return Fact("date", text, ("date", iso)) if iso else None
    if not 1 <= day <= 31:
        return None
    return Fact("date", text, ("day", f"{month:02d}-{day:02d}"))


def _take(text: list[str], pattern: re.Pattern) -> Iterator[re.Match]:
    """Each match of pattern in the working text, blanked out once taken."""
    for m in list(pattern.finditer("".join(text))):
        if not "".join(text[m.start():m.end()]).strip():
            continue
        yield m
        text[m.start():m.end()] = " " * (m.end() - m.start())


def _located_facts(text: str) -> list[tuple[int, Fact]]:
    """The facts in text with where each one starts, in the order they are written.

    Identifiers are taken first, then dates, clause references, amounts,
    quantities and bare numbers; each match is blanked before the next kind
    is looked for, so a number inside pacs.008, a date or "thirty (30) days"
    is not counted again on its own. Numbers in words count only with a unit
    or next to the digits.
    """
    work = list(text)
    found: list[tuple[int, Fact]] = []
    for pattern in (*ACCOUNT_PATTERNS, *IDENTIFIER_PATTERNS):
        for m in _take(work, pattern):
            key = ("id", m.group(0).replace(" ", ""))
            found.append((m.start(), Fact("identifier", m.group(0), key)))
    for kind, pattern in _DATES:
        for m in _take(work, pattern):
            if fact := _date_fact(kind, m):
                found.append((m.start(), fact))
    for pattern in CLAUSE_PATTERNS:
        for m in _take(work, pattern):
            key = ("clause", m.group("num").lower())
            found.append((m.start(), Fact("clause", m.group(0), key)))
    for m in _take(work, _AMOUNT):
        currency = _CURRENCIES[m.group("c1") or m.group("c2")]
        value = _amount_value(m.group("n1") or m.group("n2"))
        if value is not None:
            found.append((m.start(), Fact("amount", m.group(0), ("amount", currency, value))))
    for m in _take(work, _QUANTITY):
        if m.group("pnum"):
            value, unit = _decimal(m.group("pnum")), "%"
        else:
            value, unit = _number_value(m.group("num")), _unit(m.group("unit"))
        if value is not None:
            found.append((m.start(), Fact("quantity", m.group(0), ("quantity", unit, value))))
    for m in _take(work, _BARE_NUMBER):
        value = _number_value(m.group(0))
        if value is not None:
            found.append((m.start(), Fact("number", m.group(0), ("number", value))))
    return sorted(found, key=lambda pair: pair[0])


def extract_facts(text: str) -> list[Fact]:
    """The numbers, dates, amounts, percentages, identifiers and clause references in text."""
    return [fact for _, fact in _located_facts(text)]


class _PassageIndex:
    """One passage's facts and text, for the fact check."""

    def __init__(self, text: str):
        self.flat = normalise_text(text)
        self.lines = normalise_text(text, keep_lines=True)
        self.lower_lines = self.lines.lower()
        self.keys: set[tuple] = set()
        self.values: set[Decimal] = set()
        # Only amounts and bare numbers: a day count or an article number is no amount
        self.amount_values: set[Decimal] = set()
        self.currencies = {
            code for symbol, code in _CURRENCIES.items() if symbol in self.flat
        }
        # A list marker may still back a bare number (a heading "12. Termination"),
        # never an amount
        for m in _LIST_MARKER.finditer(self.lines):
            self.values.add(Decimal(m.group(1)))
        for fact in extract_facts(normalise_text(_strip_list_markers(self.lines))):
            self.keys.add(fact.key)
            if fact.key[0] in ("number", "quantity", "amount"):
                self.values.add(fact.key[-1])
                if fact.key[0] != "quantity":
                    self.amount_values.add(fact.key[-1])
            elif fact.key[0] == "date" and fact.key[1]:
                year, month, day = fact.key[1].split("-")
                self.keys.add(("day", f"{month}-{day}"))
                self.keys.add(("month", f"{year}-{month}"))
                self.values.add(Decimal(year))
            elif fact.key[0] == "clause":
                value = _decimal(fact.key[1])
                if value is not None:
                    self.values.add(value)

    def _literal(self, text: str) -> bool:
        # The plain substring test first: most misses end there, without a regex
        if text not in self.flat:
            return False
        return re.search(rf"(?<![\w.]){re.escape(text)}(?![\w])", self.flat) is not None

    def has(self, fact: Fact) -> bool:
        if fact.key in self.keys:
            return True
        kind = fact.key[0]
        if kind == "id":
            return self._literal(fact.key[1])
        if kind == "literal":
            return self._literal(fact.text)
        if kind == "clause":
            if fact.key[1] not in self.lower_lines:
                return False
            # A heading or list label such as "7.2 Notice" or "12. Termination"
            return re.search(
                rf"(?:^|\n)[#*\s]*{re.escape(fact.key[1])}[.)]?(?:\s|$)",
                self.lines,
                re.IGNORECASE,
            ) is not None
        if kind == "amount":
            _, currency, value = fact.key
            return value in self.amount_values and currency in self.currencies
        if kind == "number":
            return fact.key[1] in self.values
        return False


def _id_range(ids: Sequence[str]) -> str:
    """S1-S4 for a consecutive run, S1, S3 otherwise; "no passage" when empty."""
    if not ids:
        return "no passage"
    # Not passage_number: a draft may cite S0, which is no passage id but still has a number
    numbers = [int(i[1:]) for i in ids]
    if len(ids) > 2 and numbers == list(range(numbers[0], numbers[0] + len(numbers))):
        return f"{ids[0]}–{ids[-1]}"
    return ", ".join(ids)


class _Scope:
    """One set of passage ids to search, shared by everything checked against it.

    A sentence can cite tens of thousands of ids, so checking or flagging a fact
    must not cost time in proportion to them: the work is done once here, and a
    scope is hashed by identity.
    """

    def __init__(self, ids: tuple[str, ...], indexes: dict[str, _PassageIndex]) -> None:
        self.ids = ids
        self.targets = [indexes[pid] for pid in ids if pid in indexes]

    @cached_property
    def label(self) -> str:
        return _id_range(self.ids)


def _uncovered_sentences(lines: str, starts: list[int], covered: list[bool]) -> Iterator[str]:
    """The text of each sentence not covered, on one line, in draft order.

    A sentence that holds nothing but citations and punctuation is skipped.
    """
    ends = [*starts[1:], len(lines)]
    for start, end, is_covered in zip(starts, ends, covered, strict=True):
        if is_covered:
            continue
        text = normalise_text(lines[start:end])
        if re.search(r"\w", _CITATION.sub(" ", text)):
            yield text


class _Capped:
    """A list of at most limit items; the items past it are only counted."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.items: list = []
        self.unlisted = 0

    def add(self, item: object) -> None:
        if len(self.items) < self.limit:
            self.items.append(item)
        else:
            self.unlisted += 1


class _Checking:
    """What the stages of one check() share: the passages, the scopes and the flags.

    One scope per set of ids for the whole call: a scope is hashed by
    identity, so the flag dedup and the per-scope fact check rely on it.
    """

    def __init__(self, passages: dict[str, str], given_ids: list[str]) -> None:
        self.given_ids = given_ids
        self.indexes = {pid: _PassageIndex(text) for pid, text in passages.items()}
        self.scopes: dict[tuple[str, ...], _Scope] = {}
        self.seen: set[tuple[str, str, _Scope]] = set()
        self.flags = _Capped(MAX_FLAGS)

    def scope_of(self, cited: Sequence[str]) -> _Scope:
        """The scope of these citations; with none, all the passages given."""
        ids = tuple(cited or self.given_ids)
        if ids not in self.scopes:
            self.scopes[ids] = _Scope(ids, self.indexes)
        return self.scopes[ids]

    def flag(self, kind: str, text: str, message: str, scope: _Scope) -> None:
        """Flag once per kind, text and scope; message goes before the scope's label."""
        if (kind, text, scope) in self.seen:
            return
        self.seen.add((kind, text, scope))
        self.flags.add(Flag(kind, text, f"{message}{scope.label}", list(scope.ids)))


def _check_citations(state: _Checking, flat_draft: str) -> None:
    """Flag a draft with no citations, and each citation of an id not given."""
    # First, so the cap on flags cannot drop it. A [S1] inside a quote is the
    # document's text; the citations right after a quote stay
    if not _CITATION.search(_QUOTE.sub(r" \2", flat_draft)):
        state.flag(
            "citation",
            "no citation",
            "The draft has no citations; cite each claim as [S1]. The passages given are ",
            state.scope_of(()),
        )
    for marker in _CITATION.finditer(flat_draft):
        for pid in _cited_ids(marker.group(0)):
            if pid not in state.given_ids:
                message = f"{pid} is cited, but the passages given are "
                state.flag("citation", pid, message, state.scope_of(()))


def _check_quotes(
    state: _Checking, flat_draft: str, lines: str
) -> tuple[int, list[int], list[str], list[str]]:
    """Check each quote against its scope.

    Returns the number of quotes checked, where each starts, the draft with
    quotes and citations blanked (for the fact check) and the draft with the
    quotes' text blanked (for the sentence scopes).
    """
    # Quotes are checked whole; their text is then left out of the fact check
    masked = list(flat_draft)
    # flat_draft and lines have the same length: each line break is one space.
    # For the sentence scopes a quote's text goes, but not its end punctuation
    scoped = list(lines)
    quotes = 0
    quote_starts: list[int] = []
    for m in _QUOTE.finditer(flat_draft):
        quote = m.group(1).strip()
        masked[m.start():m.end()] = " " * (m.end() - m.start())
        inner = len(m.group(1).rstrip(" .!?"))
        scoped[m.start(1):m.start(1) + inner] = " " * inner
        if not re.search(r"\w", quote):
            continue
        quotes += 1
        quote_starts.append(m.start())
        cited = [pid for c in _CITATION.finditer(m.group(2)) for pid in _cited_ids(c.group(0))]
        scope = state.scope_of(cited)
        if not any(_quote_found(quote, index.flat) for index in scope.targets):
            state.flag("quote", quote, f'Quote "{quote}" not found in ', scope)
    for marker in _CITATION.finditer(flat_draft):
        masked[marker.start():marker.end()] = " " * (marker.end() - marker.start())
    return quotes, quote_starts, masked, scoped


def _check_facts(
    state: _Checking, masked: list[str], scoped: list[str], quote_starts: list[int]
) -> tuple[int, list[int], list[bool]]:
    """Check each fact outside the quotes against its sentence's scope.

    Returns the number of distinct facts, where each sentence starts and
    whether each sentence holds a quote or a fact that was checked.
    """
    # Each fact is checked once per set of citations, however often it is written
    starts, cited_ids = _sentence_scopes("".join(scoped))
    sentence_scopes = [state.scope_of(cited) for cited in cited_ids]
    # A sentence is covered when it holds a quote or a fact that was checked
    covered = [False] * len(starts)
    for pos in quote_starts:
        covered[bisect_right(starts, pos) - 1] = True
    facts: set[tuple] = set()
    checked: set[tuple] = set()
    for pos, fact in _located_facts("".join(masked)):
        facts.add(fact.key)
        sentence = bisect_right(starts, pos) - 1
        covered[sentence] = True
        scope = sentence_scopes[sentence]
        if (fact.key, scope) in checked:
            continue
        checked.add((fact.key, scope))
        if not any(index.has(fact) for index in scope.targets):
            state.flag(fact.kind, fact.text, f'"{fact.text}" not found in ', scope)
    return len(facts), starts, covered


def _unchecked_sentences(
    lines: str, starts: list[int], covered: list[bool], quotes: int, facts: int
) -> _Capped:
    """The sentences compared with nothing, capped at MAX_UNCHECKED_SENTENCES.

    They are listed, not judged. When nothing was checked at all, every
    sentence is unchecked and none is listed.
    """
    unchecked = _Capped(MAX_UNCHECKED_SENTENCES)
    if not _nothing_checked(quotes, facts):
        for text in _uncovered_sentences(lines, starts, covered):
            unchecked.add(text)
    return unchecked


def check(
    draft: str, passages: dict[str, str], given: Sequence[str] | None = None
) -> CheckResult:
    """Check the draft's citations, quotes and facts against the passages.

    passages maps an id (S1, S2, ...) to the passage's stored text. given
    lists every id the caller passed, including ones whose text could not be
    used (it defaults to the keys of passages); a citation outside it is
    flagged.
    """
    state = _Checking(passages, list(given if given is not None else passages))
    lines = _strip_list_markers(normalise_text(draft, keep_lines=True))
    flat_draft = normalise_text(lines)
    # In this order: the flags are listed, and capped, in the order raised
    _check_citations(state, flat_draft)
    quotes, quote_starts, masked, scoped = _check_quotes(state, flat_draft, lines)
    facts, starts, covered = _check_facts(state, masked, scoped, quote_starts)
    unchecked = _unchecked_sentences(lines, starts, covered, quotes, facts)
    return CheckResult(
        flags=state.flags.items,
        quotes_checked=quotes,
        facts_checked=facts,
        flags_unlisted=state.flags.unlisted,
        unchecked_sentences=unchecked.items,
        unchecked_unlisted=unchecked.unlisted,
    )
