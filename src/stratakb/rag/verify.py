"""Checks a drafted answer against the passages it cites: quotes and facts.

Two deterministic checks, in Python only (no model call):

- the quote check: every quoted span ("..." [S2]) must appear in the cited
  passage, or in any passage when it cites none, after whitespace, quote
  mark and dash normalisation;
- the fact check: every number, date, amount, percentage, identifier and
  clause reference outside the quotes must appear in at least one passage.
  Both sides are normalised, so "thirty (30) days" matches "30 days",
  "1,000.00" matches "1000" and "4 November 2026" matches "2026-11-04".

A citation to a passage id that was not given is flagged too. A paraphrase
that changes the meaning without changing a fact is not caught.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

# At most this many flags are listed; the rest are only counted
MAX_FLAGS = 50

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
        ids.extend(f"S{n}" for n in span)
    return ids


def _quote_found(quote: str, text: str) -> bool:
    """Whether every part of the quote ("..." marks an omission) is in text, in order."""
    parts = [p.strip() for p in quote.split("...") if p.strip()]
    pos = 0
    for part in parts:
        i = text.find(part, pos)
        if i < 0:
            # Punctuation inside the closing quote mark need not be the source's
            i = text.find(part.rstrip(".,;:!?"), pos)
            if i < 0:
                return False
        pos = i + len(part)
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
_WORDS = rf"(?<![\w-]){_WORD}(?:(?:\s+and\s+|\s+|-){_WORD})*(?![\w])"
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

_IDENTIFIERS = [
    # IBAN-shaped, printed in groups of four or compact
    re.compile(r"(?<![\w])[A-Z]{2}\d{2}(?: [A-Z0-9]{4}){3,7}(?: [A-Z0-9]{1,3})?(?![\w])"),
    re.compile(r"(?<![\w])[A-Z]{2}\d{2}[A-Z0-9]{10,30}(?![\w])"),
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
_CLAUSES = [
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

    @property
    def verified(self) -> bool:
        return not self.flags and not self.flags_unlisted


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


def extract_facts(text: str) -> list[Fact]:
    """The numbers, dates, amounts, percentages, identifiers and clause references in text.

    Identifiers are taken first, then dates, clause references, amounts,
    quantities and bare numbers; each match is blanked before the next kind
    is looked for, so a number inside pacs.008, a date or "thirty (30) days"
    is not counted again on its own. Numbers in words count only with a unit
    or next to the digits.
    """
    work = list(text)
    found: list[tuple[int, Fact]] = []
    for pattern in _IDENTIFIERS:
        for m in _take(work, pattern):
            key = ("id", m.group(0).replace(" ", ""))
            found.append((m.start(), Fact("identifier", m.group(0), key)))
    for kind, pattern in _DATES:
        for m in _take(work, pattern):
            if fact := _date_fact(kind, m):
                found.append((m.start(), fact))
    for pattern in _CLAUSES:
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
    # In the order they are written
    return [fact for _, fact in sorted(found, key=lambda pair: pair[0])]


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
    numbers = [int(i[1:]) for i in ids]
    if len(ids) > 2 and numbers == list(range(numbers[0], numbers[0] + len(numbers))):
        return f"{ids[0]}–{ids[-1]}"
    return ", ".join(ids)


def check(
    draft: str, passages: dict[str, str], given: Sequence[str] | None = None
) -> CheckResult:
    """Check the draft's citations, quotes and facts against the passages.

    passages maps an id (S1, S2, ...) to the passage's stored text. given
    lists every id the caller passed, including ones whose text could not be
    used (it defaults to the keys of passages); a citation outside it is
    flagged.
    """
    given_ids = list(given if given is not None else passages)
    lines = _strip_list_markers(normalise_text(draft, keep_lines=True))
    flat_draft = normalise_text(lines)
    indexes = {pid: _PassageIndex(text) for pid, text in passages.items()}
    flags: list[Flag] = []
    seen: set[tuple[str, str]] = set()
    unlisted = 0

    def flag(kind: str, text: str, message: str, searched: Sequence[str]) -> None:
        nonlocal unlisted
        if (kind, text) in seen:
            return
        seen.add((kind, text))
        if len(flags) < MAX_FLAGS:
            flags.append(Flag(kind, text, message, list(searched)))
        else:
            unlisted += 1

    for marker in _CITATION.finditer(flat_draft):
        for pid in _cited_ids(marker.group(0)):
            if pid not in given_ids:
                flag(
                    "citation",
                    pid,
                    f"{pid} is cited, but the passages given are {_id_range(given_ids)}",
                    given_ids,
                )

    # Quotes are checked whole; their text is then left out of the fact check
    masked = list(flat_draft)
    quotes = 0
    for m in _QUOTE.finditer(flat_draft):
        quote = m.group(1).strip()
        masked[m.start():m.end()] = " " * (m.end() - m.start())
        if not re.search(r"\w", quote):
            continue
        quotes += 1
        cited = [pid for c in _CITATION.finditer(m.group(2)) for pid in _cited_ids(c.group(0))]
        targets = [pid for pid in (cited or list(indexes)) if pid in indexes]
        if not any(_quote_found(quote, indexes[pid].flat) for pid in targets):
            searched = cited or given_ids
            flag(
                "quote",
                quote,
                f'Quote "{quote}" not found in {_id_range(searched)}',
                searched,
            )
    for marker in _CITATION.finditer(flat_draft):
        masked[marker.start():marker.end()] = " " * (marker.end() - marker.start())

    # Each distinct fact is checked once, however often it is written
    facts: dict[tuple, Fact] = {}
    for fact in extract_facts("".join(masked)):
        facts.setdefault(fact.key, fact)
    for fact in facts.values():
        if not any(index.has(fact) for index in indexes.values()):
            flag(
                fact.kind,
                fact.text,
                f'"{fact.text}" not found in {_id_range(given_ids)}',
                given_ids,
            )
    return CheckResult(
        flags=flags, quotes_checked=quotes, facts_checked=len(facts), flags_unlisted=unlisted
    )
