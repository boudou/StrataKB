"""The names a question asks about, and whether the passages hold them.

A question that names something (asyncio.TaskPool, PyConfig_Merge,
__match_kwargs__, config["timeout"], AC04, PEP 9123) is answered only by a
passage that holds the name. find_names picks out code-like names and
numbered references; check_names looks for them in the text going back to
the caller. Ordinary words never count as names: a name needs a digit, an
underscore, a dot, brackets or a capital inside a word. Clause references
("clause 2.2", "7.2.1") belong to the exact clause lookup, and IBAN-shaped
values are account data: neither is ever a name.

When a rule is unsure, a name counts as present, which keeps the status as
it was.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from stratakb.rag.layer_detect import pattern_text
from stratakb.rag.verify import (
    ACCOUNT_PATTERNS,
    CLAUSE_PATTERNS,
    IDENTIFIER_PATTERNS,
    normalise_text,
)

# At most this many names are checked; the rest of the question is ignored
MAX_NAMES = 10

_DUNDER = re.compile(r"(?<!\w)__[A-Za-z]\w*__(?!\w)")
_DOTTED = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?![\w])")
_KEY = re.compile(r"""\[\s*["']?([A-Za-z_][\w.-]*)["']?\s*\]""")
_SNAKE = re.compile(r"(?<![\w])_*[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+(?![\w])")
_CAMEL = re.compile(r"(?<![\w])\w*[a-z][A-Z]\w*(?![\w])")
_NUMBERED = re.compile(r"(?<![\w])([A-Z]{2,8})[\s:#-]?(\d+)(?!\w|\.\d)")
# A passage id pasted back into a follow-up question, such as [S1]
_PASSAGE_ID = re.compile(r"[Ss]\d+")
# Letters and a number, read apart so PEP 8 is found as PEP8 or pep-0008
_LETTERS_NUMBER = re.compile(r"([A-Za-z]+)[\s:#-]?(\d+)")
# Words that make a numbered reference a clause reference
_CLAUSE_KEYWORDS = frozenset(
    {
        "ARTICLE", "SECTION", "CLAUSE", "CHAPTER", "ANNEX", "APPENDIX",
        "SCHEDULE", "PART", "RULE", "PARAGRAPH",
    }
)
# A name standing alone: the rules a dotted name's last segment is tried with
_SINGLE = (_DUNDER, _SNAKE, _CAMEL, *IDENTIFIER_PATTERNS)


@dataclass(frozen=True)
class Name:
    """A name as written in the question, and how to find it in a passage.

    needles are plain strings, any one of which makes the name present;
    number is a pattern for a numbered reference, in place of the needles.
    Both are matched on the normalised form (see _squash).
    """

    text: str
    key: str
    needles: tuple[str, ...] = ()
    number: re.Pattern | None = None


@dataclass(frozen=True)
class NameCheck:
    """The names a question asks about and the ones no text holds, in question order."""

    names: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def none_present(self) -> bool:
        return bool(self.names) and len(self.missing) == len(self.names)


def _squash(text: str) -> str:
    """text compared without case, typographic marks or any whitespace."""
    return re.sub(r"\s+", "", normalise_text(text).casefold())


def _take(text: list[str], pattern: re.Pattern) -> Iterator[re.Match]:
    """Each match of pattern in the working text, blanked out once taken."""
    for m in list(pattern.finditer("".join(text))):
        if not "".join(text[m.start():m.end()]).strip():
            continue
        yield m
        text[m.start():m.end()] = " " * (m.end() - m.start())


def _numbered(text: str, letters: str, number: str) -> Name:
    """A numbered reference: the letters, any separator, then the number."""
    pattern = re.compile(
        rf"{re.escape(letters.casefold())}[\s:#_-]*0*{int(number)}(?!\d)"
    )
    return Name(text, f"{letters.casefold()}{int(number)}", number=pattern)


def _plain(text: str, *needles: str) -> Name:
    squashed = tuple(n for n in (_squash(x) for x in (needles or (text,))) if n)
    return Name(text, squashed[0] if squashed else _squash(text), needles=squashed)


def _dotted(text: str) -> Name | None:
    """A dotted name, found whole, by a tail of 2 segments or more, or by a named last segment."""
    segments = text.split(".")
    if all(len(s) == 1 for s in segments):
        return None
    needles = [".".join(segments[i:]) for i in range(len(segments) - 1)]
    last = segments[-1]
    if any(p.fullmatch(last) for p in _SINGLE):
        needles.append(last)
    return _plain(text, *needles)


def _identifier(text: str) -> Name | None:
    """A code such as AC04, AT-44 or pacs.008; a letter and digits (Q3, x86) is not one."""
    if "-" not in text and "." not in text and re.match(r"[A-Za-z]\d", text):
        return None
    if m := _LETTERS_NUMBER.fullmatch(text):
        return _numbered(text, m.group(1), m.group(2))
    return _plain(text)


def find_names(question: str) -> list[Name]:
    """The code-like names and numbered references in question, in question order.

    Read over the same prefix as the layer patterns. Each kind of match is
    blanked before the next is looked for; account numbers and clause
    references are blanked first and never count. Names that differ only in
    case, spacing or a number's leading zeros are kept once, as first written.
    At most MAX_NAMES.
    """
    work = list(normalise_text(pattern_text(question)))
    found: list[tuple[int, Name]] = []
    for pattern in (*ACCOUNT_PATTERNS, *CLAUSE_PATTERNS):
        for _ in _take(work, pattern):
            pass
    for m in _take(work, _DUNDER):
        found.append((m.start(), _plain(m.group(0))))
    for m in _take(work, _DOTTED):
        if name := _dotted(m.group(0)):
            found.append((m.start(), name))
    for m in _take(work, _KEY):
        key = m.group(1)
        if len(key) >= 2 and not _PASSAGE_ID.fullmatch(key):
            found.append((m.start(1), _plain(key)))
    for pattern in (_SNAKE, _CAMEL):
        for m in _take(work, pattern):
            found.append((m.start(), _plain(m.group(0))))
    for pattern in IDENTIFIER_PATTERNS:
        for m in _take(work, pattern):
            if name := _identifier(m.group(0)):
                found.append((m.start(), name))
    for m in _take(work, _NUMBERED):
        if m.group(1).upper() not in _CLAUSE_KEYWORDS:
            found.append((m.start(), _numbered(m.group(0), m.group(1), m.group(2))))

    names: list[Name] = []
    seen: set[str] = set()
    for _, name in sorted(found, key=lambda f: f[0]):
        if name.key not in seen:
            seen.add(name.key)
            names.append(name)
    return names[:MAX_NAMES]


def _present(name: Name, texts: list[str]) -> bool:
    if name.number is not None:
        return any(name.number.search(t) for t in texts)
    if not name.needles:
        return True
    return any(n in t for t in texts for n in name.needles)


def check_names(names: Iterable[Name], texts: Iterable[str]) -> NameCheck:
    """Which of names no text holds, compared without case or spacing."""
    names = list(names)
    if not names:
        return NameCheck((), ())
    squashed = [_squash(t) for t in texts if t]
    missing = tuple(n.text for n in names if not _present(n, squashed))
    return NameCheck(tuple(n.text for n in names), missing)


def missing_names_note(check: NameCheck, *, related: bool) -> str | None:
    """A note naming what no passage holds, or None when nothing is missing.

    With related, the note adds that the passages are related text, not the answer.
    """
    if not check.missing:
        return None
    note = "Not in any passage: " + ", ".join(f"`{n}`" for n in check.missing) + "."
    if related:
        note += " The passages are related text, not the answer."
    return note
