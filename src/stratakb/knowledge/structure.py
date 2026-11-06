"""Structure map of a document text, and chunking along it.

detect_sections finds headings and numbered clauses in the stored document
text with generic, line-based detectors that need no configuration:

- markup headings: Markdown ATX ('## Title') and setext, reStructuredText
  underlines (and overlines), AsciiDoc ('== Title');
- numbered headings on a line of their own or starting a clause: '4.3.2',
  '7.', '7(2)(b)', and keyword headings such as 'Article 12' or 'Annex II'.
  A numbered line must continue the numbering, and a numbered or keyword
  line must follow a blank line, a heading or the end of a sentence, so a
  wrapped line of prose is not a heading. Table-of-contents entries are
  skipped, for every pattern.

Leading YAML front matter is never read for headings. In Markdown text, a
run of '1.' '2.' lines with no deeper clause in it is an ordered list, not
clauses, and gives no sections.

Each section spans from its heading to the next section that is not inside
it. chunk_spans then cuts the text at every section start, so a chunk never
crosses a section boundary, and splits a long section into pieces that all
carry the section's path. A heading with no text of its own before its
first subsection is not cut off on its own: it joins that subsection's
chunk, which carries the subsection's path.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import structlog

from stratakb.knowledge.parsers.markdown_parser import _frontmatter_end

logger = structlog.get_logger()

# Joins the section names of a path
PATH_SEPARATOR = " > "

# The heading patterns take their text greedily up to its last non-space
# character, and '++' keeps them from splitting a run of spaces: a lazy text
# before a trailing space run backtracks quadratically on a long run of spaces.

# Markdown ATX heading: up to 3 spaces of indent, 1-6 '#', the text (with its
# optional closing '#'s, which _atx_text removes)
_ATX_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]++(.*\S)[ \t]*$")

# AsciiDoc section title: '=' marks the document title, '==' level 1 and so on
_ASCIIDOC_RE = re.compile(r"^(={1,6})[ \t]++(\S(?:.*\S)?)[ \t]*$")

# A setext or reStructuredText adornment line: one punctuation character, 3+
# times. '*' and '_' are left out: in Markdown they only draw a rule.
_ADORNMENT_RE = re.compile(r"^([=\-~^\"'+#:.])\1{2,}[ \t]*$")

# Lines that cannot be the title above an adornment: lists, quotes, tables
_NOT_TITLE_RE = re.compile(r"^ {0,3}(?:[>|]|[-*+](?:[ \t]|$)|\d+[.)](?:[ \t]|$))")

# Fenced code block delimiters, as in the Markdown parser
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")

# Dotted numbers: at most 3 levels of at most 3 digits ('2026.' and
# '1,000.00' do not match), then the text, which starts with a letter
_DOTTED_RE = re.compile(
    r"^ {0,3}(\d{1,3}(?:\.\d{1,3}){0,2})(\.?)(?:[ \t]++([^\W\d_](?:.*\S)?))?[ \t]*$"
)

# A one-level list item ('1.' or '1)'); next to another one it is not a clause
_LIST_ITEM_RE = re.compile(r"^ {0,3}\d{1,3}[.)][ \t]")

# An ordered-list marker at a line start, with or without text after it
_LIST_MARKER_RE = re.compile(r"^ {0,3}\d{1,3}[.)](?:[ \t]|$)")

# Statute-style numbers: '7(2)' and '7(2)(b)'
_PAREN_RE = re.compile(
    r"^ {0,3}(\d{1,3})\((\d{1,3})\)(?:\(([a-z])\))?(?:[ \t]++([^\W\d_](?:.*\S)?))?[ \t]*$"
)

# Keyword headings, with their rank: a lower rank encloses a higher one
_KEYWORD_RANKS = {
    "part": 1,
    "annex": 1,
    "appendix": 1,
    "schedule": 1,
    "chapter": 2,
    "section": 3,
    "article": 4,
    "clause": 4,
}
_KEYWORD_RE = re.compile(
    r"^ {0,3}((?i:" + "|".join(_KEYWORD_RANKS) + r"))[ \t]+"
    r"(\d{1,3}(?:\.\d{1,3}){0,2}|[IVXLC]{1,7}|[A-Z])\b\.?"
    r"(?:[ \t]*+[-–—:.]?[ \t]*+(.*\S)?)?[ \t]*$"
)

# The end of a sentence or clause: a heading may follow such a line directly
_SENTENCE_END_RE = re.compile(r"[.:;!?)\]\"'”]$")

# The longest heading text a numbered or keyword line may carry as its title;
# a longer rest of the line is the clause's own text
_MAX_TITLE_CHARS = 60

# The longest line taken as a heading above an adornment or after a keyword
_MAX_HEADING_CHARS = 120

# How much of a line's start an extra pattern is matched against
_MAX_PATTERN_CHARS = 200

# Section kinds: markup headings close every numbered or keyword section
_MARKUP, _KEYWORD, _NUMBERED = "markup", "keyword", "numbered"


@dataclass(frozen=True)
class Section:
    """One heading or clause and the span of the text it covers.

    text[char_start:char_end] runs from the heading to the next section that
    is not inside this one, so it includes the subsections. body_start is
    where the section's own text starts, just past its heading line(s); it is
    char_start when the heading line is itself the clause's text, and None
    when unknown.
    """

    label: str
    title: str | None
    path: str
    depth: int
    char_start: int
    char_end: int
    body_start: int | None = None


@dataclass
class _Candidate:
    kind: str
    level: tuple[int, int, int]
    label: str
    title: str | None
    name: str
    start: int
    body_start: int
    # A dotted or statute clause number (not an extra pattern), and whether
    # its line starts with an ordered-list marker ('1.', '2)')
    clause: bool = False
    list_marker: bool = False
    # Levels added because the clause's numbering sits inside an
    # extra-pattern section
    nest: int = 0


@dataclass(frozen=True)
class _PatternSection:
    """The extra-pattern section the scanner is in.

    under is the clause numbering that was open when the first of these
    sections started (None when none was), and depth the sections' level
    below the enclosing heading: one below under.
    """

    under: tuple[int, ...] | None
    depth: int


def _lines(text: str) -> list[tuple[int, str]]:
    """Each line of text with its offset, without the line break."""
    lines: list[tuple[int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        lines.append((pos, line.rstrip("\r\n")))
        pos += len(line)
    return lines


def normalise_label(label: str) -> str:
    """A section label as stored and looked up: lower case, single spaces."""
    return " ".join(label.lower().split())


def _heading_label(heading: str) -> tuple[str, str | None]:
    """(label, title) of a markup heading's text.

    The label is the number or keyword id the text starts with, normalised,
    and the title the text after it; without a number, the label is the
    whole text in lower case and the title the text.
    """
    heading = heading.strip()
    if m := _KEYWORD_RE.match(heading):
        return normalise_label(f"{m.group(1)} {m.group(2)}"), m.group(3) or None
    if m := _PAREN_RE.match(heading):
        label = m.group(1) + f"({m.group(2)})" + (f"({m.group(3)})" if m.group(3) else "")
        return label, m.group(4) or None
    if m := _DOTTED_RE.match(heading):
        return m.group(1), m.group(3) or None
    return normalise_label(heading), heading


def _short_title(rest: str | None) -> str | None:
    """rest as a heading title, or None when it reads as the clause's own text."""
    if not rest:
        return None
    rest = rest.strip()
    title = rest.removesuffix(".")
    if len(title) > _MAX_TITLE_CHARS or re.search(r"[.;:,!?]\s", title) or title.endswith(
        (";", ":", ",")
    ):
        return None
    return title or None


def _numbered(line: str) -> tuple[tuple[int, ...], str, str | None, bool] | None:
    """(number parts, label, title, prose) when line starts with a clause number.

    prose is True when the rest of the line reads as the clause's own text
    rather than a title.
    """
    if m := _PAREN_RE.match(line):
        parts = [int(m.group(1)), int(m.group(2))]
        label = f"{m.group(1)}({m.group(2)})"
        if m.group(3):
            parts.append(ord(m.group(3)) - ord("a") + 1)
            label += f"({m.group(3)})"
        title = _short_title(m.group(4))
        return tuple(parts), label, title, bool(m.group(4)) and title is None
    if m := _DOTTED_RE.match(line):
        if m.group(3) is None and not m.group(2) and "." not in m.group(1):
            # A bare number on its own line is a page number or an amount
            return None
        parts = tuple(int(p) for p in m.group(1).split("."))
        title = _short_title(m.group(3))
        return parts, m.group(1), title, bool(m.group(3)) and title is None
    return None


def _continues(number: tuple[int, ...], current: tuple[int, ...] | None) -> bool:
    """True when number is the next clause after current.

    The next clause is a first subclause (7.2.1 after 7.2), the next one at a
    lower level (7.3 after 7.2.4) or the next top number (8 after 7.2). So
    the '1.' of a list inside 7.2, and a wrapped line that starts with '30',
    are not clauses. Any number starts a context (current is None at the
    start and after a markup or keyword heading): an excerpt may begin at
    4.3.2.
    """
    if current is None:
        return True
    k = len(number)
    if k == 1:
        return number[0] == current[0] + 1
    if k > len(current) + 1 or number[: k - 1] != current[: k - 1]:
        return False
    expected = current[k - 1] + 1 if k <= len(current) else 1
    return number[k - 1] == expected


class _Numbering:
    """The clause numbering the scanner is in, between two markup or keyword headings.

    The open numbering run ends at clause number _current; it starts at
    found[_run_start] with number _run_first, and _run_body is True once body
    text came after its first clause. _section is the extra-pattern section
    the scanner is in, if any; _own_run is True when the open run started
    inside it, so belongs to it alone and nests below it.
    """

    def __init__(self, found: list[_Candidate]) -> None:
        self._found = found
        self.reset()

    def reset(self) -> None:
        """Forget the numbering: a markup or keyword heading starts a new one."""
        self._current: tuple[int, ...] | None = None
        self._run_start, self._run_first, self._run_body = 0, (), False
        self._section: _PatternSection | None = None
        self._own_run = False

    @property
    def nest(self) -> int:
        """Levels the open run's clauses sit below the numbering around them."""
        return self._section.depth if self._own_run and self._section else 0

    def saw_text(self) -> None:
        """Note a line of body text: the open run is not a bare table of contents."""
        self._run_body = True

    def enter_pattern_section(self) -> int:
        """Start an extra-pattern section; return its level below the enclosing heading.

        One level below the numbering open when the first of these sections
        started: under clause 4.1 an attribute is level 3, so it closes the
        attribute before it and the next clause (4.2) closes it. A numbered
        list inside the attribute before does not count: the outer numbering
        is restored, so 4.2 still continues 4.1.
        """
        under = self._section.under if self._section is not None else self._current
        self._section = _PatternSection(under, len(under) + 1 if under is not None else 1)
        self._current, self._own_run = under, False
        return self._section.depth

    def place(self, number: tuple[int, ...], *, breaks: bool, can_start: bool) -> int | None:
        """The level of the clause numbered number, or None when the line is no clause.

        breaks is True when the line follows a break, which a clause needs;
        can_start is False when the line may continue the open run but not
        start one (a first clause whose line reads as prose needs a blank
        line before it).
        """
        context, ends_section = self._context(number)
        if not breaks or (context is None and not can_start):
            self._run_body = True
            return None
        if not _continues(number, context):
            if context is not None and not self._run_body and number == self._run_first:
                # The run so far was a table of contents without page numbers:
                # the numbering starts again here
                del self._found[self._run_start:]
                context = None
            else:
                self._run_body = True
                return None
        if ends_section:
            self._section, self._own_run = None, False
        if context is None:
            self._run_start, self._run_first, self._run_body = len(self._found), number, False
            self._own_run = self._section is not None
        self._current = number
        return self.nest + len(number)

    def _context(self, number: tuple[int, ...]) -> tuple[tuple[int, ...] | None, bool]:
        """(the numbering number continues, whether it ends the extra-pattern section).

        Outside an extra-pattern section that numbering is the open run.
        Inside one, the section's own run comes first (step 5 after step 4
        of a list, even under clause 4), then the outer numbering (4.2 after
        an attribute of 4.1, which ends the section); any other number starts
        a run of the section's own when none is open.
        """
        section = self._section
        if section is None:
            return self._current, False
        if self._own_run and _continues(number, self._current):
            return self._current, False
        if section.under is not None and _continues(number, section.under):
            return section.under, True
        return (self._current if self._own_run else None), False


def _is_list_item(lines: list[tuple[int, str]], i: int) -> bool:
    """True when line i is a one-level '1.' or '1)' item next to another one."""
    if not _LIST_ITEM_RE.match(lines[i][1]):
        return False
    return any(
        0 <= j < len(lines) and _LIST_ITEM_RE.match(lines[j][1]) for j in (i - 1, i + 1)
    )


def _blank(line: str) -> bool:
    return not line.strip()


def _atx_text(text: str) -> str:
    """An ATX heading's text without its optional closing '#' sequence."""
    head = text.rstrip("#")
    if head != text and head[-1:] in (" ", "\t") and head.strip(" \t"):
        return head.rstrip(" \t")
    return text


def _is_toc_entry(line: str) -> bool:
    """True when line ends in dot leaders or a gap, then a page number.

    A table-of-contents entry: '3 Payments ..... 6', '3 Payments … 6',
    '3 Payments<tab>6'. Done with string methods, which stay linear on long
    runs of spaces.
    """
    s = line.rstrip(" \t")
    end = len(s)
    while end > 0 and len(s) - end < 5 and s[end - 1].isdecimal():
        end -= 1
    if not 1 <= len(s) - end <= 4:
        return False
    before = s[:end].rstrip(" \t")
    gap = s[len(before):end]
    return len(gap) >= 2 or "\t" in gap or before.endswith(("..", "…"))


def _pattern_match(patterns: Sequence[re.Pattern[str]], line: str) -> re.Match[str] | None:
    """The first extra pattern's match at the start of line, past its indent.

    Only the first _MAX_PATTERN_CHARS characters are matched: an id starts
    the line, and a pattern cannot then take long on a very long line.
    """
    if not patterns:
        return None
    head = line.lstrip(" \t")[:_MAX_PATTERN_CHARS]
    for pattern in patterns:
        if m := pattern.match(head):
            return m
    return None


def _scan(text: str, extra_patterns: Sequence[re.Pattern[str]]) -> list[_Candidate]:
    """Every heading line of text, in order."""
    # Leading YAML front matter holds no headings: its lines are left out, so
    # no look-back reaches into it and the line after it counts as following
    # a blank line. Offsets stay those of text.
    body_from = _frontmatter_end(text)
    lines = [(o, line) for o, line in _lines(text) if o >= body_from]
    found: list[_Candidate] = []
    adornments: list[tuple[str, bool]] = []  # reStructuredText styles in order of use
    markup_level = 0
    keyword_rank = 0
    numbering = _Numbering(found)
    fence: str | None = None
    skip_next = False
    heading_line = -1  # index of the last line of the last heading

    def add(kind: str, level: tuple[int, int, int], name: str, start: int,
            label: str, title: str | None, *, own_text: bool = False,
            clause: bool = False) -> None:
        nonlocal heading_line
        heading_line = i + 1 if skip_next else i
        # The section's own text starts on the line after its heading, or on
        # the heading line when that line is the clause's text
        if own_text:
            body_start = start
        elif heading_line + 1 < len(lines):
            body_start = lines[heading_line + 1][0]
        else:
            body_start = len(text)
        found.append(_Candidate(
            kind, level, label, title, name.strip(), start, body_start,
            clause=clause, list_marker=clause and _LIST_MARKER_RE.match(line) is not None,
            nest=numbering.nest if clause else 0,
        ))

    for i, (offset, line) in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if fence is not None:
            if line.lstrip(" ").startswith(fence):
                fence = None
            continue
        if m := _FENCE_RE.match(line):
            fence = m.group(1)[0] * len(m.group(1))
            continue
        if _blank(line):
            continue

        previous_blank = i == 0 or _blank(lines[i - 1][1])
        next_line = lines[i + 1][1] if i + 1 < len(lines) else ""

        heading: str | None = None
        level = 0
        start = offset
        if m := _ATX_RE.match(line):
            heading, level = _atx_text(m.group(2)), len(m.group(1))
        elif m := _ASCIIDOC_RE.match(line):
            heading, level = m.group(2), len(m.group(1))
        elif (
            _ADORNMENT_RE.match(next_line)
            and not _NOT_TITLE_RE.match(line)
            and len(line.strip()) <= _MAX_HEADING_CHARS
            and not _ADORNMENT_RE.match(line)
        ):
            over = (
                i >= 1
                and _ADORNMENT_RE.match(lines[i - 1][1]) is not None
                and lines[i - 1][1].strip()[0] == next_line.strip()[0]
            )
            if over or previous_blank:
                style = (next_line.strip()[0], over)
                if style not in adornments:
                    adornments.append(style)
                heading, level = line, adornments.index(style) + 1
                if over:
                    start = lines[i - 1][0]
                skip_next = True
        if heading is not None:
            markup_level, keyword_rank = level, 0
            numbering.reset()
            add(_MARKUP, (level, 0, 0), heading, start, *_heading_label(heading))
            continue

        # A wrapped line of prose can start with a number or 'Article 6': a
        # numbered or keyword heading starts after a blank line, another
        # heading or the end of a sentence
        after_break = previous_blank or heading_line == i - 1
        breaks = after_break or _SENTENCE_END_RE.search(lines[i - 1][1].rstrip()) is not None

        keyword = _KEYWORD_RE.match(line)
        if (
            keyword or _numbered(line) or _pattern_match(extra_patterns, line)
        ) and _is_toc_entry(line):
            # A table-of-contents entry ('3 Payments ..... 6') is not the clause
            continue

        if keyword and keyword.group(1)[0].isupper():
            rest = (keyword.group(3) or "").strip()
            title = _short_title(rest)
            # 'Section 3 applies to ...' is a sentence, and 'Article 6 GDPR.
            # The Provider ...' the clause's text, not a heading
            if breaks and not rest[:1].islower() and (title or not rest):
                keyword_rank = _KEYWORD_RANKS[keyword.group(1).lower()]
                numbering.reset()
                name = f"{keyword.group(1)} {keyword.group(2)}"
                label = normalise_label(name)
                add(_KEYWORD, (markup_level, keyword_rank, 0),
                    f"{name} {title}" if title else name, offset,
                    label=label, title=title)
                continue

        if m := _pattern_match(extra_patterns, line):
            label = m.groupdict().get("label") or m.group(0)
            depth = numbering.enter_pattern_section()
            add(_NUMBERED, (markup_level, keyword_rank, depth), line, offset,
                label=normalise_label(label), title=None, own_text=True)
            continue

        numbered = _numbered(line)
        if numbered is None or _is_list_item(lines, i):
            numbering.saw_text()
            continue
        parts, label, title, prose = numbered
        depth = numbering.place(parts, breaks=breaks, can_start=not prose or after_break)
        if depth is None:
            continue
        name = f"{label} {title}" if title else label
        add(_NUMBERED, (markup_level, keyword_rank, depth), name, offset,
            label=label, title=title, own_text=prose, clause=True)

    return found


def _drop_ordered_lists(candidates: list[_Candidate]) -> list[_Candidate]:
    """candidates without the runs of clauses that read as a Markdown ordered list.

    A run is the dotted or statute clauses of one numbering: those with the
    same nesting between two markup or keyword headings. An extra-pattern
    section does not end the numbering around it ('4.' '4.1' 'AT-01' '5.' is
    one run), but it ends a run nested inside the section before it, and so
    does a clause of an outer numbering. A run whose clauses are all one
    level and all start with a list marker ('1.', '2)') is an ordered list,
    not clauses.
    """
    dropped: set[int] = set()
    runs: dict[int, list[int]] = {}  # nesting -> indexes in candidates

    def flush(deeper_than: int) -> None:
        for nest in [n for n in runs if n > deeper_than]:
            run = runs.pop(nest)
            if all(
                candidates[k].level[2] - nest == 1 and candidates[k].list_marker for k in run
            ):
                dropped.update(run)

    for k, cand in enumerate(candidates):
        if cand.clause:
            flush(cand.nest)
            runs.setdefault(cand.nest, []).append(k)
        elif cand.kind == _NUMBERED:
            flush(0)
        else:
            flush(-1)
    flush(-1)
    return [cand for k, cand in enumerate(candidates) if k not in dropped]


def _closes(new: _Candidate, open_: _Candidate) -> bool:
    """True when new starts outside open_, which then ends where new starts."""
    if new.kind == _MARKUP and open_.kind != _MARKUP:
        return True
    # A keyword heading closes the numbered clauses that no keyword encloses
    if new.kind == _KEYWORD and open_.kind == _NUMBERED and open_.level[1] == 0:
        return True
    return new.level <= open_.level


def detect_sections(
    text: str,
    extra_patterns: Iterable[str | re.Pattern[str]] = (),
    *,
    markdown: bool = False,
) -> list[Section]:
    """The structure map of text: its sections in order, with paths and spans.

    extra_patterns are regular expressions matched at line starts, past any
    indent (a named group 'label' gives the label); each match starts a
    section one level below the innermost open heading or clause, which the
    next match or the next clause at that level or above ends. markdown is
    True for Markdown text, where a run of one-level '1.' lines is an ordered
    list and gives no sections.
    Returns [] when text has no structure.
    """
    patterns = [re.compile(p) if isinstance(p, str) else p for p in extra_patterns]
    candidates = _scan(text, patterns)
    if markdown:
        candidates = _drop_ordered_lists(candidates)
    sections: list[Section] = []
    # (candidate, path, depth, index in sections) of the sections still open
    stack: list[tuple[_Candidate, str, int, int]] = []
    ends: list[int] = []
    for cand in candidates:
        while stack and _closes(cand, stack[-1][0]):
            ends[stack.pop()[3]] = cand.start
        parent_path = stack[-1][1] if stack else ""
        path = f"{parent_path}{PATH_SEPARATOR}{cand.name}" if parent_path else cand.name
        depth = len(stack) + 1
        sections.append(
            Section(cand.label, cand.title, path, depth, cand.start, len(text), cand.body_start)
        )
        ends.append(len(text))
        stack.append((cand, path, depth, len(sections) - 1))
    return [
        Section(s.label, s.title, s.path, s.depth, s.char_start, end, s.body_start)
        for s, end in zip(sections, ends, strict=True)
    ]


def split_with_offsets(splitter, text: str, chunk_overlap: int) -> list[tuple[str, int | None]]:
    """Split text with splitter; each piece with its offset in text, or None.

    Overlapping pieces repeat text, and a text can repeat itself: a piece
    starts after the previous one's start and at most chunk_overlap
    characters before its end, so the search starts there.
    """
    pieces: list[tuple[str, int | None]] = []
    prev_start, prev_end = -1, 0
    for piece in splitter.split_text(text):
        pos = text.find(piece, max(prev_start + 1, prev_end - chunk_overlap))
        if pos < 0:
            pos = text.find(piece, prev_start + 1)
        if pos < 0:
            pieces.append((piece, None))
            continue
        prev_start, prev_end = pos, pos + len(piece)
        pieces.append((piece, pos))
    return pieces


def make_splitter(chunk_size: int, chunk_overlap: int):
    """The size-based splitter used inside a section and for unstructured text."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _boundary_between(boundaries: Sequence[int], start: int, end: int) -> bool:
    """True when a boundary b of the sorted boundaries has start < b <= end."""
    k = bisect_right(boundaries, start)
    return k < len(boundaries) and boundaries[k] <= end


def chunk_spans(
    sections: Sequence[Section],
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    boundaries: Iterable[int] = (),
) -> list[tuple[int, int, str]]:
    """(char_start, char_end, path) of each chunk of text, along its sections.

    The text is cut at every section start and end and at each extra boundary
    (such as page starts); each piece between two cuts is split by size.
    path is the innermost section's path, or '' before the first section.
    No chunk crosses a cut, and text[char_start:char_end] is the chunk.

    A section with no text of its own before its first subsection (a bare
    heading such as '7. Termination' directly followed by '7.1') is not cut
    off: the subsection's start is no cut, and the chunk carries the
    subsection's path; the heading is prepended to the subsection's first
    chunk, which may then exceed chunk_size by the heading's length. Folding
    never crosses an extra boundary.
    """
    extra = sorted(b for b in boundaries if 0 <= b <= len(text))
    by_start = sorted(sections, key=lambda s: (s.char_start, s.depth))
    # Where each section counts as open: its start, or the start of the bare
    # heading(s) folded into it
    opens = [s.char_start for s in by_start]
    folded: set[int] = set()  # indexes in by_start whose start is no cut
    for k in range(len(by_start) - 1):
        s, c = by_start[k], by_start[k + 1]
        if (
            s.char_start < c.char_start < s.char_end
            and s.body_start is not None
            and not text[s.body_start:c.char_start].strip()
            and not _boundary_between(extra, s.char_start, c.char_start)
        ):
            folded.add(k + 1)
            opens[k + 1] = opens[k]
    # Where the text of a piece that opens with folded heading(s) starts: the
    # innermost folded subsection's start (by_start is in order, so the last
    # one written wins)
    child_start = {opens[k]: by_start[k].char_start for k in sorted(folded)}

    cuts = {0, len(text)}
    cuts.update(extra)
    for k, s in enumerate(by_start):
        cuts.add(s.char_end)
        if k not in folded:
            cuts.add(s.char_start)
    ordered = sorted(cuts)

    splitter = make_splitter(chunk_size, chunk_overlap)
    spans: list[tuple[int, int, str]] = []
    open_: list[Section] = []
    next_section = 0
    for a, b in zip(ordered, ordered[1:], strict=False):
        while next_section < len(by_start) and opens[next_section] <= a:
            open_.append(by_start[next_section])
            next_section += 1
        open_ = [s for s in open_ if s.char_end > a]
        path = max(open_, key=lambda s: s.depth).path if open_ else ""
        # Split from the subsection's start, so the size splitter cannot cut
        # the bare heading off again; the first chunk then starts at the heading
        split_from = child_start.get(a, a)
        piece_text = text[split_from:b]
        if not piece_text.strip():
            continue
        folded_heading = split_from != a
        for piece, pos in split_with_offsets(splitter, piece_text, chunk_overlap):
            if pos is None:
                logger.warning("chunk_offsets_not_found", char_start=split_from)
                continue
            start = a if folded_heading else split_from + pos
            folded_heading = False
            spans.append((start, split_from + pos + len(piece), path))
    return spans
