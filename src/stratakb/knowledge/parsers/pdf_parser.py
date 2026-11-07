"""PDF document parser using pypdf.

Extracts text from PDF files with page-level source tracking.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.document import ParsedDocument, ParsedPage, ParserFlag
from stratakb.knowledge.structure import Heading, leading_label, strip_clause_number

logger = structlog.get_logger()

# Under this many extracted characters, a page with an image counts as scanned
_SCANNED_MIN_CHARS_PER_PAGE = 200

# The most outline entries read, and the deepest level read: a crafted outline
# cannot keep ingestion busy. Past either, the file is flagged truncated.
MAX_OUTLINE_ENTRIES = 10_000
MAX_OUTLINE_DEPTH = 9

# How many lines that start with an outline title are compared for one entry
_MAX_PREFIX_CANDIDATES = 50

# How many lines a wrapped outline title may span on its page
_MAX_TITLE_LINES = 3


def scanned_flag(doc: ParsedDocument) -> ParserFlag | None:
    """Flag a parsed PDF as probably scanned: more than half its pages are in
    doc.scanned_pages. A document with no text pages is not flagged (ingestion
    skips it as no_text).
    """
    if doc.pages and 2 * len(doc.scanned_pages) > doc.total_pages:
        return ParserFlag("probably_scanned")
    return None


def _has_image(page, source: str, page_number: int) -> bool:
    """Whether a page holds an image. Only the count is read: fetching an image
    decodes it, which needs Pillow.

    A page that cannot be inspected counts as having one: without Pillow, an
    inline image raises here. Being wrong costs a flag, not any text.
    """
    try:
        return len(page.images) > 0
    except Exception as e:
        logger.debug(
            "pdf_images_unreadable",
            file=source,
            page=page_number,
            error_type=type(e).__name__,
        )
        return True


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """Parse a PDF file and extract text per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ParsedDocument with page-level text and metadata.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    if not file_path.suffix.lower() == ".pdf":
        return ParsedDocument(source_file=source, error=f"Not a PDF file: {file_path}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages: list[ParsedPage] = []
        scanned_pages: set[int] = set()

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if len(text) < _SCANNED_MIN_CHARS_PER_PAGE and _has_image(page, source, i + 1):
                scanned_pages.add(i + 1)
            if text:
                pages.append(
                    ParsedPage(
                        text=text,
                        page_number=i + 1,
                        source_file=source,
                    )
                )

        outline_entries, left_out = _attach_outline(reader, pages, source)

        logger.info(
            "pdf_parsed",
            file=source,
            total_pages=len(reader.pages),
            pages_with_text=len(pages),
            outline_entries=outline_entries,
        )

        doc = ParsedDocument(
            source_file=source,
            pages=pages,
            total_pages=len(reader.pages),
            scanned_pages=scanned_pages,
        )
        # A probably scanned file is the more useful flag: its text is missing
        doc.flag = scanned_flag(doc) or (
            ParserFlag("truncated", left_out) if left_out else None
        )
        return doc

    except Exception as e:
        logger.error("pdf_parse_error", file=source, error_type=type(e).__name__)
        return ParsedDocument(source_file=source, error=str(e))


_END = object()


def _outline_entries(reader, outline) -> tuple[list[tuple[int, int, str]], int]:
    """(entries, left out): (page index, depth, title) of each outline entry
    read, in outline order, and how many entries the caps left out.

    pypdf gives the outline as a list where a nested list holds the children
    of the entry before it. Entries without a title or a page are skipped.
    Only the first MAX_OUTLINE_ENTRIES entries are read, and none deeper than
    MAX_OUTLINE_DEPTH. The outline is walked with a stack, not recursion.
    """
    entries: list[tuple[int, int, str]] = []
    read = left_out = 0
    stack = [(iter(outline), 1)]
    while stack:
        items, depth = stack[-1]
        item = next(items, _END)
        if item is _END:
            stack.pop()
            continue
        if isinstance(item, list):
            stack.append((iter(item), depth + 1))
            continue
        if depth > MAX_OUTLINE_DEPTH or read >= MAX_OUTLINE_ENTRIES:
            left_out += 1
            continue
        read += 1
        title = " ".join(str(getattr(item, "title", "") or "").split())
        try:
            page_index = reader.get_destination_page_number(item)
        except Exception:
            page_index = None
        if title and page_index is not None and page_index >= 0:
            entries.append((page_index, depth, title))
    return entries, left_out


def _fold(s: str) -> str:
    return " ".join(s.split()).casefold()


class _TitleLines:
    """The lines of one page's text, folded once, to place outline titles on.

    Built once per page, so placing an entry is a few dictionary lookups and
    a bisect, whatever the number of entries on the page.
    """

    def __init__(self, text: str) -> None:
        # folded line, or line less its clause number -> first offset
        self.whole: dict[str, int] = {}
        # line less its clause number -> first offset
        self.bare: dict[str, int] = {}
        # up to _MAX_TITLE_LINES lines joined (a wrapped title), whole or
        # less its clause number -> offset of the first line
        self.wrapped: dict[str, int] = {}
        # clause label -> (offset, line less its number), for the lines that
        # start with a clause number or keyword id
        self.labels: dict[str, list[tuple[int, str]]] = {}
        # (folded line or line less its number, offset), sorted, for the
        # lines that start with a title
        heads: list[tuple[str, int]] = []
        lines: list[tuple[int, str]] = []
        pos = 0
        for raw in text.splitlines(keepends=True):
            line = _fold(raw)
            if line:
                lines.append((pos, raw))
                bare = _fold(strip_clause_number(raw))
                for key in (line, bare):
                    self.whole.setdefault(key, pos)
                    heads.append((key, pos))
                self.bare.setdefault(bare, pos)
                # A bare number on its own line is a page number, not a clause
                if (leading := leading_label(raw)) is not None and not line.isdecimal():
                    found = self.labels.setdefault(leading[0], [])
                    if len(found) < _MAX_PREFIX_CANDIDATES:
                        found.append((pos, bare))
            pos += len(raw)
        for k, (offset, _) in enumerate(lines):
            for n in range(2, _MAX_TITLE_LINES + 1):
                if k + n > len(lines):
                    break
                # One line, so strip_clause_number sees the number and its title
                joined = " ".join(raw.strip() for _, raw in lines[k:k + n])
                for key in (_fold(joined), _fold(strip_clause_number(joined))):
                    self.wrapped.setdefault(key, offset)
        self.heads = sorted(heads)

    def place(self, title: str) -> int | None:
        """Offset of the line that holds title as its heading, or None.

        First choice is a line that is the title, alone or after a clause
        number or keyword id ('7. Payment', 'Article 2 Payment'); then the
        title wrapped over a few lines; then a line that is the title less
        its own clause number, alone or wrapped; then a line with the title's
        clause number whose rest starts the title or is started by it
        ('Article 6' or 'Article 6 Lawfulness' under 'Article 6 Lawfulness of
        processing'); then a line that starts with the title. A title that
        only appears inside a line (a defined term in prose) does not place
        the heading. Clause numbers are read by the structure map's grammar.
        """
        wanted = _fold(title)
        if not wanted:
            return None
        wanted_bare = _fold(strip_clause_number(title))
        # The whole title, on one line or wrapped, comes before a line that
        # only holds the title less its number: 'Article 6' wrapped above
        # 'Lawfulness of processing' is the heading's line
        for found in (self.whole.get(wanted), self.wrapped.get(wanted)):
            if found is not None:
                return found
        if wanted_bare and wanted_bare != wanted:
            for found in (self.bare.get(wanted_bare), self.wrapped.get(wanted_bare)):
                if found is not None:
                    return found
        if (leading := leading_label(title)) is not None:
            for offset, bare in self.labels.get(leading[0], ()):
                if _starts(wanted_bare, bare) or _starts(bare, wanted_bare):
                    return offset
        k = bisect_left(self.heads, (wanted, -1))
        best: int | None = None
        for head, offset in self.heads[k:k + _MAX_PREFIX_CANDIDATES]:
            if not head.startswith(wanted):
                break
            if _starts(head, wanted) and (best is None or offset < best):
                best = offset
        return best


def _starts(text: str, prefix: str) -> bool:
    """True when text starts with prefix, at a word boundary ('' starts all)."""
    return text.startswith(prefix) and not text[len(prefix):len(prefix) + 1].isalnum()


def _attach_outline(reader, pages: list[ParsedPage], source: str) -> tuple[int, int]:
    """Add the PDF outline's entries to pages as headings; return (placed, left out).

    Each entry is placed at the line of its page that holds its title (see
    _TitleLines.place), or, when no line does, at the page's start with
    on_line False; such an entry is dropped when the next heading starts at
    the same offset at its depth or above, as it would cover no text. An
    entry whose page has no text (and was dropped) goes to the start of the
    next page that has text; one past the last such page is left out. A
    broken outline is logged and ignored.
    """
    try:
        entries, left_out = _outline_entries(reader, reader.outline)
    except Exception as e:
        logger.warning("pdf_outline_unreadable", file=source, error_type=type(e).__name__)
        return 0, 0
    if left_out:
        logger.warning("pdf_outline_truncated", file=source, entries_left_out=left_out)
    numbers = [p.page_number for p in pages]
    title_lines: dict[int, _TitleLines] = {}
    for page_index, depth, title in entries:
        k = bisect_left(numbers, page_index + 1)
        if k == len(pages):
            continue
        page = pages[k]
        offset = None
        if page.page_number == page_index + 1:
            if k not in title_lines:
                title_lines[k] = _TitleLines(page.text)
            offset = title_lines[k].place(title)
        page.headings.append(Heading(offset or 0, depth, title, on_line=offset is not None))
    placed = 0
    for page in pages:
        if not page.headings:
            continue
        page.headings.sort(key=lambda h: h.offset)
        page.headings = [
            h for h, after in zip(page.headings, [*page.headings[1:], None], strict=True)
            if h.on_line or after is None or after.offset != h.offset or after.depth > h.depth
        ]
        placed += len(page.headings)
    return placed, left_out
