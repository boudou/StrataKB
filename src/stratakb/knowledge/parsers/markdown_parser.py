"""Markdown parser that splits documents by heading hierarchy.

Chunks markdown files by H1/H2/H3 sections. Each section keeps its own text;
the parent headings travel on the side as the page's section path. Leading
YAML front matter is not text.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.document import ParsedDocument, ParsedPage
from stratakb.knowledge.structure import frontmatter_end

logger = structlog.get_logger()

# ATX headings: # H1, ## H2, ### H3, indented by up to 3 spaces (4 is indented code)
_HEADING_RE = re.compile(r"^ {0,3}(#{1,3})[ \t]+(.+)$")

# Setext heading underline: '=' makes the line above an H1, two or more '-' an H2
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-{2,})[ \t]*$")

# Lines that cannot be the text of a setext heading: block quotes and list items
_NOT_PARAGRAPH_RE = re.compile(r"^ {0,3}(?:>|[-*+](?:[ \t]|$)|\d+[.)](?:[ \t]|$))")

# Fenced code block delimiters (CommonMark): up to 3 spaces, then 3+ backticks or tildes
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")


def parse_markdown(file_path: str | Path) -> ParsedDocument:
    """Parse a Markdown file, splitting by heading hierarchy.

    Each H1/H2/H3 section becomes a separate page whose text is the section's
    own text. The heading chain is in the page's section_path.

    Args:
        file_path: Path to the Markdown file.

    Returns:
        ParsedDocument with one page per heading section, or no pages for whitespace-only text.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in (".md", ".markdown"):
        return ParsedDocument(source_file=source, error=f"Not a Markdown file: {file_path}")

    try:
        # utf-8-sig: a byte-order mark would hide the front matter
        text = file_path.read_text(encoding="utf-8-sig")
    except Exception as e:
        return ParsedDocument(source_file=source, error=f"Read error: {e}")

    # The front matter is never text, with or without headings
    body = text[frontmatter_end(text):]

    if not body.strip():
        return ParsedDocument(source_file=source, pages=[], total_pages=0)

    sections = _split_by_headings(body)

    if not sections:
        # No headings found — return as single page
        page = ParsedPage(text=body.strip(), page_number=1, source_file=source)
        return ParsedDocument(source_file=source, pages=[page], total_pages=1)

    pages: list[ParsedPage] = []
    for i, section in enumerate(sections):
        section_text = section["text"].strip()
        if section_text:
            pages.append(ParsedPage(
                text=section_text,
                page_number=i + 1,
                source_file=source,
                section_path=section.get("section_path", []),
            ))

    logger.info(
        "markdown_parsed",
        file=source,
        sections=len(pages),
    )

    return ParsedDocument(source_file=source, pages=pages, total_pages=len(pages))


def _split_by_headings(text: str) -> list[dict]:
    """Split markdown text into sections by H1/H2/H3 headings.

    Returns a list of dicts with keys: level, heading, text, section_path.
    The text is the document's own text for the section, heading line included;
    section_path is the heading chain (parent headings, then this heading).
    """
    headings = _find_headings(text)

    if not headings:
        return []

    # Track parent headings for context
    h1_context = ""
    h2_context = ""

    sections: list[dict] = []

    # Handle any preamble before the first heading
    preamble = text[:headings[0][0]].strip()
    if preamble:
        sections.append({
            "level": 0,
            "heading": "Preamble",
            "text": preamble,
        })

    for i, (start, level, heading) in enumerate(headings):
        # Section text is the document's own text, from this heading to the next one
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)

        # Update context
        if level == 1:
            h1_context = heading
            h2_context = ""
        elif level == 2:
            h2_context = heading

        path = []
        if level >= 2 and h1_context:
            path.append(h1_context)
        if level >= 3 and h2_context:
            path.append(h2_context)
        path.append(heading)

        sections.append({
            "level": level,
            "heading": heading,
            "text": text[start:end].strip(),
            "section_path": path,
        })

    return sections


def _find_headings(text: str) -> list[tuple[int, int, str]]:
    """Return (start offset, level, heading text) for each H1/H2/H3 heading, in order.

    One pass over the lines. ATX headings ('# Title') may be indented by up to
    3 spaces. A setext underline ('===' for H1, '---' for H2) makes the line
    directly above it a heading when that line is paragraph text; after a
    blank line it is a thematic break. Of a multi-line paragraph over an
    underline only the last line becomes the heading.

    '#' lines and underlines inside code fences are not headings. A fence opens
    with 3+ backticks or tildes (a backtick fence's info string has no
    backtick, else the line is inline code) and closes on a line holding only
    the same character, at least as many times. An unclosed fence runs to the end.
    """
    headings: list[tuple[int, int, str]] = []
    fence = ""
    # The previous line, when it could be the text of a setext heading
    paragraph: tuple[int, str] | None = None
    start = 0
    for line in text.split("\n"):
        line_start, start = start, start + len(line) + 1
        if fence:
            close = _FENCE_CLOSE_RE.match(line)
            if close and close.group(1)[0] == fence[0] and len(close.group(1)) >= len(fence):
                fence = ""
            continue
        opening = _FENCE_OPEN_RE.match(line)
        if opening and not (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            fence = opening.group(1)
            paragraph = None
            continue

        atx = _HEADING_RE.match(line)
        if atx:
            headings.append((line_start, len(atx.group(1)), atx.group(2).strip()))
            paragraph = None
            continue

        underline = _SETEXT_RE.match(line)
        if underline:
            if paragraph:
                level = 1 if underline.group(1)[0] == "=" else 2
                headings.append((paragraph[0], level, paragraph[1]))
            paragraph = None
            continue

        indent = len(line) - len(line.lstrip(" "))
        is_paragraph = (
            line.strip()
            and indent <= 3
            and not line[indent:].startswith("\t")
            and not _NOT_PARAGRAPH_RE.match(line)
        )
        paragraph = (line_start, line.strip()) if is_paragraph else None

    return headings
