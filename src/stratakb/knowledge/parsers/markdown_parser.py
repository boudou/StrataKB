"""Markdown parser that splits documents by heading hierarchy.

Chunks markdown files by H1/H2/H3 sections. Each section keeps its own text;
the parent headings travel on the side as the page's section path.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()

# Matches ATX headings: # H1, ## H2, ### H3 (the space after # stays on the line)
_HEADING_RE = re.compile(r"^(#{1,3})[ \t]+(.+)$", re.MULTILINE)

# Fenced code block delimiters (CommonMark): up to 3 spaces, then 3+ backticks or tildes
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")

# Leading YAML front matter: a first line holding only '---' (not followed by a
# blank line, which would make it a thematic break), up to a line holding only '---' or '...'
_FRONTMATTER_RE = re.compile(
    r"\A\s*---[ \t]*\n(?![ \t]*\n).*?^(?:---|\.\.\.)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


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
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return ParsedDocument(source_file=source, error=f"Read error: {e}")

    if not text.strip():
        return ParsedDocument(source_file=source, pages=[], total_pages=0)

    sections = _split_by_headings(text)

    if not sections:
        # No headings found — return as single page
        page = ParsedPage(text=text.strip(), page_number=1, source_file=source)
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
    # '#' lines inside code fences or the front matter are not headings
    fm_end = _frontmatter_end(text)
    skipped = _non_heading_ranges(text, fm_end)
    matches = [
        m for m in _HEADING_RE.finditer(text)
        if not any(start <= m.start() < end for start, end in skipped)
    ]

    if not matches:
        return []

    # Track parent headings for context
    h1_context = ""
    h2_context = ""

    sections: list[dict] = []

    # Handle any preamble before the first heading (the front matter is not text)
    preamble = text[fm_end:matches[0].start()].strip()
    if preamble:
        sections.append({
            "level": 0,
            "heading": "Preamble",
            "text": preamble,
        })

    for i, match in enumerate(matches):
        level = len(match.group(1))  # Number of # chars
        heading = match.group(2).strip()

        # Section text is the document's own text, from this heading to the next one
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

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
            "text": text[match.start():end].strip(),
            "section_path": path,
        })

    return sections


def _frontmatter_end(text: str) -> int:
    """Return the offset just past the leading YAML front matter, or 0 when there is none."""
    match = _FRONTMATTER_RE.match(text)
    return match.end() if match else 0


def _non_heading_ranges(text: str, fm_end: int) -> list[tuple[int, int]]:
    """Return (start, end) offsets of the leading front matter and of fenced code blocks.

    fm_end is _frontmatter_end(text). A fence opens with 3+ backticks or tildes
    (a backtick fence's info string has no backtick, else the line is inline
    code) and closes on a line holding only the same character, at least as
    many times. An unclosed fence runs to the end.
    """
    ranges: list[tuple[int, int]] = []
    pos = fm_end
    if fm_end:
        ranges.append((0, fm_end))

    fence = ""
    fence_start = 0
    start = 0
    for line in text.split("\n"):
        line_start, start = start, start + len(line) + 1
        if line_start < pos:
            continue
        if fence:
            close = _FENCE_CLOSE_RE.match(line)
            if close and close.group(1)[0] == fence[0] and len(close.group(1)) >= len(fence):
                ranges.append((fence_start, start))
                fence = ""
            continue
        opening = _FENCE_OPEN_RE.match(line)
        if opening and not (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            fence = opening.group(1)
            fence_start = line_start

    if fence:
        ranges.append((fence_start, len(text)))
    return ranges
