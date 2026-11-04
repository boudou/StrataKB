"""Markdown parser that splits documents by heading hierarchy.

Chunks markdown files by H1/H2/H3 sections, preserving heading context
so that each chunk is self-contained and searchable.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()

# Matches ATX headings: # H1, ## H2, ### H3
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def parse_markdown(file_path: str | Path) -> ParsedDocument:
    """Parse a Markdown file, splitting by heading hierarchy.

    Each H1/H2/H3 section becomes a separate page. The heading context
    (parent headings) is preserved in each page's text.

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
            ))

    logger.info(
        "markdown_parsed",
        file=source,
        sections=len(pages),
    )

    return ParsedDocument(source_file=source, pages=pages, total_pages=len(pages))


def _split_by_headings(text: str) -> list[dict]:
    """Split markdown text into sections by H1/H2/H3 headings.

    Returns a list of dicts with keys: level, heading, text.
    Parent heading context is prepended to each section.
    """
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return []

    # Track parent headings for context
    h1_context = ""
    h2_context = ""

    sections: list[dict] = []

    # Handle any preamble before the first heading
    preamble = text[:matches[0].start()].strip()
    if preamble:
        # Strip YAML frontmatter
        preamble_clean = _strip_frontmatter(preamble)
        if preamble_clean.strip():
            sections.append({
                "level": 0,
                "heading": "Preamble",
                "text": preamble_clean.strip(),
            })

    for i, match in enumerate(matches):
        level = len(match.group(1))  # Number of # chars
        heading = match.group(2).strip()

        # Determine section content (from after this heading to the next heading)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Update context
        if level == 1:
            h1_context = heading
            h2_context = ""
        elif level == 2:
            h2_context = heading

        # Build section text with context
        lines = []
        if level >= 2 and h1_context:
            lines.append(f"# {h1_context}")
        if level >= 3 and h2_context:
            lines.append(f"## {h2_context}")
        lines.append(f"{'#' * level} {heading}")
        if body:
            lines.append(body)

        sections.append({
            "level": level,
            "heading": heading,
            "text": "\n\n".join(lines),
        })

    return sections


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- ... ---) from the beginning of text."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:]
    return text
