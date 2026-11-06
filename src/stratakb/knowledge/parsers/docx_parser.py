"""DOCX document parser using python-docx.

Extracts text from Word documents with paragraph-level source tracking.
Paragraphs and tables are read in document order; tables are converted to
plain text rows, and paragraphs styled as headings (Word's 'Heading 1' to
'Heading 9', in any language, or a style with an outline level) are
reported as headings.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage
from stratakb.knowledge.structure import Heading

logger = structlog.get_logger()

# The ids of Word's built-in heading styles: 'Heading1' to 'Heading9'. The
# id, unlike the display name ('Heading 1', 'Überschrift 1'), does not follow
# the language of the Word install.
_HEADING_STYLE_ID_RE = re.compile(r"Heading([1-9])")

# How many styles a style may be based on, in a chain, before it is ignored
_MAX_STYLE_CHAIN = 10

# Word's body-text outline level: no heading
_BODY_OUTLINE_LEVEL = 9


def _outline_level(element) -> int | None:
    """The outline level (0 to 9) a paragraph or style element sets, else None."""
    from docx.oxml.ns import qn

    ppr = element.find(qn("w:pPr"))
    level = ppr.find(qn("w:outlineLvl")) if ppr is not None else None
    if level is None:
        return None
    try:
        return int(level.get(qn("w:val")))
    except (TypeError, ValueError):
        return None


def _heading_depth(paragraph) -> int | None:
    """The depth (1 to 9) of a paragraph Word marks as a heading, else None.

    A built-in heading style id ('Heading2') gives the depth. Otherwise the
    outline level the paragraph sets, or else its style or a style that one
    is based on, does: localized and custom heading styles carry one. The
    body-text level is no heading.
    """
    style = paragraph.style
    if style is not None and (m := _HEADING_STYLE_ID_RE.fullmatch(style.style_id or "")):
        return int(m.group(1))
    level = _outline_level(paragraph._p)
    for _ in range(_MAX_STYLE_CHAIN):
        if level is not None or style is None:
            break
        if m := _HEADING_STYLE_ID_RE.fullmatch(style.style_id or ""):
            return int(m.group(1))
        level = _outline_level(style.element)
        style = style.base_style
    if level is None or not 0 <= level < _BODY_OUTLINE_LEVEL:
        return None
    return level + 1


def parse_docx(file_path: str | Path) -> ParsedDocument:
    """Parse a DOCX file and extract text.

    Each non-empty paragraph or table becomes a ParsedPage entry.
    Content is grouped into logical pages of ~3000 chars to match
    the chunking granularity of other parsers.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        ParsedDocument with extracted text.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    if file_path.suffix.lower() != ".docx":
        return ParsedDocument(source_file=source, error=f"Not a DOCX file: {file_path}")

    try:
        from docx import Document
        from docx.table import Table

        doc = Document(str(file_path))
        # Each paragraph and table in document order, with the depth of a
        # paragraph styled as a heading (else None)
        blocks: list[tuple[str, int | None]] = []

        for item in doc.iter_inner_content():
            if isinstance(item, Table):
                rows_text: list[str] = []
                for row in item.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows_text.append(" | ".join(cells))
                if rows_text:
                    blocks.append(("\n".join(rows_text), None))
                continue
            text = item.text.strip()
            if text:
                blocks.append((text, _heading_depth(item)))

        # Group blocks into logical pages (~3000 chars each)
        pages: list[ParsedPage] = []
        current_text = ""
        headings: list[Heading] = []
        page_num = 1

        for block, depth in blocks:
            if current_text and len(current_text) + len(block) > 3000:
                pages.append(ParsedPage(
                    text=current_text,
                    page_number=page_num,
                    source_file=source,
                    headings=headings,
                ))
                page_num += 1
                current_text, headings = "", []
            offset = len(current_text) + 1 if current_text else 0
            if depth is not None:
                headings.append(Heading(offset, depth, " ".join(block.split())))
            current_text = f"{current_text}\n{block}" if current_text else block

        if current_text.strip():
            pages.append(ParsedPage(
                text=current_text,
                page_number=page_num,
                source_file=source,
                headings=headings,
            ))

        logger.info(
            "docx_parsed",
            file=source,
            paragraphs=len(doc.paragraphs),
            tables=len(doc.tables),
            pages=len(pages),
        )

        return ParsedDocument(
            source_file=source,
            pages=pages,
            total_pages=len(pages),
        )

    except Exception as e:
        logger.error("docx_parse_error", file=source, error_type=type(e).__name__)
        return ParsedDocument(source_file=source, error=str(e))
