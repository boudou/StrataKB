"""PDF document parser using pypdf.

Extracts text from PDF files with page-level source tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger()

# Below this many extracted characters per page on average, a PDF is probably scanned
_SCANNED_MIN_CHARS_PER_PAGE = 200

FlagReason = Literal["probably_scanned", "truncated"]


@dataclass(frozen=True)
class ParserFlag:
    """Why a parsed file is stored but flagged, and how many items a limit left
    out (None when nothing was counted)."""

    reason: FlagReason
    count: int | None = None


@dataclass
class ParsedPage:
    """A single page extracted from a PDF."""

    text: str
    page_number: int
    source_file: str
    # Heading chain of a section, outermost first (Markdown and HTML); empty for other formats
    section_path: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Result of parsing a PDF document."""

    source_file: str
    pages: list[ParsedPage] = field(default_factory=list)
    total_pages: int = 0
    error: str | None = None
    # Set by a parser that stored only part of a file or found it probably scanned; None otherwise
    flag: ParserFlag | None = None


def scanned_flag(doc: ParsedDocument) -> ParserFlag | None:
    """Flag a parsed PDF as probably scanned: under half its pages have text, or
    under _SCANNED_MIN_CHARS_PER_PAGE characters per page on average.

    parse_pdf keeps only pages with text, so len(doc.pages) counts those and
    total_pages counts all pages. A document with no text pages is not flagged
    (ingestion skips it as no_text).
    """
    if doc.total_pages <= 0 or not doc.pages:
        return None
    if (
        2 * len(doc.pages) < doc.total_pages
        or sum(len(p.text) for p in doc.pages) / doc.total_pages < _SCANNED_MIN_CHARS_PER_PAGE
    ):
        return ParserFlag("probably_scanned")
    return None


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

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(
                    ParsedPage(
                        text=text,
                        page_number=i + 1,
                        source_file=source,
                    )
                )

        logger.info(
            "pdf_parsed",
            file=source,
            total_pages=len(reader.pages),
            pages_with_text=len(pages),
        )

        doc = ParsedDocument(
            source_file=source,
            pages=pages,
            total_pages=len(reader.pages),
        )
        doc.flag = scanned_flag(doc)
        return doc

    except Exception as e:
        logger.error("pdf_parse_error", file=source, error_type=type(e).__name__)
        return ParsedDocument(source_file=source, error=str(e))
