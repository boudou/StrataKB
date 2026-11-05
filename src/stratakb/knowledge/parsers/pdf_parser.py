"""PDF document parser using pypdf.

Extracts text from PDF files with page-level source tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger()


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
    # Rows a row-limited parser (CSV) did not read; 0 for other formats
    rows_left_out: int = 0


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

        return ParsedDocument(
            source_file=source,
            pages=pages,
            total_pages=len(reader.pages),
        )

    except Exception as e:
        logger.error("pdf_parse_error", file=source, error=str(e))
        return ParsedDocument(source_file=source, error=str(e))
