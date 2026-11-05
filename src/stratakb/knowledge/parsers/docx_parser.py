"""DOCX document parser using python-docx.

Extracts text from Word documents with paragraph-level source tracking.
Tables are converted to plain text rows.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()


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

        doc = Document(str(file_path))
        blocks: list[str] = []

        # Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                blocks.append(text)

        # Extract tables as text
        for table in doc.tables:
            rows_text: list[str] = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows_text.append(" | ".join(cells))
            if rows_text:
                blocks.append("\n".join(rows_text))

        # Group blocks into logical pages (~3000 chars each)
        pages: list[ParsedPage] = []
        current_text = ""
        page_num = 1

        for block in blocks:
            if current_text and len(current_text) + len(block) > 3000:
                pages.append(ParsedPage(
                    text=current_text,
                    page_number=page_num,
                    source_file=source,
                ))
                page_num += 1
                current_text = block
            else:
                current_text = f"{current_text}\n{block}" if current_text else block

        if current_text.strip():
            pages.append(ParsedPage(
                text=current_text,
                page_number=page_num,
                source_file=source,
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
