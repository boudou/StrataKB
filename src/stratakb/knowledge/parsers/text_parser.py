"""Plain-text parser for .txt, reStructuredText and AsciiDoc files.

The whole decoded text becomes one page, unchanged (not stripped), so the
page starts at offset 0 of the extracted text.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()


def parse_text(file_path: str | Path) -> ParsedDocument:
    """Parse a plain-text file into a single page.

    Reads UTF-8 (a leading BOM is removed); invalid bytes are replaced with
    U+FFFD instead of raising. Whitespace-only text gives no pages and no error.

    Args:
        file_path: Path to the .txt, .rst or .adoc file.

    Returns:
        ParsedDocument with one page, or no pages for whitespace-only text.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    # Imported here: the registry imports this module
    from stratakb.knowledge.parsers.registry import extensions_for

    ext = file_path.suffix.lower()
    if ext not in extensions_for(parse_text):
        return ParsedDocument(source_file=source, error=f"Not a text file: {file_path}")

    try:
        text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        return ParsedDocument(source_file=source, error=f"Read error: {e}")

    if not text.strip():
        return ParsedDocument(source_file=source, pages=[], total_pages=0)

    page = ParsedPage(text=text, page_number=1, source_file=source)
    logger.info("text_parsed", file=source, chars=len(text))
    return ParsedDocument(source_file=source, pages=[page], total_pages=1)
