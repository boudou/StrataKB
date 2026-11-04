"""CSV parser for tabular reference data (code lists, catalogues, glossaries).

Converts each row into a structured text chunk, labelling every value with
its column header.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()

# Maximum rows to parse (safety limit)
MAX_ROWS = 10_000


def parse_csv(file_path: str | Path) -> ParsedDocument:
    """Parse a CSV file into searchable text chunks.

    Each row becomes a page, formatted with the column headers as labels.

    Args:
        file_path: Path to the CSV file.

    Returns:
        ParsedDocument with one page per row/group.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        return ParsedDocument(source_file=source, error=f"Not a CSV file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return ParsedDocument(source_file=source, error="CSV has no header row")

            headers = list(reader.fieldnames)
            rows = []
            for i, row in enumerate(reader):
                if i >= MAX_ROWS:
                    break
                rows.append(row)

    except Exception as e:
        return ParsedDocument(source_file=source, error=f"CSV parse error: {e}")

    if not rows:
        return ParsedDocument(source_file=source, error="CSV file is empty (no data rows)")

    pages: list[ParsedPage] = []

    for page_num, row in enumerate(rows, start=1):
        text = _row_to_text(row, headers, source)
        if text.strip():
            pages.append(ParsedPage(
                text=text,
                page_number=page_num,
                source_file=source,
            ))

    logger.info(
        "csv_parsed",
        file=source,
        columns=len(headers),
        rows=len(rows),
        pages=len(pages),
    )

    return ParsedDocument(
        source_file=source,
        pages=pages,
        total_pages=len(pages),
    )


def _row_to_text(row: dict, headers: list[str], source: str) -> str:
    """Convert a CSV row into structured text with column labels."""
    lines = []

    # Use the first common naming column as the chunk title
    title_candidates = [
        "Title",
        "Name",
        "Term",
        "Code",
        "Operation",
    ]
    title = ""
    for col in title_candidates:
        if col in row and row[col] and row[col].strip():
            title = row[col].strip()
            break

    if title:
        lines.append(f"# {title}")

    for header in headers:
        value = row.get(header, "").strip()
        if value:
            lines.append(f"**{header}:** {value}")

    return "\n".join(lines)
