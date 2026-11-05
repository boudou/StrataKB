"""CSV parser for tabular reference data (code lists, catalogues, glossaries).

Converts each row into a structured text chunk, labelling every value with
its column header.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()

# Maximum rows to parse (safety limit); rows past it are counted, not read,
# and ingestion reports the file as flagged ("truncated")
MAX_ROWS = 10_000


def parse_csv(file_path: str | Path) -> ParsedDocument:
    """Parse a CSV file into searchable text chunks.

    Each row becomes a page, formatted with the column headers as labels.
    Rows past MAX_ROWS are not read; their count is in rows_left_out.

    Args:
        file_path: Path to the CSV file.

    Returns:
        ParsedDocument with one page per row/group, or no pages when the file has no data rows.
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
                return ParsedDocument(source_file=source, pages=[], total_pages=0)

            headers = list(reader.fieldnames)
            rows = list(itertools.islice(reader, MAX_ROWS))
            # Count the rest without keeping it
            rows_left_out = sum(1 for _ in reader)

    except Exception as e:
        return ParsedDocument(source_file=source, error=f"CSV parse error: {e}")

    if rows_left_out:
        logger.warning(
            "csv_truncated",
            file=source,
            rows_read=len(rows),
            rows_left_out=rows_left_out,
        )

    if not rows:
        return ParsedDocument(source_file=source, pages=[], total_pages=0)

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
        rows_left_out=rows_left_out,
        pages=len(pages),
    )

    return ParsedDocument(
        source_file=source,
        pages=pages,
        total_pages=len(pages),
        rows_left_out=rows_left_out,
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
