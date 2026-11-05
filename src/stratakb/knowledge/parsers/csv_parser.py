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
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers is None:
                return ParsedDocument(source_file=source, pages=[], total_pages=0)

            # Blank lines are not rows: skip them before the row limit
            data = (row for row in reader if row)
            rows = list(itertools.islice(data, MAX_ROWS))
            # Count the rest without keeping it
            rows_left_out = sum(1 for _ in data)

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

    labels = _column_labels(headers)
    pages: list[ParsedPage] = []

    for page_num, row in enumerate(rows, start=1):
        text = _row_to_text(row, labels)
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


def _column_labels(headers: list[str]) -> list[str]:
    """Label each column by its header name.

    An empty name becomes "Column N" (N is the 1-based position), and the
    second and later uses of a name get a number: "Name (2)", "Name (3)".
    """
    labels = []
    seen: dict[str, int] = {}
    for position, header in enumerate(headers, start=1):
        name = header.strip()
        if not name:
            labels.append(f"Column {position}")
            continue
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return labels


def _row_to_text(values: list[str], labels: list[str]) -> str:
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
        if col in labels:
            index = labels.index(col)
            if index < len(values) and values[index].strip():
                title = values[index].strip()
                break

    if title:
        lines.append(f"# {title}")

    # A short row simply has fewer values; a long row's extra values past the
    # header are labelled by position
    for position, raw in enumerate(values, start=1):
        value = raw.strip()
        if value:
            label = labels[position - 1] if position <= len(labels) else f"Column {position}"
            lines.append(f"**{label}:** {value}")

    return "\n".join(lines)
