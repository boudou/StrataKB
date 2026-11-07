"""JSON parser for message samples and structured data.

Converts JSON files into structured text for embedding and search.
Each file becomes a single page with a human-readable representation.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.document import ParsedDocument, ParsedPage

logger = structlog.get_logger()


def parse_json(file_path: str | Path) -> ParsedDocument:
    """Parse a JSON file into searchable text.

    Args:
        file_path: Path to the JSON file.

    Returns:
        ParsedDocument with one page containing the structured text.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    try:
        text = file_path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return ParsedDocument(source_file=source, error=f"JSON parse error: {e}")

    lines = [f"# JSON Document: {source}"]
    _json_to_text(data, lines, depth=0)

    content = "\n".join(lines)
    page = ParsedPage(text=content, page_number=1, source_file=source)

    logger.info("json_parsed", file=source, chars=len(content))

    return ParsedDocument(source_file=source, pages=[page], total_pages=1)


def _json_to_text(data, lines: list[str], depth: int, key: str = "") -> None:
    """Recursively convert JSON data to structured text."""
    indent = "  " * depth
    prefix = f"**{key}:** " if key else ""

    if isinstance(data, dict):
        if key:
            lines.append(f"{indent}{prefix}")
        for k, v in data.items():
            _json_to_text(v, lines, depth + 1, key=k)
    elif isinstance(data, list):
        lines.append(f"{indent}{prefix}[{len(data)} items]")
        for i, item in enumerate(data):
            _json_to_text(item, lines, depth + 1, key=f"[{i}]")
    else:
        lines.append(f"{indent}{prefix}{data}")
