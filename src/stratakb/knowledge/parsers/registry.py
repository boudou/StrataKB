"""Extension → parser table, shared by ingestion and the ingest policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from stratakb.knowledge.parsers.csv_parser import parse_csv
from stratakb.knowledge.parsers.docx_parser import parse_docx
from stratakb.knowledge.parsers.html_parser import parse_html
from stratakb.knowledge.parsers.json_parser import parse_json
from stratakb.knowledge.parsers.markdown_parser import parse_markdown
from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, parse_pdf
from stratakb.knowledge.parsers.text_parser import parse_text
from stratakb.knowledge.parsers.xml_parser import parse_xml
from stratakb.knowledge.parsers.yaml_parser import parse_yaml

# Supported file extensions and their parsers
SUPPORTED_PARSERS: dict[str, Callable[[str | Path], ParsedDocument]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".yaml": parse_yaml,
    ".yml": parse_yaml,
    ".csv": parse_csv,
    ".xml": parse_xml,
    ".xsd": parse_xml,
    ".json": parse_json,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
    ".txt": parse_text,
    ".rst": parse_text,
    ".adoc": parse_text,
}
