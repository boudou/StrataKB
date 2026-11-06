"""The one table of supported file formats, shared by ingestion and the ingest
policy: per extension, its parser, the stored doc type, and whether the file is
binary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Format:
    """How StrataKB handles one file extension.

    binary: read as bytes by its parser, so never decoded to check for empty text.
    """

    parser: Callable[[str | Path], ParsedDocument]
    doc_type: str
    binary: bool = False


# Supported formats. Adding a format means adding one row here.
FORMATS: dict[str, Format] = {
    ".pdf": Format(parse_pdf, "pdf", binary=True),
    ".docx": Format(parse_docx, "docx", binary=True),
    ".yaml": Format(parse_yaml, "yaml"),
    ".yml": Format(parse_yaml, "yaml"),
    ".csv": Format(parse_csv, "csv"),
    ".xml": Format(parse_xml, "xml"),
    ".xsd": Format(parse_xml, "xsd"),
    ".json": Format(parse_json, "json"),
    ".md": Format(parse_markdown, "markdown"),
    ".markdown": Format(parse_markdown, "markdown"),
    ".html": Format(parse_html, "html"),
    ".htm": Format(parse_html, "html"),
    ".txt": Format(parse_text, "text"),
    ".rst": Format(parse_text, "rst"),
    ".adoc": Format(parse_text, "adoc"),
}


def extensions_for(parser: Callable[[str | Path], ParsedDocument]) -> tuple[str, ...]:
    """The extensions whose row uses this parser, in table order."""
    return tuple(ext for ext, row in FORMATS.items() if row.parser is parser)
