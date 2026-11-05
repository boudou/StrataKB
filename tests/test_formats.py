"""Tests for the format table (parsers/registry.FORMATS).

Every row must reach discovery, dispatch, the stored doc type and the ingest
policy, so adding a format means adding one row. No database needed.
"""

from __future__ import annotations

import dataclasses

import pytest

from stratakb.knowledge import policy
from stratakb.knowledge.ingestion import _chunk_document, _discover_files, _parse_file
from stratakb.knowledge.parsers.csv_parser import parse_csv
from stratakb.knowledge.parsers.docx_parser import parse_docx
from stratakb.knowledge.parsers.html_parser import parse_html
from stratakb.knowledge.parsers.json_parser import parse_json
from stratakb.knowledge.parsers.markdown_parser import parse_markdown
from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage, parse_pdf
from stratakb.knowledge.parsers.registry import FORMATS, extensions_for
from stratakb.knowledge.parsers.text_parser import parse_text
from stratakb.knowledge.parsers.xml_parser import parse_xml
from stratakb.knowledge.parsers.yaml_parser import parse_yaml
from stratakb.knowledge.selection import FileEntry

EXTENSIONS = list(FORMATS)


def test_table_matches_previous_values():
    table = {
        ext: (row.parser, row.doc_type, row.binary, row.scanned_check)
        for ext, row in FORMATS.items()
    }
    assert table == {
        ".pdf": (parse_pdf, "pdf", True, True),
        ".docx": (parse_docx, "docx", True, False),
        ".yaml": (parse_yaml, "yaml", False, False),
        ".yml": (parse_yaml, "yaml", False, False),
        ".csv": (parse_csv, "csv", False, False),
        ".xml": (parse_xml, "xml", False, False),
        ".xsd": (parse_xml, "xsd", False, False),
        ".json": (parse_json, "json", False, False),
        ".md": (parse_markdown, "markdown", False, False),
        ".markdown": (parse_markdown, "markdown", False, False),
        ".html": (parse_html, "html", False, False),
        ".htm": (parse_html, "html", False, False),
        ".txt": (parse_text, "text", False, False),
        ".rst": (parse_text, "rst", False, False),
        ".adoc": (parse_text, "adoc", False, False),
    }


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_every_row_is_discovered(tmp_path, ext):
    path = tmp_path / f"sample{ext}"
    path.write_bytes(b"x")
    assert path in _discover_files(tmp_path)
    assert _discover_files(path) == [path]


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_every_row_dispatches_to_its_parser(tmp_path, monkeypatch, ext):
    sentinel = ParsedDocument(source_file="sentinel")
    seen = []

    def spy(file_path):
        seen.append(file_path)
        return sentinel

    monkeypatch.setitem(FORMATS, ext, dataclasses.replace(FORMATS[ext], parser=spy))
    path = tmp_path / f"sample{ext}"
    assert _parse_file(path) is sentinel
    assert seen == [path]


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_every_row_sets_the_doc_type(ext):
    source = f"x{ext}"
    doc = ParsedDocument(
        source_file=source,
        pages=[ParsedPage(text="Some text.", page_number=1, source_file=source)],
        total_pages=1,
    )
    chunks = _chunk_document(doc, layer_name="t", chunk_size=500, chunk_overlap=50)
    assert chunks
    assert {c.doc_type for c in chunks} == {FORMATS[ext].doc_type}


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_every_row_is_accepted_by_the_policy(tmp_path, ext):
    entry = FileEntry(f"docs/sample{ext}", 10, tmp_path / f"sample{ext}")
    assert policy.evaluate(entry, 10_000_000) == policy.Verdict(True, None)


def test_unknown_extension(tmp_path):
    code = tmp_path / "x.py"
    code.write_text("print()")
    assert _discover_files(code) == []
    assert _parse_file(tmp_path / "x.fpr").error == "Unsupported format: .fpr"
    doc = ParsedDocument(
        source_file="x.fpr",
        pages=[ParsedPage(text="Some text.", page_number=1, source_file="x.fpr")],
        total_pages=1,
    )
    chunks = _chunk_document(doc, layer_name="t", chunk_size=500, chunk_overlap=50)
    assert {c.doc_type for c in chunks} == {"fpr"}


def test_extensions_for():
    assert extensions_for(parse_text) == (".txt", ".rst", ".adoc")
    assert extensions_for(parse_xml) == (".xml", ".xsd")
