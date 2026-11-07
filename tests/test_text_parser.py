"""Tests for the plain-text parser (.txt, .rst, .adoc) and the parser registry.

The parser tests need no database; TestIngestTextFormats needs it (embeddings mocked).
"""

from __future__ import annotations

import pytest

from stratakb.knowledge.parsers.registry import FORMATS, extensions_for
from stratakb.knowledge.parsers.text_parser import parse_text
from tests.helpers import file_outcomes, ingest


def test_text_with_content(tmp_path):
    text = "\n  First paragraph.\n\nSecond paragraph.\n  "
    f = tmp_path / "notes.txt"
    f.write_bytes(text.encode("utf-8"))
    doc = parse_text(f)
    assert doc.error is None
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert doc.pages[0].text == text  # unchanged, not stripped
    assert doc.pages[0].source_file == "notes.txt"
    assert doc.source_file == "notes.txt"


def test_whitespace_only(tmp_path):
    f = tmp_path / "blank.txt"
    f.write_bytes(b"  \n\t\n")
    doc = parse_text(f)
    assert doc.pages == []
    assert doc.total_pages == 0
    assert doc.error is None


def test_bom_removed(tmp_path):
    f = tmp_path / "bom.txt"
    f.write_bytes(b"\xef\xbb\xbfHello")
    doc = parse_text(f)
    assert doc.pages[0].text == "Hello"


def test_invalid_utf8_replaced(tmp_path):
    f = tmp_path / "latin1.txt"
    f.write_bytes(b"caf\xe9 au lait")
    doc = parse_text(f)
    assert doc.error is None
    assert doc.pages[0].text == "caf� au lait"


def test_missing_file_is_error(tmp_path):
    doc = parse_text(tmp_path / "missing.txt")
    assert doc.pages == []
    assert doc.error is not None
    assert "File not found" in doc.error


def test_wrong_extension_is_error(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\n", encoding="utf-8")
    doc = parse_text(f)
    assert doc.error is not None
    assert "Not a text file" in doc.error


@pytest.mark.parametrize("name", ["guide.rst", "manual.adoc", "NOTES.TXT"])
def test_rst_and_adoc_parse(tmp_path, name):
    f = tmp_path / name
    f.write_text("Title\n=====\n\nSome text.\n", encoding="utf-8")
    doc = parse_text(f)
    assert doc.error is None
    assert doc.total_pages == 1
    assert doc.pages[0].text == "Title\n=====\n\nSome text.\n"


def test_registry_maps_text_formats():
    for ext in (".txt", ".rst", ".adoc"):
        assert FORMATS[ext].parser is parse_text
    assert extensions_for(parse_text) == (".txt", ".rst", ".adoc")


def test_ingestion_uses_the_registry():
    from stratakb.knowledge import ingestion

    assert ingestion.FORMATS is FORMATS


@pytest.mark.parametrize(("name", "doc_type"), [
    ("notes.txt", "text"),
    ("guide.rst", "rst"),
    ("manual.adoc", "adoc"),
])
def test_chunk_doc_types(tmp_path, name, doc_type):
    from stratakb.knowledge.ingestion import _chunk_document

    f = tmp_path / name
    f.write_text("Some plain text to chunk.\n", encoding="utf-8")
    chunks = _chunk_document(parse_text(f), layer_name="docs", chunk_size=500, chunk_overlap=50)
    assert chunks
    assert {c.doc_type for c in chunks} == {doc_type}


# --- Database-backed ingest ---


class TestIngestTextFormats:
    async def test_ingest_text_formats(self, db_url, layer, tmp_path):
        from stratakb.core.database import get_pool

        (tmp_path / "notes.txt").write_text("Plain text notes.\n", encoding="utf-8")
        (tmp_path / "guide.rst").write_text("Guide\n=====\n\nSome guidance.\n", encoding="utf-8")
        (tmp_path / "manual.adoc").write_text("= Manual\n\nSome manual text.\n", encoding="utf-8")
        (tmp_path / "blank.txt").write_text("  \n\t\n", encoding="utf-8")

        result = await ingest(db_url, layer, tmp_path)

        outcomes = file_outcomes(result)
        assert outcomes["notes.txt"] == ("added", None)
        assert outcomes["guide.rst"] == ("added", None)
        assert outcomes["manual.adoc"] == ("added", None)
        assert outcomes["blank.txt"] == ("skipped", "no_text")

        pool = await get_pool(db_url)
        rows = await pool.fetch(
            "SELECT DISTINCT source_file, doc_type FROM strata_vectors.chunks"
            " WHERE layer_name = $1",
            layer,
        )
        assert {r["source_file"]: r["doc_type"] for r in rows} == {
            "notes.txt": "text",
            "guide.rst": "rst",
            "manual.adoc": "adoc",
        }
