"""Tests for the structure map and chunking along it (stratakb.knowledge.structure).

The detector and chunking tests are pure; the last ones ingest through the
real pipeline against the database with embeddings mocked.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from stratakb.knowledge.documents import build_document_text
from stratakb.knowledge.ingestion import _chunk_document, _parse_file
from stratakb.knowledge.structure import chunk_spans, detect_sections

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_EMBEDDING = [0.01] * 768


def _document_text(name: str) -> str:
    return build_document_text(_parse_file(FIXTURES / name))[0]


def _starting(text: str, prefix: str) -> int:
    """Offset of the line that starts with prefix."""
    pos = text.index("\n" + prefix) + 1 if not text.startswith(prefix) else 0
    return pos


def _assert_no_chunk_crosses(sections, spans, text):
    for start, end, path in spans:
        assert text[start:end].strip() == text[start:end]
        for s in sections:
            assert not (start < s.char_start < end), (path, s.path)
            assert not (start < s.char_end < end), (path, s.path)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_contract_sections():
    text = _document_text("contract.md")
    sections = detect_sections(text)

    assert [(s.label, s.path, s.depth) for s in sections] == [
        ("1", "1 Definitions", 1),
        ("2", "2 Termination", 1),
        ("2.1", "2 Termination > 2.1 Notice", 2),
        ("2.2", "2 Termination > 2.2 Termination for cause", 2),
        ("annex ii", "Annex II", 1),
    ]
    by_label = {s.label: s for s in sections}
    assert by_label["2.2"].title == "Termination for cause"
    assert by_label["annex ii"].title is None
    # Each span runs from its heading to the next section outside it
    assert by_label["1"].char_start == _starting(text, "1. Definitions")
    assert by_label["1"].char_end == by_label["2"].char_start == _starting(text, "2. Termination")
    assert by_label["2.1"].char_start == _starting(text, "2.1 Notice")
    assert by_label["2.1"].char_end == by_label["2.2"].char_start
    assert by_label["2.2"].char_end == by_label["2"].char_end == _starting(text, "Annex II")
    assert by_label["annex ii"].char_end == len(text)


def test_markdown_sections():
    text = _document_text("test_document.md")
    sections = detect_sections(text)

    assert [s.path for s in sections] == [
        "Customer Order Service",
        "Customer Order Service > Overview",
        "Customer Order Service > Overview > Key Capabilities",
        "Customer Order Service > Overview > Behavior Qualifiers",
        "Customer Order Service > API Operations",
        "Customer Order Service > API Operations > Initiate",
        "Customer Order Service > API Operations > Retrieve",
        "Customer Order Service > Connected Services",
    ]
    overview = sections[1]
    assert overview.label == "overview"
    assert text[overview.char_start:].startswith("## Overview")
    assert overview.char_end == text.index("## API Operations")
    assert sections[0].char_end == len(text)


def test_rst_and_asciidoc_sections():
    rst = (
        "=====\nGuide\n=====\n\nIntro.\n\n"
        "Install\n=======\n\nSteps.\n\n"
        "From source\n-----------\n\nBuild it.\n\n"
        "Usage\n=====\n\nRun it.\n"
    )
    assert [(s.path, s.depth) for s in detect_sections(rst)] == [
        ("Guide", 1),
        ("Guide > Install", 2),
        ("Guide > Install > From source", 3),
        ("Guide > Usage", 2),
    ]
    assert detect_sections(rst)[0].char_start == 0

    adoc = "= Manual\n\nText.\n\n== Setup\n\nMore.\n\n=== Linux\n\nApt.\n\n== Use\n"
    assert [s.path for s in detect_sections(adoc)] == [
        "Manual", "Manual > Setup", "Manual > Setup > Linux", "Manual > Use",
    ]


def test_numbered_heading_forms():
    text = (
        "Article 12 Scope\n\nText.\n\n"
        "1. General\n\nText.\n\n"
        "1.1 Rules\n\nText.\n\n"
        "1.1.1 Details\n\nText.\n\n"
        "Article 13\n\nText.\n\n"
        "4(1) First\n\nText.\n\n"
        "4(2)\n\nText.\n\n"
        "4(2)(a) Sub\n\nText.\n"
    )
    assert [(s.label, s.path) for s in detect_sections(text)] == [
        ("article 12", "Article 12 Scope"),
        ("1", "Article 12 Scope > 1 General"),
        ("1.1", "Article 12 Scope > 1 General > 1.1 Rules"),
        ("1.1.1", "Article 12 Scope > 1 General > 1.1 Rules > 1.1.1 Details"),
        ("article 13", "Article 13"),
        ("4(1)", "Article 13 > 4(1) First"),
        ("4(2)", "Article 13 > 4(2)"),
        ("4(2)(a)", "Article 13 > 4(2) > 4(2)(a) Sub"),
    ]


def test_numbered_list_is_not_a_heading():
    text = _document_text("contract.md")
    labels = [s.label for s in detect_sections(text)]
    assert "3" not in labels
    assert labels.count("1") == 1 and labels.count("2") == 1

    # A loose list inside a clause does not continue its numbering either
    loose = "7.2 Notice\n\nBy one of:\n\n1. hand\n\n2. post\n"
    assert [s.label for s in detect_sections(loose)] == ["7.2"]


def test_wrapped_line_starting_with_a_number_is_not_a_clause():
    text = (
        "7.2 Termination for cause\n\n"
        "Either party may terminate this agreement if the other party fails to pay within\n"
        "30 days of being notified. The Customer shall pay all fees due up to that day within\n"
        "8 weeks after notice.\n\n"
        "7.3 Effect\n\nThe data is returned.\n\n"
        "8. Law\n\nThe law of the seat applies.\n"
    )
    sections = detect_sections(text)
    assert [s.path for s in sections] == [
        "7.2 Termination for cause",
        "7.3 Effect",
        "8 Law",
    ]
    assert sections[0].char_end == text.index("7.3 Effect")

    # A first clause that reads as prose needs a blank line before it
    assert detect_sections("The fee is due within\n30 days. Then it lapses.\n") == []
    # A top-level number that skips ahead does not continue the numbering
    assert [s.label for s in detect_sections("1. Scope\n\nText.\n\n12 Months\n\nMore.\n")] == ["1"]


def test_wrapped_keyword_reference_is_not_a_heading():
    text = (
        "7. Processing\n\n"
        "7.1 Lawful basis\n\n"
        "The Provider processes personal data only on a lawful basis as required by\n"
        "Article 6 GDPR. The Provider keeps records of every processing\n"
        "activity.\n\n"
        "7.2 Records\n\nKept for ten years.\n\n"
        "8. Law\n\nText.\n\n"
        "Article 9 Final provisions\n\nText.\n"
    )
    assert [s.path for s in detect_sections(text)] == [
        "7 Processing",
        "7 Processing > 7.1 Lawful basis",
        "7 Processing > 7.2 Records",
        "8 Law",
        "Article 9 Final provisions",
    ]
    # The rest of a keyword line that reads as prose makes it a sentence
    assert detect_sections("Article 6 GDPR. The Provider keeps records.\n") == []


def test_table_of_contents_does_not_use_up_the_numbering():
    body = (
        "1 Scope\n\nThe scope.\n\n"
        "2 Definitions\n\nThe terms.\n\n"
        "3 Payments\n\nThe payments.\n"
    )
    expected = ["1 Scope", "2 Definitions", "3 Payments"]
    leaders = "Contents\n\n1 Scope ........ 3\n2 Definitions ........ 4\n3 Payments ........ 6\n\n"
    text = leaders + body
    sections = detect_sections(text)
    assert [s.path for s in sections] == expected
    assert sections[0].char_start == text.index("1 Scope\n")

    # Without page numbers, the numbering starts again at the body's first clause
    plain = "Contents\n\n1 Scope\n2 Definitions\n3 Payments\n\n" + body
    sections = detect_sections(plain)
    assert [s.path for s in sections] == expected
    assert sections[0].char_start == plain.index("1 Scope\n\nThe scope")

    # A tab, a wide gap or an ellipsis before the page number also marks an entry
    gaps = "Contents\n\n1 Scope\t3\n2 Definitions    4\n3 Payments … 6\n\n"
    sections = detect_sections(gaps + body)
    assert [s.path for s in sections] == expected
    assert sections[0].char_start == len(gaps)


def test_wide_whitespace_runs_stay_fast():
    # Badly extracted or hostile text: very long runs of spaces and tabs inside
    # and at the end of lines must not make the detector backtrack
    gap = " " * 50_000
    tabs = "\t" * 50_000
    lines = [
        f"a{gap}x",
        f"1.2 Scope{gap}x{gap}",
        f"7(2) Scope{gap}x",
        f"Article 5{gap}x{tabs}",
        f"Article 5 -{gap}",
        f"# Title{gap}x{gap}#",
        f"== Title{gap}x{gap}",
        f"3 Payments{gap}y",
        f"1 Scope {gap}12345",
        f"#{gap}",
    ]
    text = "\n\n".join(lines) + "\n"
    started = time.perf_counter()
    detect_sections(text)
    assert time.perf_counter() - started < 1.0


def test_dates_amounts_and_prose_are_not_headings():
    text = (
        "2026. The parties agree.\n\n"
        "1,000.00 is the fee.\n\n"
        "12\n\n"
        "Section 3 applies to all payments.\n\n"
        "1234.5 Too many digits\n\n"
        "```\n# not a heading\n```\n"
    )
    assert detect_sections(text) == []


def test_extra_pattern_adds_sections():
    text = "Rules\n=====\n\nAT-44 The amount\n\nText.\n\nAT-45 The date\n\nText.\n"
    plain = detect_sections(text)
    assert [s.path for s in plain] == ["Rules"]

    extra = detect_sections(text, [r"(?P<label>AT-\d+)\b"])
    assert [(s.label, s.path) for s in extra] == [
        ("rules", "Rules"),
        ("at-44", "Rules > AT-44 The amount"),
        ("at-45", "Rules > AT-45 The date"),
    ]


# ---------------------------------------------------------------------------
# Chunking along the structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["contract.md", "test_document.md", "test_schema.xsd"])
def test_no_chunk_crosses_a_section(name):
    text = _document_text(name)
    sections = detect_sections(text)
    spans = chunk_spans(sections, text, 120, 30)
    assert spans
    _assert_no_chunk_crosses(sections, spans, text)
    # Every non-blank character is in a chunk
    covered = set()
    for start, end, _ in spans:
        covered.update(range(start, end))
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


def test_long_clause_keeps_path():
    doc = _parse_file(FIXTURES / "contract.md")
    text = build_document_text(doc)[0]
    sections = detect_sections(text)
    clause = next(s for s in sections if s.label == "2.2")
    assert clause.char_end - clause.char_start > 300

    chunks = _chunk_document(doc, layer_name="t", chunk_size=200, chunk_overlap=40)
    pieces = [c for c in chunks if clause.char_start <= c.char_start < clause.char_end]
    assert len(pieces) > 1
    for c in pieces:
        assert c.clause_path == "2 Termination > 2.2 Termination for cause"
        assert c.char_end <= clause.char_end
        assert text[c.char_start:c.char_end] == c.content
    # The text before the first section has no clause path
    preamble = [c for c in chunks if c.char_start < sections[0].char_start]
    assert preamble and all(c.clause_path is None for c in preamble)


def test_fallback_without_structure(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(
        "Plain notes without any heading. " * 20 + "\n\nAnother paragraph of text. " * 10,
        encoding="utf-8",
    )
    doc = _parse_file(f)
    doc.source_file = "notes.txt"
    text = build_document_text(doc)[0]
    assert detect_sections(text) == []

    chunks = _chunk_document(doc, layer_name="t", chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        assert c.clause_path is None
        assert c.char_start is not None
        assert text[c.char_start:c.char_end] == c.content


def test_pages_are_cuts_and_keep_their_numbers():
    from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

    doc = ParsedDocument(
        source_file="c.pdf",
        pages=[
            ParsedPage("1. Scope\n\nThe scope starts here", 1, "c.pdf"),
            ParsedPage("and ends on the next page.\n\n2. Fees\n\nNone.", 2, "c.pdf"),
        ],
        total_pages=2,
    )
    chunks = _chunk_document(doc, layer_name="t", chunk_size=500, chunk_overlap=50)
    assert [(c.metadata["page"], c.clause_path) for c in chunks] == [
        (1, "1 Scope"),
        (2, "1 Scope"),
        (2, "2 Fees"),
    ]
    text = build_document_text(doc)[0]
    assert all(text[c.char_start:c.char_end] == c.content for c in chunks)


# ---------------------------------------------------------------------------
# Stored map (database)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def layer(db_url):
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_struct_{uuid4().hex[:10]}"
    await create_layer(
        db_url,
        name=name,
        display_name="Test Structure",
        chunk_config={"chunk_size": 200, "chunk_overlap": 40},
    )
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


async def _ingest(db_url: str, layer: str, source: Path):
    from stratakb.knowledge.ingestion import ingest_documents

    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda texts, **_: [FAKE_EMBEDDING for _ in texts]
        return await ingest_documents(
            db_url, "fake-key", layer_name=layer, source_path=str(source)
        )


async def _stored(db_url: str, layer: str):
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    sections = await pool.fetch(
        """
        SELECT s.document_id, s.label, s.title, s.path, s.depth, s.char_start, s.char_end
        FROM strata_vectors.document_sections s
        JOIN strata_vectors.documents d ON d.id = s.document_id
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1
        ORDER BY s.document_id, s.char_start
        """,
        layer,
    )
    chunks = await pool.fetch(
        """
        SELECT c.content, c.char_start, c.char_end, c.clause_path, c.document_id, d.text
        FROM strata_vectors.chunks c
        JOIN strata_vectors.documents d ON d.id = c.document_id
        WHERE c.layer_name = $1
        ORDER BY c.char_start
        """,
        layer,
    )
    return sections, chunks


async def test_ingest_stores_the_map_and_clause_paths(db_url, layer, tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    shutil.copy(FIXTURES / "contract.md", src / "contract.md")

    result = await _ingest(db_url, layer, src)
    assert result.docs_failed == 0
    sections, chunks = await _stored(db_url, layer)

    expected = detect_sections(_document_text("contract.md"))
    assert [
        (r["label"], r["title"], r["path"], r["depth"], r["char_start"], r["char_end"])
        for r in sections
    ] == [(s.label, s.title, s.path, s.depth, s.char_start, s.char_end) for s in expected]
    assert {c["clause_path"] for c in chunks} == {
        None,
        "1 Definitions",
        "2 Termination",
        "2 Termination > 2.1 Notice",
        "2 Termination > 2.2 Termination for cause",
        "Annex II",
    }
    for c in chunks:
        assert c["text"][c["char_start"]:c["char_end"]] == c["content"]

    # Ingesting the changed file again stores a new version with its own map;
    # the old version and its sections are pruned
    (src / "contract.md").write_text(
        (FIXTURES / "contract.md").read_text(encoding="utf-8") + "\nAnnex III\n\nMore.\n",
        encoding="utf-8",
    )
    await _ingest(db_url, layer, src)
    sections, chunks = await _stored(db_url, layer)
    assert {r["document_id"] for r in sections} == {c["document_id"] for c in chunks}
    assert [r["label"] for r in sections][-2:] == ["annex ii", "annex iii"]


async def test_search_results_carry_the_clause_path_into_citations(db_url, layer, tmp_path):
    from stratakb.rag.passages import cite
    from stratakb.rag.retriever import search_chunks
    from stratakb.rag.tools import _ask_chunk

    src = tmp_path / "docs"
    src.mkdir()
    shutil.copy(FIXTURES / "contract.md", src / "contract.md")
    await _ingest(db_url, layer, src)
    _, chunks = await _stored(db_url, layer)
    stored = {c["content"]: c["clause_path"] for c in chunks}

    with (
        patch("stratakb.rag.retriever._get_cache", return_value=None),
        patch(
            "stratakb.rag.retriever.embed_texts", new_callable=AsyncMock,
            return_value=[FAKE_EMBEDDING],
        ),
    ):
        results = await search_chunks(db_url, "fake-key", "termination", layers=[layer], top_k=50)

    assert results
    assert {r.clause_path for r in results} >= {"2 Termination > 2.1 Notice"}
    for r in results:
        assert r.clause_path == stored[r.content]
        chunk = _ask_chunk(r, r.score)
        if r.clause_path:
            assert cite(chunk, None)["clause_path"] == r.clause_path


def test_cite_prefers_the_structure_map_path():
    from stratakb.rag.passages import cite

    chunk = {
        "source_file": "a.md",
        "layer": "specs",
        "clause_path": "7 Termination > 7.2 Notice",
        "metadata": {"section_path": ["Master agreement"]},
    }
    assert cite(chunk, None)["clause_path"] == "7 Termination > 7.2 Notice"
    del chunk["clause_path"]
    assert cite(chunk, None)["clause_path"] == "Master agreement"
