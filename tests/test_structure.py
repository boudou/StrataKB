"""Tests for the structure map and chunking along it (stratakb.knowledge.structure).

The detector and chunking tests are pure; the last ones ingest through the
real pipeline against the database with embeddings mocked.
"""

from __future__ import annotations

import contextlib
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from stratakb.knowledge.documents import build_document_text
from stratakb.knowledge.ingestion import _chunk_document, _map_document, _parse_file
from stratakb.knowledge.structure import (
    PATH_SEPARATOR,
    chunk_spans,
    detect_sections,
    frontmatter_end,
)
from tests.helpers import FAKE_EMBEDDING, call_tool, ingest

FIXTURES = Path(__file__).parent / "fixtures"


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
            # A bare heading folds into its first subsection: the chunk may
            # hold the start of its own section or of one of its ancestors
            if s.path == path or path.startswith(s.path + PATH_SEPARATOR):
                continue
            assert not (start < s.char_start < end), (path, s.path)
        for s in sections:
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


def test_extra_pattern_nests_under_numbered_clauses():
    # A numbered rulebook: the attributes sit inside clause 4.1, and the
    # numbering goes on after them
    text = (
        "4 Business requirements\n\n"
        "4.1 Attributes\n\n"
        "AT-01 The IBAN of the Originator\n\nText.\n\n"
        "AT-02 The name of the Originator\n\nText.\n\n"
        "4.2 Data sets\n\nText.\n\n"
        "5 Security\n\nText.\n"
    )
    plain = detect_sections(text)
    assert [s.path for s in plain] == [
        "4 Business requirements",
        "4 Business requirements > 4.1 Attributes",
        "4 Business requirements > 4.2 Data sets",
        "5 Security",
    ]

    sections = detect_sections(text, [r"^(?P<label>AT-R?\d{1,3})\b"])
    assert [(s.label, s.path) for s in sections] == [
        ("4", "4 Business requirements"),
        ("4.1", "4 Business requirements > 4.1 Attributes"),
        ("at-01", "4 Business requirements > 4.1 Attributes > AT-01 The IBAN of the Originator"),
        ("at-02", "4 Business requirements > 4.1 Attributes > AT-02 The name of the Originator"),
        ("4.2", "4 Business requirements > 4.2 Data sets"),
        ("5", "5 Security"),
    ]
    at01, at02, s42 = sections[2], sections[3], sections[4]
    assert at01.char_end == at02.char_start
    assert at02.char_end == s42.char_start
    assert sections[1].char_end == s42.char_start


_RUNBOOKS = (
    "# Runbooks\n\n"
    "RB-01 Restart the queue\n\n"
    "1. Stop the consumer.\n\n"
    "2. Restart the broker.\n\n"
    "RB-02 Fail over the database\n\n"
    "Promote the replica.\n\n"
    "RB-03 Rotate keys\n\n"
    "Run the script.\n"
)
_RB = [r"^(?P<label>RB-\d+)\b"]


def test_extra_pattern_section_with_loose_list_markdown():
    # The loose list inside RB-01 is not a clause, and it does not push the
    # next runbooks below RB-01
    sections = detect_sections(_RUNBOOKS, _RB, markdown=True)
    assert [s.path for s in sections] == [
        "Runbooks",
        "Runbooks > RB-01 Restart the queue",
        "Runbooks > RB-02 Fail over the database",
        "Runbooks > RB-03 Rotate keys",
    ]
    rb01, rb02 = sections[1], sections[2]
    assert rb01.char_end == rb02.char_start


def test_extra_pattern_section_with_numbered_list_plain_text():
    # In plain text the list items are clauses of RB-01, and RB-02 and RB-03
    # stay siblings of RB-01
    sections = detect_sections(_RUNBOOKS, _RB)
    rb = [s.path for s in sections if s.label.startswith("rb-")]
    assert rb == [
        "Runbooks > RB-01 Restart the queue",
        "Runbooks > RB-02 Fail over the database",
        "Runbooks > RB-03 Rotate keys",
    ]
    for s in sections:
        if s.label in ("1", "2"):
            assert s.path.startswith("Runbooks > RB-01 Restart the queue > ")
    rb01 = next(s for s in sections if s.label == "rb-01")
    rb02 = next(s for s in sections if s.label == "rb-02")
    assert rb01.char_end == rb02.char_start


def test_extra_pattern_list_then_outer_clause_continues():
    # A list inside an attribute does not stop the outer numbering: 4.2
    # still continues 4.1
    text = (
        "4 Business requirements\n\n"
        "4.1 Attributes\n\n"
        "AT-01 The IBAN of the Originator\n\n"
        "1. First rule.\n\n2. Second rule.\n\n"
        "AT-02 The name of the Originator\n\nText.\n\n"
        "4.2 Data sets\n\nText.\n"
    )
    at = [r"^(?P<label>AT-R?\d{1,3})\b"]
    expected = [
        "4 Business requirements",
        "4 Business requirements > 4.1 Attributes",
        "4 Business requirements > 4.1 Attributes > AT-01 The IBAN of the Originator",
        "4 Business requirements > 4.1 Attributes > AT-02 The name of the Originator",
        "4 Business requirements > 4.2 Data sets",
    ]
    sections = detect_sections(text, at, markdown=True)
    assert [s.path for s in sections] == expected
    # In plain text the rules are clauses of AT-01
    plain = detect_sections(text, at)
    assert [s.path for s in plain if s.label not in ("1", "2")] == expected
    at01 = expected[2]
    assert [s.path for s in plain if s.label in ("1", "2")] == [
        f"{at01} > 1 First rule",
        f"{at01} > 2 Second rule",
    ]


_AT = [r"^(?P<label>AT-R?\d{1,3})\b"]


@pytest.mark.parametrize(
    ("after", "expected_after"),
    [
        ("4.2 Data sets", "Rulebook > 4 Business > 4.2 Data sets"),
        ("5 Security", "Rulebook > 5 Security"),
    ],
)
def test_extra_pattern_list_in_last_attribute_then_outer_clause_markdown(after, expected_after):
    # The list in the last attribute is an ordered list, even when the next
    # line is a clause of the outer numbering
    text = (
        "# Rulebook\n\n4 Business\n\n4.1 Attributes\n\n"
        "AT-01 Name\n\nText.\n\n"
        "AT-02 Amount\n\n1. Rule one.\n\n2. Rule two.\n\n"
        f"{after}\n\nText.\n"
    )
    sections = detect_sections(text, _AT, markdown=True)
    assert [s.path for s in sections] == [
        "Rulebook",
        "Rulebook > 4 Business",
        "Rulebook > 4 Business > 4.1 Attributes",
        "Rulebook > 4 Business > 4.1 Attributes > AT-01 Name",
        "Rulebook > 4 Business > 4.1 Attributes > AT-02 Amount",
        expected_after,
    ]


@pytest.mark.parametrize("markdown", [True, False])
@pytest.mark.parametrize(
    ("clause", "steps", "attr_path", "next_clause"),
    [
        # Five steps under 4.1: step 5 would also continue clause 4
        ("4 Payments\n\n4.1 Credit", 5,
         "4 Payments > 4.1 Credit > AT-02 BIC", ("4.2 Debit", "4 Payments > 4.2 Debit")),
        # Three steps under clause 2: step 3 would also continue clause 2
        ("2 Payments", 3,
         "2 Payments > AT-02 BIC", ("3 Debit", "3 Debit")),
    ],
)
def test_extra_pattern_list_number_continuing_outer_clause_stays_in_section(
    markdown, clause, steps, attr_path, next_clause
):
    # A list step whose number happens to continue the outer numbering is
    # still a step of the section's list
    body = "".join(f"{n}. Step {n}.\n\n" for n in range(1, steps + 1))
    text = (
        f"{clause}\n\nAT-01 The IBAN\n\n{body}"
        f"AT-02 BIC\n\nText.\n\n{next_clause[0]}\n\nText.\n"
    )
    sections = detect_sections(text, _AT, markdown=markdown)
    by_label = {s.label: s.path for s in sections}
    assert by_label["at-02"] == attr_path
    assert by_label[next_clause[0].split()[0]] == next_clause[1]
    at01 = by_label["at-01"]
    steps_found = [s.path for s in sections if s.path.startswith(f"{at01} > ")]
    if markdown:
        assert steps_found == []
    else:
        assert steps_found == [f"{at01} > {n} Step {n}" for n in range(1, steps + 1)]


@pytest.mark.parametrize(
    "attribute_body",
    [
        "text\n",
        # A loose list in the last attribute, then the outer clause
        "1. a.\n\n2. b.\n",
    ],
)
def test_extra_pattern_does_not_split_outer_numbering_markdown(attribute_body):
    # 4. / 4.1 / 5. is one numbering with a subclause, not an ordered list:
    # the attribute between 4.1 and 5. must not cut it in two, or '5.'
    # alone reads as a list and clause 5 disappears
    text = (
        "# R\n\n4. Business\n\n4.1 Attributes\n\n"
        f"AT-01 X\n\n{attribute_body}\n"
        "5. Security\n\ntext\n"
    )
    plain = detect_sections(text.replace("AT-01 X\n\n", ""), markdown=True)
    assert "R > 5 Security" in [s.path for s in plain]

    sections = detect_sections(text, _AT, markdown=True)
    assert [s.path for s in sections] == [
        "R",
        "R > 4 Business",
        "R > 4 Business > 4.1 Attributes",
        "R > 4 Business > 4.1 Attributes > AT-01 X",
        "R > 5 Security",
    ]
    at01, security = sections[3], sections[4]
    assert at01.char_end == security.char_start


def test_extra_pattern_matches_an_indented_line():
    # PDF text often indents a table's first column
    text = "Attributes\n==========\n\n   AT-44 The amount\n\nText.\n\n\tAT-45 The date\n\nText.\n"
    sections = detect_sections(text, _AT)
    assert [(s.label, s.path) for s in sections] == [
        ("attributes", "Attributes"),
        ("at-44", "Attributes > AT-44 The amount"),
        ("at-45", "Attributes > AT-45 The date"),
    ]
    assert sections[1].char_start == text.index("   AT-44")


def test_extra_pattern_reads_only_a_line_start():
    # An id far into a very long line is not read: the line is not a section
    line = "x" * 300 + " AT-44"
    assert detect_sections(f"{line}\n", [r"^.*?(?P<label>AT-\d+)"]) == []
    assert [s.label for s in detect_sections("x AT-44\n", [r"^.*?(?P<label>AT-\d+)"])] == [
        "at-44"
    ]


def test_markdown_loose_list_is_not_a_clause():
    guide = (
        "# Guide\n\nIntro.\n\n1. Install the tool\n\nRun it once.\n\n"
        "2. Configure it\n\nEdit the file.\n"
    )
    lone = "Some intro.\n\n1. Install the tool\n\nMore prose.\n"
    assert [s.path for s in detect_sections(guide, markdown=True)] == ["Guide"]
    assert detect_sections(lone, markdown=True) == []
    # Outside Markdown the same lines stay clauses
    assert [s.path for s in detect_sections(guide)] == [
        "Guide", "Guide > 1 Install the tool", "Guide > 2 Configure it",
    ]
    assert [s.path for s in detect_sections(lone)] == ["1 Install the tool"]


def test_markdown_run_with_subclauses_is_kept():
    sections = _map_document(_parse_file(FIXTURES / "contract.md")).sections
    assert [s.path for s in sections] == [
        "1 Definitions",
        "2 Termination",
        "2 Termination > 2.1 Notice",
        "2 Termination > 2.2 Termination for cause",
        "Annex II",
    ]
    text = "1. Scope\n\nText.\n\n1.1 Rules\n\nText.\n\n2. Fees\n\n2.1 Due\n\nText.\n"
    assert [s.label for s in detect_sections(text, markdown=True)] == ["1", "1.1", "2", "2.1"]


def test_front_matter_is_not_a_heading():
    assert detect_sections("---\ntitle: Foo\n---\n\nBody text here.\n") == []
    sections = detect_sections("---\ntitle: Foo\n---\nIntro\n=====\n\nText.\n")
    assert [s.path for s in sections] == ["Intro"]


def test_front_matter_key_with_spaces_is_not_a_heading():
    t = "---\ntitle: x\ndate created: 2024\n---\n# H\n\nbody\n"
    assert [s.path for s in detect_sections(t)] == ["H"]
    assert [s.path for s in detect_sections(t, markdown=True)] == ["H"]


def test_underline_right_after_front_matter():
    # The closing '---' is not an overline: the section starts at 'Intro'
    t = "---\ntitle: Foo\n---\nIntro\n-----\n\nText.\n"
    sections = detect_sections(t)
    assert [(s.path, s.char_start) for s in sections] == [("Intro", t.index("Intro"))]


@pytest.mark.parametrize(
    ("line", "is_front_matter"),
    [("a" + " " * 100_000 + "b: x", True), ("a" + " " * 100_000 + "b", False)],
    ids=["key", "not-a-key"],
)
def test_run_of_blanks_in_a_front_matter_line_is_read_in_linear_time(line, is_front_matter):
    text = f"---\n{line}\n---\n# H\n"
    began = time.perf_counter()

    end = frontmatter_end(text)

    # The old key pattern needed over ten seconds for each of these
    assert time.perf_counter() - began < 1.0
    assert (end > 0) is is_front_matter


def test_toc_entry_with_extra_pattern_is_skipped():
    text = "Contents\n\nAT-01 Payments ..... 6\nAT-02 Refunds ..... 8\n\nAT-01 Payments\n\nText.\n"
    sections = detect_sections(text, [r"(?P<label>AT-\d+)\b"])
    assert [(s.label, s.char_start) for s in sections] == [
        ("at-01", text.rindex("AT-01 Payments")),
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
    from stratakb.knowledge.parsers.document import ParsedDocument, ParsedPage

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


_TERMINATION = (
    "6. Fees\n\nPay on time.\n\n7. Termination\n\n"
    "7.1 The Provider may end this.\n\n7.2 The Client may end this.\n"
)


def test_heading_only_section_folds_into_first_child():
    t = _TERMINATION
    sections = detect_sections(t)
    assert [(s.path, s.char_start, s.char_end) for s in sections] == [
        ("6 Fees", 0, t.index("7. ")),
        ("7 Termination", t.index("7. "), len(t)),
        ("7 Termination > 7.1 The Provider may end this", t.index("7.1"), t.index("7.2")),
        ("7 Termination > 7.2 The Client may end this", t.index("7.2"), len(t)),
    ]
    spans = chunk_spans(sections, t, 500, 50)
    assert [(t[a:b], path) for a, b, path in spans] == [
        ("6. Fees\n\nPay on time.", "6 Fees"),
        (
            "7. Termination\n\n7.1 The Provider may end this.",
            "7 Termination > 7.1 The Provider may end this",
        ),
        ("7.2 The Client may end this.", "7 Termination > 7.2 The Client may end this"),
    ]
    _assert_no_chunk_crosses(sections, spans, t)


def test_heading_folds_into_a_first_child_longer_than_chunk_size():
    first = "7.1 The Provider may end this agreement on notice. " * 23
    t = f"7. Termination\n\n{first.strip()}\n\n7.2 Done.\n"
    assert len(first) > 1000
    sections = detect_sections(t)
    spans = chunk_spans(sections, t, 1000, 200)
    assert all(t[a:b] != "7. Termination" for a, b, _ in spans)
    assert spans[0][0] == 0
    assert t[spans[0][0]:spans[0][1]].startswith("7. Termination\n\n7.1 ")
    assert spans[0][2] == sections[1].path
    _assert_no_chunk_crosses(sections, spans, t)
    covered = set()
    for start, end, _ in spans:
        covered.update(range(start, end))
    assert all(i in covered for i, ch in enumerate(t) if not ch.isspace())


def test_heading_chain_folds():
    t = "Part I\n\nArticle 1 Scope\n\n1.1 The rules apply.\n\n1.2 Others.\n"
    spans = chunk_spans(detect_sections(t), t, 500, 50)
    assert [(t[a:b], path) for a, b, path in spans] == [
        (
            "Part I\n\nArticle 1 Scope\n\n1.1 The rules apply.",
            "Part I > Article 1 Scope > 1.1 The rules apply",
        ),
        ("1.2 Others.", "Part I > Article 1 Scope > 1.2 Others"),
    ]


def test_fold_stops_at_page_boundary():
    t = _TERMINATION
    spans = chunk_spans(detect_sections(t), t, 500, 50, boundaries=[t.index("7.1")])
    assert ("7. Termination", "7 Termination") in [(t[a:b], path) for a, b, path in spans]


def test_prose_clause_keeps_its_own_chunk():
    t = (
        "7. The Provider may end this agreement by notice, as follows:\n\n"
        "7.1 By post.\n\n7.2 By email.\n"
    )
    spans = chunk_spans(detect_sections(t), t, 500, 50)
    assert [path for _, _, path in spans] == ["7", "7 > 7.1 By post", "7 > 7.2 By email"]


# ---------------------------------------------------------------------------
# Stored map (database)
# ---------------------------------------------------------------------------


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

    result = await ingest(db_url, layer, src)
    assert result.docs_failed == 0
    sections, chunks = await _stored(db_url, layer)

    expected = _map_document(_parse_file(FIXTURES / "contract.md")).sections
    assert [
        (r["label"], r["title"], r["path"], r["depth"], r["char_start"], r["char_end"])
        for r in sections
    ] == [(s.label, s.title, s.path, s.depth, s.char_start, s.char_end) for s in expected]
    assert {c["clause_path"] for c in chunks} == {
        None,
        "1 Definitions",
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
    await ingest(db_url, layer, src)
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
    await ingest(db_url, layer, src)
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


async def test_ingest_builds_text_once_off_the_loop(db_url, layer, tmp_path):
    from stratakb.knowledge import ingestion

    src = tmp_path / "docs"
    src.mkdir()
    shutil.copy(FIXTURES / "contract.md", src / "contract.md")
    (src / "notes.txt").write_text(_TERMINATION, encoding="utf-8")

    off_loop: list[tuple[str, bool]] = []

    def spy(name, real):
        def wrapper(*args, **kwargs):
            off_loop.append((name, threading.current_thread() is not threading.main_thread()))
            return real(*args, **kwargs)
        return wrapper

    with (
        patch.object(ingestion, "build_document_text",
                     spy("build_document_text", ingestion.build_document_text)),
        patch.object(ingestion, "_parse_file", spy("_parse_file", ingestion._parse_file)),
        patch.object(ingestion, "detect_sections",
                     spy("detect_sections", ingestion.detect_sections)),
    ):
        result = await ingest(db_url, layer, src)

    assert result.docs_processed == 2
    names = [name for name, _ in off_loop]
    assert names.count("build_document_text") == 2
    assert names.count("_parse_file") == 2
    assert names.count("detect_sections") == 2
    assert all(threaded for name, threaded in off_loop)


async def test_ingest_markdown_loose_list_stores_no_section(db_url, layer, tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "guide.md").write_text(
        "Some intro.\n\n1. Install the tool\n\nRun it once.\n\n2. Configure it\n\nEdit the file.\n",
        encoding="utf-8",
    )
    result = await ingest(db_url, layer, src)
    assert result.docs_processed == 1
    sections, chunks = await _stored(db_url, layer)
    assert sections == []
    assert chunks and all(c["clause_path"] is None for c in chunks)


SEPA_STYLE = (
    "Attributes\n==========\n\n"
    "Identification: AT-44\n"
    "Name: The amount of the credit transfer in euro\n"
    "Description: The amount is given in euro, with two decimals. It covers the transfer.\n\n"
    "Identification: AT-45\n"
    "Name: The requested execution date of the instruction\n"
    "Description: The date on which the Originator's bank debits the account.\n"
)


async def test_preset_pattern_adds_sections(db_url, layer, tmp_path):
    from stratakb.knowledge.layers import create_layer, delete_layer
    from stratakb.knowledge.presets import load_preset

    patterns = load_preset(
        Path(__file__).parent.parent / "examples" / "sepa" / "preset.yaml"
    ).structure_patterns
    src = tmp_path / "docs"
    src.mkdir()
    (src / "rulebook.txt").write_text(SEPA_STYLE, encoding="utf-8")

    # A layer without patterns is unchanged: the attributes are body text
    await ingest(db_url, layer, src)
    sections, chunks = await _stored(db_url, layer)
    assert [r["label"] for r in sections] == ["attributes"]
    assert {c["clause_path"] for c in chunks} == {"Attributes"}

    name = f"test_struct_{uuid4().hex[:10]}"
    await create_layer(
        db_url,
        name=name,
        display_name="Test Structure Patterns",
        chunk_config={"chunk_size": 200, "chunk_overlap": 40},
        structure_patterns=patterns,
    )
    try:
        result = await ingest(db_url, name, src)
        assert result.docs_failed == 0
        sections, chunks = await _stored(db_url, name)
        assert [(r["label"], r["path"]) for r in sections] == [
            ("attributes", "Attributes"),
            ("at-44", "Attributes > Identification: AT-44"),
            ("at-45", "Attributes > Identification: AT-45"),
        ]
        at44 = next(r for r in sections if r["label"] == "at-44")
        assert SEPA_STYLE.index("Identification: AT-44") == at44["char_start"]
        assert any(c["clause_path"] == sections[1]["path"] for c in chunks)
        assert any(c["clause_path"] == sections[2]["path"] for c in chunks)
    finally:
        await delete_layer(db_url, name=name)


async def test_apply_preset_dry_run_runs_in_a_thread(server):
    from stratakb.knowledge import presets

    threads: list[bool] = []
    real = presets.dry_run_parse

    def spy(*args, **kwargs):
        threads.append(threading.current_thread() is not threading.main_thread())
        return real(*args, **kwargs)

    preset_path = Path(__file__).parent.parent / "examples" / "sepa" / "preset.yaml"
    with (
        patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
        patch("stratakb.knowledge.tools.get_current_user", return_value=("tester", "admin")),
        patch("stratakb.knowledge.presets.dry_run_parse", side_effect=spy),
        patch(
            "stratakb.knowledge.presets.check_preset_layer",
            new_callable=AsyncMock,
            return_value={"status": "would_create", "changes": []},
        ),
    ):
        result = await call_tool(
            server, "strata_apply_preset", preset_path=str(preset_path), dry_run=True
        )

    assert result["status"] == "dry_run_complete"
    assert threads == [True]
