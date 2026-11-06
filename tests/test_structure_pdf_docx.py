"""Tests for the headings PDF outlines and Word heading styles give the structure map.

The PDFs are the fixture PDF with an outline added, and the DOCX files are
built with python-docx, both at test time, so no binary fixture is added.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from stratakb.knowledge.documents import build_document_text, document_headings, locate
from stratakb.knowledge.ingestion import _chunk_document, _map_document, _parse_file
from stratakb.knowledge.parsers import pdf_parser
from stratakb.knowledge.parsers.pdf_parser import ParsedPage, ParserFlag, _attach_outline
from stratakb.knowledge.structure import Heading, detect_sections

FIXTURES = Path(__file__).parent / "fixtures"


def _pdf_with_outline(path: Path, *, blank_page_at: int | None = None, outline=()) -> Path:
    """The fixture PDF (2 pages) with an optional blank page inserted and an outline.

    outline holds (title, page index, parent title or None), in order.
    """
    from pypdf import PdfWriter

    writer = PdfWriter(clone_from=str(FIXTURES / "test_doc.pdf"))
    if blank_page_at is not None:
        writer.insert_blank_page(index=blank_page_at)
    items = {}
    for title, page_index, parent in outline:
        items[title] = writer.add_outline_item(title, page_index, parent=items.get(parent))
    with path.open("wb") as f:
        writer.write(f)
    return path


def _docx(path: Path, blocks) -> Path:
    """A DOCX of blocks: ('h1'..'h9', text), ('p', text) or ('table', rows)."""
    from docx import Document

    doc = Document()
    for kind, content in blocks:
        if kind == "p":
            doc.add_paragraph(content)
        elif kind == "table":
            table = doc.add_table(rows=len(content), cols=len(content[0]))
            for r, row in enumerate(content):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
        else:
            doc.add_heading(content, level=int(kind[1:]))
    doc.save(str(path))
    return path


# ---------------------------------------------------------------------------
# PDF outlines
# ---------------------------------------------------------------------------


def test_pdf_outline_sections(tmp_path):
    pdf = _pdf_with_outline(
        tmp_path / "orders.pdf",
        outline=[
            ("Service: Customer Order", 0, None),
            ("Service Operations", 1, "Service: Customer Order"),
        ],
    )
    doc = _parse_file(pdf)
    mapped = _map_document(doc)

    assert [(s.path, s.depth) for s in mapped.sections] == [
        ("Service: Customer Order", 1),
        ("Service: Customer Order > Service Operations", 2),
    ]
    first, second = mapped.sections
    assert first.char_end == second.char_end == len(mapped.text)
    # One section per entry, each on its page
    pages = [locate(mapped.text, mapped.page_map, s.char_start, s.char_start)["page"]
             for s in mapped.sections]
    assert pages == [1, 2]
    assert mapped.text[second.char_start:].startswith("Service Operations")
    # The heading line is the heading, not the section's own text
    assert mapped.text[first.body_start:].startswith("The Customer Order service")

    chunks = _chunk_document(doc, layer_name="t", chunk_size=500, chunk_overlap=50,
                             mapped=mapped)
    assert [(c.metadata["page"], c.clause_path) for c in chunks] == [
        (1, "Service: Customer Order"),
        (2, "Service: Customer Order > Service Operations"),
    ]


def test_pdf_outline_title_not_in_text(tmp_path):
    # A title the page text does not hold is placed at the page's start,
    # and the page's first line stays the section's own text
    pdf = _pdf_with_outline(tmp_path / "o.pdf", outline=[("Overview", 0, None)])
    mapped = _map_document(_parse_file(pdf))

    assert [(s.label, s.title, s.path, s.char_start) for s in mapped.sections] == [
        ("overview", "Overview", "Overview", 0)
    ]
    assert mapped.sections[0].body_start == 0


def test_pdf_outline_entry_on_empty_page(tmp_path):
    # The parser drops a page without text: its entry goes to the next page
    pdf = _pdf_with_outline(
        tmp_path / "gap.pdf",
        blank_page_at=1,
        outline=[("Service: Customer Order", 0, None), ("Intermission", 1, None)],
    )
    doc = _parse_file(pdf)
    assert doc.total_pages == 3
    assert [p.page_number for p in doc.pages] == [1, 3]
    mapped = _map_document(doc)

    assert [s.path for s in mapped.sections] == ["Service: Customer Order", "Intermission"]
    intermission = mapped.sections[1]
    assert locate(mapped.text, mapped.page_map, intermission.char_start,
                  intermission.char_start)["page"] == 3


def test_pdf_without_outline_unchanged():
    doc = _parse_file(FIXTURES / "test_doc.pdf")
    assert all(page.headings == [] for page in doc.pages)
    assert document_headings(doc) == []
    text = build_document_text(doc)[0]
    assert _map_document(doc).sections == detect_sections(text)


# ---------------------------------------------------------------------------
# Word heading styles
# ---------------------------------------------------------------------------


_CONTRACT = [
    ("h1", "Scope"),
    ("p", "This agreement covers the delivery of services."),
    ("h1", "Fees"),
    ("p", "Fees are due monthly."),
    ("h2", "Late fees"),
    ("p", "Late payment adds interest:"),
    ("table", [["Days late", "Rate"], ["30", "1%"], ["60", "2%"]]),
    ("p", "Interest is capped."),
    ("h1", "Termination"),
    ("p", "Either party may terminate with notice."),
]


def test_docx_heading_styles(tmp_path):
    doc = _parse_file(_docx(tmp_path / "contract.docx", _CONTRACT))
    mapped = _map_document(doc)

    assert [(s.label, s.path, s.depth) for s in mapped.sections] == [
        ("scope", "Scope", 1),
        ("fees", "Fees", 1),
        ("late fees", "Fees > Late fees", 2),
        ("termination", "Termination", 1),
    ]
    for s in mapped.sections:
        assert mapped.text[s.char_start:].startswith(s.title)
        assert mapped.text[s.body_start - 1] == "\n"


def test_docx_table_stays_in_section(tmp_path):
    doc = _parse_file(_docx(tmp_path / "contract.docx", _CONTRACT))
    mapped = _map_document(doc)
    text = mapped.text

    # Paragraphs and tables keep their document order
    assert text.index("Late payment adds interest:") < text.index("Days late | Rate")
    assert text.index("Days late | Rate") < text.index("Interest is capped.")
    assert text.index("60 | 2%") < text.index("Termination")

    late = next(s for s in mapped.sections if s.label == "late fees")
    table_at = text.index("Days late | Rate")
    assert late.char_start < table_at < late.char_end

    chunks = _chunk_document(doc, layer_name="t", chunk_size=500, chunk_overlap=50,
                             mapped=mapped)
    holding = [c for c in chunks if "Days late | Rate" in c.content]
    assert [c.clause_path for c in holding] == ["Fees > Late fees"]


def test_docx_headings_across_pseudo_pages(tmp_path):
    # Pseudo-pages are about 3000 characters: a heading on a later one keeps
    # its offset in the document text
    filler = "Words that fill the section with plain text. " * 20
    blocks = []
    for n in range(1, 6):
        blocks += [("h1", f"Part {n} heading"), ("p", filler), ("p", filler)]
    doc = _parse_file(_docx(tmp_path / "long.docx", blocks))
    assert len(doc.pages) > 1
    mapped = _map_document(doc)

    assert [s.path for s in mapped.sections] == [f"Part {n} heading" for n in range(1, 6)]
    for s in mapped.sections:
        assert mapped.text[s.char_start:].startswith(s.path)


def test_docx_numbered_headings_and_clauses(tmp_path):
    # A numbered heading style is read once, and numbered clauses in the text
    # nest inside the styled heading
    blocks = [
        ("h1", "1. Payments"),
        ("p", "1.1 Invoices are sent monthly."),
        ("p", "1.2 Payment is due in 30 days."),
        ("h1", "2. Liability"),
        ("p", "Liability is limited."),
    ]
    mapped = _map_document(_parse_file(_docx(tmp_path / "n.docx", blocks)))

    assert [(s.label, s.path) for s in mapped.sections] == [
        ("1", "1 Payments"),
        ("1.1", "1 Payments > 1.1 Invoices are sent monthly"),
        ("1.2", "1 Payments > 1.2 Payment is due in 30 days"),
        ("2", "2 Liability"),
    ]


def test_docx_without_heading_styles_unchanged(tmp_path):
    blocks = [
        ("p", "1. Scope"),
        ("p", "The scope of the agreement."),
        ("table", [["a", "b"]]),
        ("p", "2. Fees"),
        ("p", "None."),
    ]
    doc = _parse_file(_docx(tmp_path / "plain.docx", blocks))
    assert document_headings(doc) == []
    text = build_document_text(doc)[0]
    assert text == "1. Scope\nThe scope of the agreement.\na | b\n2. Fees\nNone."
    sections = _map_document(doc).sections
    assert sections == detect_sections(text)
    assert [s.label for s in sections] == ["1"]


# ---------------------------------------------------------------------------
# detect_sections with explicit headings
# ---------------------------------------------------------------------------


def test_explicit_headings_come_first():
    text = "Intro text.\n\nOverview\n\n## Details\n\nSome text.\n\nAnnex\n\n3. Rules\n\nMore.\n"
    explicit = [
        Heading(text.index("Overview"), 1, "Overview"),
        Heading(text.index("Annex"), 1, "Annex"),
    ]

    sections = detect_sections(text, explicit=explicit)
    assert [(s.path, s.depth) for s in sections] == [
        ("Overview", 1),
        ("Overview > Details", 2),
        ("Annex", 1),
        ("Annex > 3 Rules", 2),
    ]
    # A text heading does not close the explicit one it lies in
    assert sections[0].char_end == sections[2].char_start
    assert sections[2].char_end == len(text)


def test_explicit_heading_mid_line_starts_at_its_line():
    text = "Preamble.\n\nPage header Terms of use\nBody.\n"
    sections = detect_sections(text, explicit=[Heading(text.index("Terms"), 1, "Terms of use")])
    assert [(s.path, s.char_start) for s in sections] == [
        ("Terms of use", text.index("Page header"))
    ]
    assert sections[0].body_start == text.index("Body.")


def test_explicit_empty_is_unchanged():
    text = "1. Scope\n\nText.\n\n2. Fees\n\nNone.\n"
    assert detect_sections(text, explicit=()) == detect_sections(text)


def test_unnumbered_outline_keeps_line_numbers():
    # Outline titles without numbers (a common PDF bookmark style) over
    # numbered heading lines: the line gives the label, so a lookup by
    # clause number still finds the section
    text = "1 Introduction\nIntro.\n\n1.1 Scope\nScope text.\n\nArticle 2 Definitions\nTerms.\n"
    explicit = [
        Heading(text.index("1 Introduction"), 1, "Introduction"),
        Heading(text.index("1.1 Scope"), 2, "Scope"),
        Heading(text.index("Article 2"), 1, "Definitions"),
    ]
    sections = detect_sections(text, explicit=explicit)

    assert [(s.label, s.title, s.path) for s in sections] == [
        ("1", "Introduction", "1 Introduction"),
        ("1.1", "Scope", "1 Introduction > 1.1 Scope"),
        ("article 2", "Definitions", "Article 2 Definitions"),
    ]


def test_numbered_explicit_heading_shows_its_number_once():
    # A styled heading or bookmark that holds its own number, whatever its
    # length or punctuation: the number is the label and shows once
    cases = [
        ("7 Warranties, representations and covenants",
         "7", "Warranties, representations and covenants"),
        ("Article 6 GDPR. Lawfulness of processing",
         "article 6", "GDPR. Lawfulness of processing"),
        ("1.1 Scope and purpose of this agreement and its relation to all other agreements",
         "1.1", "Scope and purpose of this agreement and its relation to all other agreements"),
        ("7 days notice, as agreed by both parties.",
         "7", "days notice, as agreed by both parties."),
    ]
    for heading, label, title in cases:
        text = f"Preamble.\n\n{heading}\nBody.\n"
        sections = detect_sections(text, explicit=[Heading(text.index(heading), 1, heading)])
        assert [(s.label, s.title, s.path) for s in sections] == [(label, title, heading)]


def test_bare_number_line_takes_the_outline_title_once():
    text = "Intro.\n\n7.\nPay now.\n"
    sections = detect_sections(text, explicit=[Heading(text.index("7."), 1, "7. Payment")])
    assert [(s.label, s.title, s.path) for s in sections] == [("7", "Payment", "7 Payment")]


def test_run_in_heading_line_takes_the_outline_title():
    # A line that is both the heading and the start of the clause text: the
    # outline gives the title, the line only the number
    text = (
        "Preamble.\n\n"
        "1. Definitions. In this Agreement the following terms apply to all orders\n"
        "More terms.\n\n"
        "1.1 Scope. This Agreement covers all services provided by the Supplier under any order.\n"
        "Scope text.\n"
    )
    explicit = [
        Heading(text.index("1. Definitions"), 1, "Definitions"),
        Heading(text.index("1.1 Scope"), 2, "Scope"),
    ]
    sections = detect_sections(text, explicit=explicit)

    assert [(s.label, s.title, s.path) for s in sections] == [
        ("1", "Definitions", "1 Definitions"),
        ("1.1", "Scope", "1 Definitions > 1.1 Scope"),
    ]


def test_explicit_heading_off_line_leaves_the_line_to_the_detectors():
    # An outline entry whose title is not on its page is placed at the page
    # start without occupying that line: the clause the page starts with
    # survives, inside the entry
    text = "1. Definitions\nThe terms.\n\n2. Payment\nPay now.\n"
    sections = detect_sections(text, explicit=[Heading(0, 1, "Part I", on_line=False)])

    assert [(s.label, s.path) for s in sections] == [
        ("part i", "Part I"),
        ("1", "Part I > 1 Definitions"),
        ("2", "Part I > 2 Payment"),
    ]
    assert sections[0].body_start == 0


class _FakeReader:
    """The part of a pypdf reader _attach_outline uses: an outline of one level."""

    def __init__(self, entries):
        self.outline = [SimpleNamespace(title=t, page=p) for t, p in entries]

    def get_destination_page_number(self, item):
        return item.page


def _outline_on(text: str, entries) -> list[Heading]:
    pages = [ParsedPage(text=text, page_number=1, source_file="x.pdf")]
    _attach_outline(_FakeReader(entries), pages, "x.pdf")
    return pages[0].headings


def test_outline_title_in_prose_before_its_heading():
    # A defined term in prose above its heading does not place the entry:
    # the heading line does, with or without its clause number
    text = (
        "6. Fees\nPayment is due in 30 days. Late Payment adds interest.\n\n"
        "7. Payment\nBy transfer.\n"
    )
    assert _outline_on(text, [("Payment", 0)]) == [
        Heading(text.index("7. Payment"), 1, "Payment", on_line=True)
    ]
    explicit = _outline_on(text, [("Fees", 0), ("Payment", 0)])
    sections = detect_sections(text, explicit=explicit)
    assert [(s.label, s.path) for s in sections] == [("6", "6 Fees"), ("7", "7 Payment")]
    assert sections[0].char_end == text.index("7. Payment")


def test_outline_title_only_mid_sentence_goes_to_page_start():
    text = "Some prose that mentions Scope in passing.\nMore prose.\n"
    assert _outline_on(text, [("Scope", 0)]) == [Heading(0, 1, "Scope", on_line=False)]
    sections = detect_sections(text, explicit=_outline_on(text, [("Scope", 0)]))
    assert [(s.path, s.body_start) for s in sections] == [("Scope", 0)]


def test_outline_title_does_not_take_digits_of_a_longer_number():
    # '2023' is no clause number: an outline entry '3 Annual report' does not
    # place on '2023 annual report' as if that were '3 annual report'
    text = "Cover.\n2023 annual report\n3 Annual report\nFigures.\n"
    assert _outline_on(text, [("3 Annual report", 0)]) == [
        Heading(text.index("3 Annual"), 1, "3 Annual report", on_line=True)
    ]
    # A numbered title places on its numbered line, and an unnumbered one on
    # a line with any clause number or keyword id
    text = "Intro.\nArticle 2: Definitions\nTerms.\n"
    assert _outline_on(text, [("Definitions", 0)]) == [
        Heading(text.index("Article 2"), 1, "Definitions", on_line=True)
    ]
    assert _outline_on(text, [("2. Definitions", 0)]) == [
        Heading(text.index("Article 2"), 1, "2. Definitions", on_line=True)
    ]


def test_outline_title_at_line_start():
    # Without a line that is the title, a line that starts with it places it
    text = "Intro.\nTermination. Either party may end this.\n"
    assert _outline_on(text, [("Termination", 0)]) == [
        Heading(text.index("Termination"), 1, "Termination", on_line=True)
    ]


# ---------------------------------------------------------------------------
# Bare clause numbers as headings
# ---------------------------------------------------------------------------


def test_bare_number_explicit_heading_shows_its_number_once():
    # A styled heading or bookmark that is nothing but a clause number or
    # keyword id, on a line that holds only that: no title, the number once
    cases = [
        ("Article 6", "article 6"),
        ("Annex II", "annex ii"),
        ("Section 3", "section 3"),
        ("7.", "7"),
        ("1.1", "1.1"),
        ("7(2)(b)", "7(2)(b)"),
    ]
    for heading, label in cases:
        text = f"Preamble.\n\n{heading}\nLawfulness of processing\nBody.\n"
        sections = detect_sections(text, explicit=[Heading(text.index(heading), 1, heading)])
        number = heading.removesuffix(".")
        assert [(s.label, s.title, s.path) for s in sections] == [(label, None, number)]


def test_bare_number_heading_takes_its_line_title():
    # The line's own title stands in when the heading is only the number
    text = "Preamble.\n\nArticle 6 Lawfulness\nBody.\n"
    sections = detect_sections(text, explicit=[Heading(text.index("Article 6"), 1, "Article 6")])
    assert [(s.label, s.title, s.path) for s in sections] == [
        ("article 6", "Lawfulness", "Article 6 Lawfulness")
    ]


def test_docx_bare_keyword_heading_style(tmp_path):
    # A Word heading 'Annex II' with its title in the next paragraph
    blocks = [
        ("h1", "Annex II"),
        ("p", "Technical and organisational measures"),
        ("p", "The processor shall encrypt data."),
        ("h1", "Article 12"),
        ("p", "Transparent information."),
    ]
    mapped = _map_document(_parse_file(_docx(tmp_path / "annex.docx", blocks)))
    assert [(s.label, s.title, s.path) for s in mapped.sections] == [
        ("annex ii", None, "Annex II"),
        ("article 12", None, "Article 12"),
    ]


def test_pdf_bare_keyword_bookmark_on_its_line():
    # The EUR-Lex layout: 'Article 6' on a line, its title on the next one
    text = "Recitals.\n\nArticle 6\nLawfulness of processing\n1. Processing shall be lawful.\n"
    explicit = _outline_on(text, [("Article 6", 0)])
    assert explicit == [Heading(text.index("Article 6"), 1, "Article 6", on_line=True)]
    sections = detect_sections(text, explicit=explicit)
    assert [(s.label, s.title, s.path) for s in sections][0] == ("article 6", None, "Article 6")
    assert all("Article 6 Article 6" not in s.path for s in sections)


# ---------------------------------------------------------------------------
# Outline placement: wrapped and longer titles, entries without a line
# ---------------------------------------------------------------------------


def test_outline_title_wrapped_over_lines():
    # The bookmark holds the whole title; the page wraps it after the number
    text = "Recitals.\n\nArticle 6\nLawfulness of processing\nBody text.\n"
    title = "Article 6 Lawfulness of processing"
    explicit = _outline_on(text, [(title, 0)])
    assert explicit == [Heading(text.index("Article 6"), 1, title, on_line=True)]
    sections = detect_sections(text, explicit=explicit)
    # One section, not the entry with 'Article 6' nested inside it again
    assert [(s.label, s.title, s.path) for s in sections] == [
        ("article 6", "Lawfulness of processing", "Article 6 Lawfulness of processing")
    ]


def test_outline_title_without_its_number_wrapped_over_lines():
    # An unnumbered bookmark over a numbered heading that wraps
    text = "Intro.\n\n7. Payment of the\nfees due\nPay now.\n"
    title = "Payment of the fees due"
    explicit = _outline_on(text, [(title, 0)])
    assert explicit == [Heading(text.index("7."), 1, title, on_line=True)]
    sections = detect_sections(text, explicit=explicit)
    assert [(s.label, s.path) for s in sections] == [("7", "7 Payment of the fees due")]


def test_outline_title_with_extra_whitespace():
    text = "Intro.\nPayment terms\nPay now.\n"
    assert _outline_on(text, [("Payment \n  terms", 0)]) == [
        Heading(text.index("Payment"), 1, "Payment terms", on_line=True)
    ]


def test_outline_title_longer_than_its_line():
    # The bookmark carries more than the heading line: the clause number places it
    text = "Recitals.\n\nArticle 6 Lawfulness of processing\nBody text.\n"
    title = "Article 6 Lawfulness of processing of personal data"
    explicit = _outline_on(text, [(title, 0)])
    assert explicit == [Heading(text.index("Article 6"), 1, title, on_line=True)]
    sections = detect_sections(text, explicit=explicit)
    assert [s.label for s in sections] == ["article 6"]


def test_outline_title_number_does_not_place_on_a_page_number():
    # A line holding only '6' is a page number, not clause 6
    text = "6\nSome prose without the title.\n"
    assert _outline_on(text, [("6 Termination", 0)]) == [
        Heading(0, 1, "6 Termination", on_line=False)
    ]


def test_outline_entries_without_a_line_keep_only_those_with_text():
    # Two sibling entries not found on the page would both start at its
    # start; the first would cover no text and is left out. A child entry
    # nests inside its parent and is kept.
    text = "Some prose.\nMore prose.\n"
    assert _outline_on(text, [("Part I", 0), ("Part II", 0)]) == [
        Heading(0, 1, "Part II", on_line=False)
    ]
    pages = [ParsedPage(text=text, page_number=1, source_file="x.pdf")]
    reader = _FakeReader([("Part I", 0)])
    reader.outline.append([SimpleNamespace(title="Chapter 1", page=0)])
    _attach_outline(reader, pages, "x.pdf")
    assert pages[0].headings == [
        Heading(0, 1, "Part I", on_line=False),
        Heading(0, 2, "Chapter 1", on_line=False),
    ]


# ---------------------------------------------------------------------------
# Outline caps
# ---------------------------------------------------------------------------


def test_large_outline_is_capped_and_fast():
    # A crafted outline with many entries over a long page cannot keep the
    # ingest busy: entries past the cap are left out and counted
    import time

    lines = [f"Clause {n} text that does not match any title." for n in range(5000)]
    text = "\n".join(lines)
    entries = [(f"Heading {n}", 0) for n in range(100_000)]
    pages = [ParsedPage(text=text, page_number=1, source_file="x.pdf")]

    started = time.monotonic()
    placed, left_out = _attach_outline(_FakeReader(entries), pages, "x.pdf")
    elapsed = time.monotonic() - started

    assert left_out == 100_000 - pdf_parser.MAX_OUTLINE_ENTRIES
    assert placed <= pdf_parser.MAX_OUTLINE_ENTRIES
    assert elapsed < 10


def test_deep_outline_is_capped():
    # Entries nested deeper than the cap are left out and counted
    outline: list = []
    level = outline
    for depth in range(1, 13):
        level.append(SimpleNamespace(title=f"Level {depth}", page=0))
        child: list = []
        level.append(child)
        level = child
    reader = _FakeReader([])
    reader.outline = outline
    pages = [ParsedPage(text="Body.\n", page_number=1, source_file="x.pdf")]

    placed, left_out = _attach_outline(reader, pages, "x.pdf")
    assert left_out == 12 - pdf_parser.MAX_OUTLINE_DEPTH
    assert max(h.depth for h in pages[0].headings) == pdf_parser.MAX_OUTLINE_DEPTH


def test_pdf_past_the_outline_cap_is_flagged_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_parser, "MAX_OUTLINE_ENTRIES", 2)
    # The fixture's pages are short: it would be flagged probably scanned,
    # which wins over truncated
    monkeypatch.setattr(pdf_parser, "_SCANNED_MIN_CHARS_PER_PAGE", 0)
    pdf =_pdf_with_outline(
        tmp_path / "many.pdf",
        outline=[("Service: Customer Order", 0, None), ("Service Operations", 1, None),
                 ("Extra one", 1, None), ("Extra two", 1, None)],
    )
    doc = _parse_file(pdf)
    assert doc.flag == ParserFlag("truncated", 2)
    assert [s.path for s in _map_document(doc).sections] == [
        "Service: Customer Order", "Service Operations"
    ]


# ---------------------------------------------------------------------------
# Localized and custom Word heading styles
# ---------------------------------------------------------------------------


def test_docx_localized_heading_style(tmp_path):
    # A German Word install names its heading style 'Überschrift 1', with its
    # own style id; its outline level marks it as a heading
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document()
    style = doc.styles.add_style("Überschrift 1", WD_STYLE_TYPE.PARAGRAPH)
    ppr = style.element.get_or_add_pPr()
    level = OxmlElement("w:outlineLvl")
    level.set(qn("w:val"), "0")
    ppr.append(level)
    sub = doc.styles.add_style("Eigene Unterüberschrift", WD_STYLE_TYPE.PARAGRAPH)
    sub.base_style = doc.styles["Heading 2"]
    doc.add_paragraph("Zahlung", style="Überschrift 1")
    doc.add_paragraph("Zahlbar in 30 Tagen.")
    doc.add_paragraph("Verzug", style="Eigene Unterüberschrift")
    doc.add_paragraph("Zinsen fallen an.")
    path = tmp_path / "de.docx"
    doc.save(str(path))

    mapped = _map_document(_parse_file(path))
    assert [(s.path, s.depth) for s in mapped.sections] == [
        ("Zahlung", 1),
        ("Zahlung > Verzug", 2),
    ]
