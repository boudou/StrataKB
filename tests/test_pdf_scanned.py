"""Tests for the probably-scanned rule on real PDF files read by parse_pdf.

A page counts as scanned when it has an image and little text; a file is
flagged when more than half its pages count. The PDFs are built with pypdf at
test time, so no binary fixture is added. No database needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import PropertyMock, patch

from pypdf import PageObject, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from stratakb.knowledge.parsers import pdf_parser
from stratakb.knowledge.parsers.document import ParserFlag
from stratakb.knowledge.parsers.pdf_parser import parse_pdf

FIXTURES = Path(__file__).parent / "fixtures"

SCANNED = ParserFlag("probably_scanned")
SHORT_TEXT = "x" * 30
LONG_TEXT = "x" * 250
PIXELS = b"\x00\xff\x00\xff"


def _names(**entries) -> DictionaryObject:
    return DictionaryObject({NameObject(f"/{k}"): v for k, v in entries.items()})


def _stream(writer: PdfWriter, data: bytes, **entries):
    stream = DecodedStreamObject()
    stream.update(_names(**entries))
    stream.set_data(data)
    return writer._add_object(stream)


def _image(writer: PdfWriter):
    return _stream(
        writer,
        PIXELS,
        Type=NameObject("/XObject"),
        Subtype=NameObject("/Image"),
        Width=NumberObject(2),
        Height=NumberObject(2),
        ColorSpace=NameObject("/DeviceGray"),
        BitsPerComponent=NumberObject(8),
    )


def _write_pdf(
    path: Path, pages: list[tuple[str, str | None]], outline: tuple[str, ...] = ()
) -> Path:
    """A PDF of (text, image) pages; image is None, "xobject", "form" or "inline".

    Each outline title points at the first page.
    """
    writer = PdfWriter()
    font = writer._add_object(
        _names(
            Type=NameObject("/Font"),
            Subtype=NameObject("/Type1"),
            BaseFont=NameObject("/Helvetica"),
        )
    )
    for text, image in pages:
        page = writer.add_blank_page(width=612, height=792)
        xobjects = {}
        content = b""
        if text:
            content += f"BT /F1 6 Tf 10 700 Td ({text}) Tj ET\n".encode("ascii")
        if image == "xobject":
            xobjects["Im0"] = _image(writer)
            content += b"q 100 0 0 100 50 50 cm /Im0 Do Q\n"
        elif image == "form":
            inner = _names(XObject=_names(Im0=_image(writer)))
            xobjects["Fm0"] = _stream(
                writer,
                b"q 100 0 0 100 0 0 cm /Im0 Do Q\n",
                Type=NameObject("/XObject"),
                Subtype=NameObject("/Form"),
                BBox=ArrayObject(NumberObject(n) for n in (0, 0, 612, 792)),
                Resources=inner,
            )
            content += b"/Fm0 Do\n"
        elif image == "inline":
            inline = b"BI /W 2 /H 2 /CS /G /BPC 8 ID " + PIXELS + b" EI"
            content += b"q 100 0 0 100 50 50 cm " + inline + b" Q\n"
        page[NameObject("/Resources")] = _names(
            Font=_names(F1=font),
            XObject=_names(**xobjects),
        )
        if content:
            page[NameObject("/Contents")] = _stream(writer, content)
    for title in outline:
        writer.add_outline_item(title, 0)
    with path.open("wb") as f:
        writer.write(f)
    return path


def test_short_text_pdf_is_not_flagged():
    # 2 pages of 146 and 135 characters, no image
    doc = parse_pdf(FIXTURES / "test_doc.pdf")

    assert doc.error is None
    assert doc.flag is None
    assert doc.scanned_pages == set()
    assert len(doc.pages) == doc.total_pages == 2


def test_one_of_four_with_images_flagged(tmp_path):
    pdf = _write_pdf(
        tmp_path / "scan.pdf",
        [(LONG_TEXT, None), ("", "xobject"), ("", "xobject"), ("", "xobject")],
    )
    doc = parse_pdf(pdf)

    assert doc.flag == SCANNED
    assert doc.scanned_pages == {2, 3, 4}
    assert [p.page_number for p in doc.pages] == [1]


def test_two_of_four_with_images_not_flagged(tmp_path):
    pdf = _write_pdf(
        tmp_path / "half.pdf",
        [(LONG_TEXT, None), (LONG_TEXT, None), ("", "xobject"), ("", "xobject")],
    )
    doc = parse_pdf(pdf)

    assert doc.flag is None
    assert doc.scanned_pages == {3, 4}


def test_one_of_four_without_images_not_flagged(tmp_path):
    pdf = _write_pdf(
        tmp_path / "blank.pdf",
        [(LONG_TEXT, None), ("", None), ("", None), ("", None)],
    )
    doc = parse_pdf(pdf)

    assert doc.flag is None
    assert doc.scanned_pages == set()
    assert doc.total_pages == 4


def test_no_text_at_all_is_not_flagged(tmp_path):
    # Ingestion reports a file with no text pages as no_text, images or not
    pdf = _write_pdf(tmp_path / "scan.pdf", [("", "xobject"), ("", None)])
    doc = parse_pdf(pdf)

    assert doc.pages == []
    assert doc.flag is None
    assert doc.scanned_pages == {1}


def test_short_text_with_image_on_every_page_flagged(tmp_path):
    pdf = _write_pdf(tmp_path / "logo.pdf", [(SHORT_TEXT, "xobject")])
    doc = parse_pdf(pdf)

    assert doc.flag == SCANNED
    assert doc.scanned_pages == {1}


def test_just_under_200_chars_with_image_flagged(tmp_path):
    pdf = _write_pdf(tmp_path / "under.pdf", [("x" * 199, "xobject")])
    doc = parse_pdf(pdf)

    assert doc.flag == SCANNED
    assert doc.scanned_pages == {1}


def test_200_chars_with_image_not_flagged(tmp_path):
    pdf = _write_pdf(tmp_path / "limit.pdf", [("x" * 200, "xobject")])
    doc = parse_pdf(pdf)

    assert doc.flag is None
    assert doc.scanned_pages == set()


def test_long_text_with_image_not_flagged(tmp_path):
    pdf = _write_pdf(tmp_path / "report.pdf", [(LONG_TEXT, "xobject")])

    # A page with enough text never pays for looking at its images
    with patch.object(PageObject, "images", new_callable=PropertyMock) as images:
        doc = parse_pdf(pdf)

    images.assert_not_called()
    assert doc.flag is None
    assert doc.scanned_pages == set()


def test_image_in_form_xobject_counts(tmp_path):
    pdf = _write_pdf(tmp_path / "form.pdf", [(SHORT_TEXT, "form")])
    doc = parse_pdf(pdf)

    assert doc.flag == SCANNED
    assert doc.scanned_pages == {1}


def test_inline_image_does_not_fail_the_file(tmp_path):
    # Reading an inline image needs Pillow, which may not be installed
    pdf = _write_pdf(tmp_path / "inline.pdf", [(SHORT_TEXT, "inline")])
    doc = parse_pdf(pdf)

    assert doc.error is None
    assert doc.scanned_pages == {1}
    assert doc.pages[0].text == SHORT_TEXT


def test_unreadable_images_do_not_fail_the_file(tmp_path):
    pdf = _write_pdf(tmp_path / "broken.pdf", [(SHORT_TEXT, None)])

    with patch.object(PageObject, "images", new_callable=PropertyMock, side_effect=ValueError):
        doc = parse_pdf(pdf)

    assert doc.error is None
    assert doc.scanned_pages == {1}
    assert doc.pages[0].text == SHORT_TEXT


def test_scanned_wins_over_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_parser, "MAX_OUTLINE_ENTRIES", 1)
    pdf = _write_pdf(
        tmp_path / "scan.pdf",
        [(SHORT_TEXT, "xobject")],
        outline=("One", "Two", "Three"),
    )

    assert parse_pdf(pdf).flag == SCANNED
