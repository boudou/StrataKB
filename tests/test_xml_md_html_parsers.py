"""Tests for XML/XSD, Markdown, and HTML parsers.

Tests cover:
- XSD schema parsing with complex types, simple types, elements, groups
- Generic XML fallback
- Markdown splitting by heading hierarchy (H1/H2/H3)
- HTML parsing with tag stripping and section chunking
- Error handling for bad files
- Integration with ingestion pipeline (SUPPORTED_PARSERS)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_XSD = FIXTURES_DIR / "test_schema.xsd"
TEST_MD = FIXTURES_DIR / "test_document.md"
TEST_HTML = FIXTURES_DIR / "test_page.html"


# ==========================================
# XML/XSD parser tests
# ==========================================


class TestXSDParser:
    def test_detects_xsd_format(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        assert result.error is None
        assert result.total_pages > 0

    def test_summary_page_present(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        summary = result.pages[0]
        assert summary.page_number == 0
        assert "XSD Schema" in summary.text
        assert "Complex Types" in summary.text

    def test_extracts_complex_types(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        texts = [p.text for p in result.pages]
        complex_type_pages = [t for t in texts if "ComplexType:" in t]
        assert len(complex_type_pages) == 4  # Document, CustomerCreditTransfer, GroupHeader, PaymentInstruction

    def test_complex_type_has_description(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        # Find the Document complex type
        doc_ct = [p for p in result.pages if "ComplexType: Document" in p.text]
        assert len(doc_ct) == 1
        assert "SEPA Credit Transfer" in doc_ct[0].text

    def test_complex_type_has_child_elements(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        # GroupHeader32 has child elements — match the exact ComplexType page
        gh = [p for p in result.pages if "ComplexType: GroupHeader32" in p.text]
        assert len(gh) == 1
        assert "MsgId" in gh[0].text
        assert "CreDtTm" in gh[0].text

    def test_element_constraints_extracted(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        # CustomerCreditTransferInitiationV03 has PmtInf with maxOccurs=unbounded
        ct = [p for p in result.pages if "ComplexType: CustomerCreditTransferInitiationV03" in p.text]
        assert len(ct) == 1
        assert "unbounded" in ct[0].text

    def test_extracts_simple_types(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        texts = [p.text for p in result.pages]
        simple_type_pages = [t for t in texts if "SimpleType:" in t]
        assert len(simple_type_pages) == 2  # PaymentMethod3Code, IBANIdentifier

    def test_simple_type_has_enumerations(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        pmc = [p for p in result.pages if "PaymentMethod3Code" in p.text and "SimpleType:" in p.text]
        assert len(pmc) == 1
        assert "CHK" in pmc[0].text
        assert "TRF" in pmc[0].text
        assert "TRA" in pmc[0].text

    def test_simple_type_has_pattern(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        iban = [p for p in result.pages if "IBANIdentifier" in p.text and "SimpleType:" in p.text]
        assert len(iban) == 1
        assert "pattern" in iban[0].text

    def test_extracts_top_level_elements(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        element_pages = [p for p in result.pages if "Element: Document" in p.text]
        assert len(element_pages) == 1

    def test_extracts_groups(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        group_pages = [p for p in result.pages if "Group:" in p.text]
        assert len(group_pages) == 1
        assert "PaymentIdentificationGroup" in group_pages[0].text
        assert "EndToEndId" in group_pages[0].text

    def test_target_namespace_in_summary(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        summary = result.pages[0]
        assert "pain.001" in summary.text

    def test_page_count(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(TEST_XSD)
        # 1 summary + 4 complex types + 2 simple types + 1 element + 1 group = 9
        assert result.total_pages == 9


class TestGenericXMLParser:
    def test_parses_generic_xml(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write('<?xml version="1.0"?><root><item name="test">Hello</item></root>')
            f.flush()

            result = parse_xml(f.name)
            assert result.error is None
            assert result.total_pages == 1
            assert "Hello" in result.pages[0].text
            assert "item" in result.pages[0].text

        Path(f.name).unlink()

    def test_generic_xml_preserves_attributes(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write('<config version="2.0"><setting key="timeout">30</setting></config>')
            f.flush()

            result = parse_xml(f.name)
            assert result.error is None
            assert 'version="2.0"' in result.pages[0].text
            assert "30" in result.pages[0].text

        Path(f.name).unlink()


class TestXMLErrors:
    def test_nonexistent_file(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml("/nonexistent/file.xsd")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_non_xml_file(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        result = parse_xml(__file__)  # .py file
        assert result.error is not None
        assert "Not an XML" in result.error

    def test_malformed_xml(self):
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write("<broken><unclosed>")
            f.flush()

            result = parse_xml(f.name)
            assert result.error is not None
            assert "parse error" in result.error.lower()

        Path(f.name).unlink()


# ==========================================
# Markdown parser tests
# ==========================================


class TestMarkdownParser:
    def test_parses_markdown(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        assert result.error is None
        assert result.total_pages > 0

    def test_splits_by_headings(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        # H1: Customer Order (1) + H2: Overview (1) + H3: Key Capabilities (1)
        # + H3: Behavior Qualifiers (1) + H2: API Operations (1) + H3: Initiate (1)
        # + H3: Retrieve (1) + H2: Connected Services (1) = 8 sections
        assert result.total_pages == 8

    def test_h1_section_present(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        h1_page = result.pages[0]
        assert "# Customer Order Service" in h1_page.text
        assert "orchestrates" in h1_page.text

    def test_h2_inherits_h1_context(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        # Find the "Overview" section
        overview = [p for p in result.pages if "## Overview" in p.text]
        assert len(overview) >= 1
        # Should have H1 context
        assert "# Customer Order" in overview[0].text

    def test_h3_inherits_h1_and_h2_context(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        # Find "Key Capabilities" section
        kc = [p for p in result.pages if "### Key Capabilities" in p.text]
        assert len(kc) == 1
        # Should have H1 and H2 context
        assert "# Customer Order" in kc[0].text
        assert "## Overview" in kc[0].text

    def test_frontmatter_stripped(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        for page in result.pages:
            assert "---" not in page.text or "sd_overview" not in page.text

    def test_content_preserved(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        all_text = " ".join(p.text for p in result.pages)
        assert "DepositsAndWithdrawals" in all_text
        assert "POST /CustomerOrder/Initiate" in all_text
        assert "Payment Initiation" in all_text

    def test_no_headings_single_page(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("Just some plain text without any headings.\nAnother line.")
            f.flush()

            result = parse_markdown(f.name)
            assert result.error is None
            assert result.total_pages == 1
            assert "plain text" in result.pages[0].text

        Path(f.name).unlink()


class TestMarkdownErrors:
    def test_nonexistent_file(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown("/nonexistent/file.md")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_non_markdown_file(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(__file__)
        assert result.error is not None
        assert "Not a Markdown" in result.error

    def test_empty_markdown(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("")
            f.flush()

            result = parse_markdown(f.name)
            assert result.error is not None
            assert "empty" in result.error.lower()

        Path(f.name).unlink()


# ==========================================
# HTML parser tests
# ==========================================


class TestHTMLParser:
    def test_parses_html(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        assert result.error is None
        assert result.total_pages > 0

    def test_splits_by_headings(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        # Preamble (title text) + H1: SEPA Credit Transfer + H2: Message Format
        # + H3: Group Header + H3: Payment Information + H2: Processing Rules
        # + H3: Validation = 7
        assert result.total_pages == 7

    def test_strips_script_and_style(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        all_text = " ".join(p.text for p in result.pages)
        assert "console.log" not in all_text
        assert "font-family" not in all_text

    def test_strips_nav_and_footer(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        all_text = " ".join(p.text for p in result.pages)
        assert "Navigation menu" not in all_text
        assert "Copyright" not in all_text

    def test_h1_content_present(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        # Page 0 is preamble (title), page 1 is the H1 section
        h1_page = result.pages[1]
        assert "SEPA Credit Transfer" in h1_page.text
        assert "euro-denominated" in h1_page.text

    def test_h2_has_h1_context(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        # Find Message Format section
        mf = [p for p in result.pages if "Message Format" in p.text and "##" in p.text]
        assert len(mf) >= 1
        assert "# SEPA Credit Transfer" in mf[0].text

    def test_h3_has_parent_context(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        # Find Group Header section
        gh = [p for p in result.pages if "Group Header" in p.text and "###" in p.text]
        assert len(gh) == 1
        assert "# SEPA Credit Transfer" in gh[0].text
        assert "## Message Format" in gh[0].text

    def test_content_preserved(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        all_text = " ".join(p.text for p in result.pages)
        assert "pain.001" in all_text
        assert "IBAN validation" in all_text

    def test_no_headings_single_page(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
            f.write("<html><head><title>Test</title></head><body><p>Simple content</p></body></html>")
            f.flush()

            result = parse_html(f.name)
            assert result.error is None
            assert result.total_pages == 1
            assert "Simple content" in result.pages[0].text

        Path(f.name).unlink()


class TestHTMLErrors:
    def test_nonexistent_file(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html("/nonexistent/file.html")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_non_html_file(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(__file__)
        assert result.error is not None
        assert "Not an HTML" in result.error

    def test_empty_html(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
            f.write("")
            f.flush()

            result = parse_html(f.name)
            assert result.error is not None
            assert "empty" in result.error.lower()

        Path(f.name).unlink()


# ==========================================
# Pipeline integration tests
# ==========================================


class TestPipelineIntegration:
    def test_xml_registered_in_supported_parsers(self):
        from stratakb.knowledge.ingestion import SUPPORTED_PARSERS

        assert ".xml" in SUPPORTED_PARSERS
        assert ".xsd" in SUPPORTED_PARSERS

    def test_markdown_registered_in_supported_parsers(self):
        from stratakb.knowledge.ingestion import SUPPORTED_PARSERS

        assert ".md" in SUPPORTED_PARSERS
        assert ".markdown" in SUPPORTED_PARSERS

    def test_html_registered_in_supported_parsers(self):
        from stratakb.knowledge.ingestion import SUPPORTED_PARSERS

        assert ".html" in SUPPORTED_PARSERS
        assert ".htm" in SUPPORTED_PARSERS

    def test_parse_file_dispatches_xsd(self):
        from stratakb.knowledge.ingestion import _parse_file

        result = _parse_file(TEST_XSD)
        assert result.error is None
        assert result.total_pages > 0

    def test_parse_file_dispatches_markdown(self):
        from stratakb.knowledge.ingestion import _parse_file

        result = _parse_file(TEST_MD)
        assert result.error is None
        assert result.total_pages > 0

    def test_parse_file_dispatches_html(self):
        from stratakb.knowledge.ingestion import _parse_file

        result = _parse_file(TEST_HTML)
        assert result.error is None
        assert result.total_pages > 0

    def test_chunk_document_sets_xsd_doc_type(self):
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.xml_parser import parse_xml

        doc = parse_xml(TEST_XSD)
        chunks = _chunk_document(doc, layer_name="test", chunk_size=2000, chunk_overlap=100)
        assert len(chunks) > 0
        assert chunks[0].doc_type == "xsd"

    def test_chunk_document_sets_markdown_doc_type(self):
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        doc = parse_markdown(TEST_MD)
        chunks = _chunk_document(doc, layer_name="test", chunk_size=2000, chunk_overlap=100)
        assert len(chunks) > 0
        assert chunks[0].doc_type == "markdown"

    def test_chunk_document_sets_html_doc_type(self):
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.html_parser import parse_html

        doc = parse_html(TEST_HTML)
        chunks = _chunk_document(doc, layer_name="test", chunk_size=2000, chunk_overlap=100)
        assert len(chunks) > 0
        assert chunks[0].doc_type == "html"

    def test_discover_xml_files(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(FIXTURES_DIR)
        xsd_files = [f for f in files if f.suffix in (".xml", ".xsd")]
        assert len(xsd_files) >= 1

    def test_discover_markdown_files(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(FIXTURES_DIR)
        md_files = [f for f in files if f.suffix in (".md", ".markdown")]
        assert len(md_files) >= 1

    def test_discover_html_files(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(FIXTURES_DIR)
        html_files = [f for f in files if f.suffix in (".html", ".htm")]
        assert len(html_files) >= 1
