"""Tests for XML/XSD, Markdown, and HTML parsers.

Tests cover:
- XSD schema parsing with complex types, simple types, elements, groups
- Generic XML fallback
- Markdown splitting by heading hierarchy (H1/H2/H3)
- HTML parsing with tag stripping and section chunking
- Error handling for bad files
- Integration with ingestion pipeline (FORMATS)
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

    def test_h2_has_h1_in_section_path(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        overview = [p for p in result.pages if "## Overview" in p.text]
        assert len(overview) == 1
        assert overview[0].section_path == ["Customer Order Service", "Overview"]
        # The parent heading is on the side, not in the section's text
        assert "# Customer Order" not in overview[0].text

    def test_h3_has_h1_and_h2_in_section_path(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        kc = [p for p in result.pages if "### Key Capabilities" in p.text]
        assert len(kc) == 1
        assert kc[0].section_path == ["Customer Order Service", "Overview", "Key Capabilities"]
        assert "# Customer Order" not in kc[0].text
        assert "## Overview" not in kc[0].text

    def test_sections_are_slices_of_the_document(self):
        """Each section's text is the document's own text: own heading once, no parent headings."""
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        source = TEST_MD.read_text(encoding="utf-8")
        result = parse_markdown(TEST_MD)
        for page in result.pages:
            assert page.text in source
            heading_lines = [ln for ln in page.text.splitlines() if ln.startswith("#")]
            assert len(heading_lines) == 1
            assert heading_lines[0].lstrip("#").strip() == page.section_path[-1]

    def test_h1_section_path(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        result = parse_markdown(TEST_MD)
        assert result.pages[0].section_path == ["Customer Order Service"]

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

    @staticmethod
    def _parse(tmp_path, source: str):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        path = tmp_path / "doc.md"
        path.write_text(source, encoding="utf-8")
        result = parse_markdown(path)
        assert result.error is None
        for page in result.pages:
            assert page.text in source
        return result

    def test_hash_lines_in_backtick_fence_are_not_headings(self, tmp_path):
        source = (
            "# Install\n\nRun this:\n\n"
            "```bash\n# install deps\nuv sync\n```\n"
            "\nDone.\n\n## Next\n\ntext\n"
        )
        result = self._parse(tmp_path, source)
        assert result.total_pages == 2
        first, second = result.pages
        assert "```bash\n# install deps\nuv sync\n```" in first.text
        assert "Done." in first.text
        assert first.section_path == ["Install"]
        assert second.text.startswith("## Next")
        assert second.section_path == ["Install", "Next"]

    def test_hash_lines_in_tilde_fence_are_not_headings(self, tmp_path):
        source = (
            "# Install\n\n~~~\n## not a heading\n```\n# still code\n~~~\n"
            "\n## Next\n\ntext\n"
        )
        result = self._parse(tmp_path, source)
        assert result.total_pages == 2
        assert "## not a heading" in result.pages[0].text
        assert "# still code" in result.pages[0].text
        assert result.pages[0].section_path == ["Install"]
        assert result.pages[1].section_path == ["Install", "Next"]

    def test_longer_fence_is_not_closed_by_shorter_one(self, tmp_path):
        source = "# A\n\n````\n```\n# code\n```\n````\n\n## B\n\ntext\n"
        result = self._parse(tmp_path, source)
        assert [p.section_path for p in result.pages] == [["A"], ["A", "B"]]
        assert "# code" in result.pages[0].text

    def test_unclosed_fence_runs_to_end(self, tmp_path):
        source = "# A\n\n```\n# code\n\n## Not a heading either\n"
        result = self._parse(tmp_path, source)
        assert result.total_pages == 1
        assert result.pages[0].section_path == ["A"]
        assert "## Not a heading either" in result.pages[0].text

    def test_inline_triple_backticks_do_not_open_fence(self, tmp_path):
        source = "# A\n\n```x```\n\n## B\n\ntext\n"
        result = self._parse(tmp_path, source)
        assert [p.section_path for p in result.pages] == [["A"], ["A", "B"]]

    def test_frontmatter_comment_is_not_heading(self, tmp_path):
        source = "---\ntitle: x\n# a yaml comment\n---\n\n# Real\n\ntext\n"
        result = self._parse(tmp_path, source)
        assert result.total_pages == 1
        assert result.pages[0].section_path == ["Real"]
        assert result.pages[0].text.startswith("# Real")

    def test_frontmatter_text_before_first_heading_is_preamble(self, tmp_path):
        source = "---\ntitle: x\n...\nIntro text.\n\n# Real\n\ntext\n"
        result = self._parse(tmp_path, source)
        assert [p.text for p in result.pages] == ["Intro text.", "# Real\n\ntext"]

    def test_leading_thematic_break_is_not_frontmatter(self, tmp_path):
        source = (
            "---\n\n# Intro\n\nSome text\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
            "\n## Next\n\nmore\n"
        )
        result = self._parse(tmp_path, source)
        sections = [(p.section_path, p.text) for p in result.pages]
        assert sections[0] == ([], "---")
        assert sections[1][0] == ["Intro"]
        assert sections[1][1].startswith("# Intro\n\nSome text\n\n| a | b |\n|---|---|")
        assert sections[2] == (["Intro", "Next"], "## Next\n\nmore")

    def test_dashes_inside_a_line_do_not_close_frontmatter(self, tmp_path):
        source = "---\ntitle: Wait --- what\n# yaml comment\n---\n\n# Real\n\ntext\n"
        result = self._parse(tmp_path, source)
        assert [p.section_path for p in result.pages] == [["Real"]]
        assert result.pages[0].text == "# Real\n\ntext"

    def test_bare_hash_line_does_not_take_next_line(self, tmp_path):
        source = "# A\n\n#\nplain line\n\n## B\n\ntext\n"
        result = self._parse(tmp_path, source)
        assert [p.section_path for p in result.pages] == [["A"], ["A", "B"]]
        assert "plain line" in result.pages[0].text

    def test_frontmatter_left_out_without_headings(self, tmp_path):
        result = self._parse(tmp_path, "---\ntitle: x\n---\nJust text.\n")
        assert [p.text for p in result.pages] == ["Just text."]

    def test_frontmatter_only_file_has_no_pages(self, tmp_path):
        result = self._parse(tmp_path, "---\ntitle: x\n---\n")
        assert result.pages == []
        assert result.total_pages == 0

    def test_thematic_break_then_text_is_not_frontmatter(self, tmp_path):
        source = "---\n# Intro\ntext\n\n---\n\n## Next\nmore\n"
        result = self._parse(tmp_path, source)
        assert [(p.section_path, p.text) for p in result.pages] == [
            ([], "---"),
            (["Intro"], "# Intro\ntext\n\n---"),
            (["Intro", "Next"], "## Next\nmore"),
        ]

    def test_frontmatter_may_start_with_comment(self, tmp_path):
        source = "---\n# comment\ntitle: x\n---\n\n# Real\n\ntext\n"
        result = self._parse(tmp_path, source)
        assert [p.section_path for p in result.pages] == [["Real"]]
        assert result.pages[0].text == "# Real\n\ntext"

    def test_key_then_prose_is_not_frontmatter(self, tmp_path):
        result = self._parse(tmp_path, "---\nNote: x\nprose\n---\n")
        all_text = "\n".join(p.text for p in result.pages)
        assert "Note: x" in all_text
        assert "prose" in all_text

    def test_sentence_with_colon_is_frontmatter(self, tmp_path):
        # Pinned on purpose: a front-matter key is anything up to the first ": ",
        # so a leading block of such lines is dropped as front matter
        result = self._parse(tmp_path, "---\nStep 1: install it\n---\n# H\n")
        assert [(p.section_path, p.text) for p in result.pages] == [(["H"], "# H")]

    def test_frontmatter_key_with_spaces(self, tmp_path):
        source = "---\ntitle: x\ndate created: 2024\n---\n# H\n\nbody\n"
        result = self._parse(tmp_path, source)
        assert [(p.section_path, p.text) for p in result.pages] == [(["H"], "# H\n\nbody")]

    def test_setext_headings(self, tmp_path):
        source = "Title\n=====\n\nintro\n\nPart\n----\n\nbody\n"
        result = self._parse(tmp_path, source)
        assert [p.section_path for p in result.pages] == [["Title"], ["Title", "Part"]]
        assert result.pages[0].text.startswith("Title\n=====")
        assert "intro" in result.pages[0].text
        assert result.pages[1].text.startswith("Part\n----")

    @pytest.mark.parametrize("source", [
        "# A\n\ntext\n\n---\n\nmore\n",
        "# A\n\n- item\n---\n",
        "# A\n\n```\nCode\n---\n```\n",
        "# A\n\n> quote\n---\n",
    ])
    def test_underline_after_blank_list_or_fence_is_not_heading(self, tmp_path, source):
        result = self._parse(tmp_path, source)
        assert [p.section_path for p in result.pages] == [["A"]]

    def test_atx_heading_indented_up_to_three_spaces(self, tmp_path):
        result = self._parse(tmp_path, "   # A\n\ntext\n\n    # code\n")
        assert [p.section_path for p in result.pages] == [["A"]]
        assert "    # code" in result.pages[0].text

    def test_frontmatter_after_byte_order_mark(self, tmp_path):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        path = tmp_path / "bom.md"
        path.write_text("---\ntitle: x\n---\nJust text.\n", encoding="utf-8-sig")
        result = parse_markdown(path)
        assert result.error is None
        assert [p.text for p in result.pages] == ["Just text."]


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
            assert result.error is None
            assert result.pages == []
            assert result.total_pages == 0

        Path(f.name).unlink()

    def test_whitespace_only_markdown(self):
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("  \n\t\n")
            f.flush()

            result = parse_markdown(f.name)
            assert result.error is None
            assert result.pages == []
            assert result.total_pages == 0

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
        # H1: SEPA Credit Transfer + H2: Message Format + H3: Group Header
        # + H3: Payment Information + H2: Processing Rules + H3: Validation = 6
        # (the <title> is not section text)
        assert result.total_pages == 6
        assert result.pages[0].text.startswith("# SEPA Credit Transfer")
        assert all("SEPA Credit Transfer Guide" not in p.text for p in result.pages)

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
        h1_page = result.pages[0]
        assert "SEPA Credit Transfer" in h1_page.text
        assert "euro-denominated" in h1_page.text

    def test_h2_has_h1_in_section_path(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        mf = [p for p in result.pages if p.text.startswith("## Message Format")]
        assert len(mf) == 1
        assert mf[0].section_path == ["SEPA Credit Transfer", "Message Format"]
        assert "SEPA Credit Transfer" not in mf[0].text

    def test_h3_has_parents_in_section_path(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        gh = [p for p in result.pages if p.text.startswith("### Group Header")]
        assert len(gh) == 1
        assert gh[0].section_path == ["SEPA Credit Transfer", "Message Format", "Group Header"]
        assert "SEPA Credit Transfer" not in gh[0].text
        assert "Message Format" not in gh[0].text

    def test_each_section_has_its_own_heading_once(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        result = parse_html(TEST_HTML)
        for page in result.pages:
            heading_lines = [ln for ln in page.text.splitlines() if ln.startswith("#")]
            assert len(heading_lines) == 1
            assert heading_lines[0].lstrip("#").strip() == page.section_path[-1]

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
            assert result.pages[0].text == "Simple content"
            assert result.pages[0].section_path == ["Test"]

        Path(f.name).unlink()

    def test_title_only_in_nav_page_not_in_text(self, tmp_path):
        from stratakb.knowledge.parsers.html_parser import parse_html

        path = tmp_path / "nav.html"
        path.write_text(
            "<html><head><title>My Page</title></head><body><nav>only nav</nav></body></html>",
            encoding="utf-8",
        )
        result = parse_html(path)
        assert result.error is None
        assert result.total_pages == 1
        assert result.pages[0].text == "only nav"
        assert result.pages[0].section_path == ["My Page"]

    def test_title_entities_unescaped(self, tmp_path):
        from stratakb.knowledge.parsers.html_parser import parse_html

        path = tmp_path / "amp.html"
        path.write_text(
            "<html><head><title>T &amp;\n  U</title></head><body><p>body</p></body></html>",
            encoding="utf-8",
        )
        result = parse_html(path)
        assert result.pages[0].text == "body"
        assert result.pages[0].section_path == ["T & U"]

    def test_text_before_first_heading_gets_title_path(self, tmp_path):
        from stratakb.knowledge.parsers.html_parser import parse_html

        path = tmp_path / "intro.html"
        path.write_text(
            "<html><head><title>Guide</title></head><body><p>Intro</p><h1>Main</h1><p>x</p></body></html>",
            encoding="utf-8",
        )
        result = parse_html(path)
        assert [p.text for p in result.pages] == ["Intro", "# Main\n\nx"]
        assert [p.section_path for p in result.pages] == [["Guide"], ["Main"]]

    @staticmethod
    def _parse(tmp_path, source: str):
        from stratakb.knowledge.parsers.html_parser import parse_html

        path = tmp_path / "page.html"
        path.write_text(source, encoding="utf-8")
        result = parse_html(path)
        assert result.error is None
        return result

    def test_svg_title_in_body_is_text(self, tmp_path):
        result = self._parse(
            tmp_path,
            "<html><head><title>Page</title></head><body><h1>A</h1><p>x</p>"
            "<svg><title>Sales chart</title></svg></body></html>",
        )
        assert result.total_pages == 1
        assert "Sales chart" in result.pages[0].text
        assert "Page" not in result.pages[0].text
        assert result.pages[0].section_path == ["A"]

    def test_svg_title_is_not_page_title(self, tmp_path):
        result = self._parse(
            tmp_path,
            "<html><body><p>Intro</p><svg><title>Logo</title></svg><h1>A</h1><p>x</p></body></html>",
        )
        assert result.pages[0].section_path == []
        assert "Logo" in result.pages[0].text
        assert "Intro" in result.pages[0].text
        assert result.pages[1].section_path == ["A"]

    def test_svg_title_is_not_glued_to_neighbours(self, tmp_path):
        result = self._parse(tmp_path, "<p>x<svg><title>Chart</title></svg>y</p>")
        assert result.pages[0].text == "x\nChart\ny"

    def test_svg_title_is_not_heading_text(self, tmp_path):
        result = self._parse(
            tmp_path,
            "<h1><svg><title>link icon</title></svg>Install</h1><p>Run it."
            "<svg><title>Diagram</title></svg></p>",
        )
        assert [p.section_path for p in result.pages] == [["Install"]]
        assert result.pages[0].text.startswith("# Install\n\n")
        assert "link icon" not in result.pages[0].text
        assert "Diagram" in result.pages[0].text

    def test_unclosed_svg_title_in_heading_ends_at_svg_end(self, tmp_path):
        result = self._parse(
            tmp_path,
            "<h1><svg><title>x</svg>Install</h1><p>Step one.</p>"
            "<h2>Next</h2><p>Step two.</p>",
        )
        assert [(p.section_path, p.text) for p in result.pages] == [
            (["Install"], "# Install\n\nStep one."),
            (["Install", "Next"], "## Next\n\nStep two."),
        ]

    def test_unclosed_svg_title_ends_at_heading_end(self, tmp_path):
        result = self._parse(
            tmp_path, "<h1>Install<svg><title>icon</h1><p>Step one.</p><h2>Next</h2><p>two</p>"
        )
        assert [(p.section_path, p.text) for p in result.pages] == [
            (["Install"], "# Install\n\nStep one."),
            (["Install", "Next"], "## Next\n\ntwo"),
        ]

    def test_unclosed_svg_title_in_body_is_text(self, tmp_path):
        result = self._parse(tmp_path, "<p>x<svg><title>Chart</svg>y</p><h1>A</h1><p>z</p>")
        assert [(p.section_path, p.text) for p in result.pages] == [
            ([], "x\nChart\ny"),
            (["A"], "# A\n\nz"),
        ]

    def test_title_without_head_element(self, tmp_path):
        result = self._parse(tmp_path, "<!doctype html><title>Doc</title><p>text</p>")
        assert [(p.section_path, p.text) for p in result.pages] == [(["Doc"], "text")]

    def test_unclosed_title_gives_one_untitled_page(self, tmp_path):
        result = self._parse(
            tmp_path, "<html><head><title>Broken</head><body><p>text</p></body></html>"
        )
        assert result.total_pages == 1
        assert result.pages[0].section_path == []


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
            assert result.error is None
            assert result.pages == []
            assert result.total_pages == 0

        Path(f.name).unlink()

    def test_html_without_text(self):
        from stratakb.knowledge.parsers.html_parser import parse_html

        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
            f.write("<html><body><p> </p></body></html>")
            f.flush()

            result = parse_html(f.name)
            assert result.error is None
            assert result.pages == []
            assert result.total_pages == 0

        Path(f.name).unlink()


# ==========================================
# Pipeline integration tests
# ==========================================


class TestPipelineIntegration:
    def test_xml_registered_in_formats(self):
        from stratakb.knowledge.parsers.registry import FORMATS

        assert ".xml" in FORMATS
        assert ".xsd" in FORMATS

    def test_markdown_registered_in_formats(self):
        from stratakb.knowledge.parsers.registry import FORMATS

        assert ".md" in FORMATS
        assert ".markdown" in FORMATS

    def test_html_registered_in_formats(self):
        from stratakb.knowledge.parsers.registry import FORMATS

        assert ".html" in FORMATS
        assert ".htm" in FORMATS

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
