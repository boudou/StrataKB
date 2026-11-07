"""HTML parser that strips tags and chunks by heading sections.

Converts HTML documents into clean text, splitting by heading elements
(h1-h3) into separate pages for searchable chunks.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.document import ParsedDocument, ParsedPage

logger = structlog.get_logger()


def parse_html(file_path: str | Path) -> ParsedDocument:
    """Parse an HTML file, stripping tags and chunking by headings.

    Args:
        file_path: Path to the HTML file.

    Returns:
        ParsedDocument with one page per heading section, or no pages when there is no text.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in (".html", ".htm"):
        return ParsedDocument(source_file=source, error=f"Not an HTML file: {file_path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return ParsedDocument(source_file=source, error=f"Read error: {e}")

    if not raw.strip():
        return ParsedDocument(source_file=source, pages=[], total_pages=0)

    # Parse HTML to structured sections. No close(): an unclosed <title> would
    # then pour the rest of the page into the title
    parser = _SectionParser()
    parser.feed(raw)
    sections = parser.get_sections()
    title = parser.title

    if not sections:
        # No section text — fall back to one untitled-heading page of stripped text
        text = _strip_tags(raw).strip()
        if not text:
            return ParsedDocument(source_file=source, pages=[], total_pages=0)
        sections = [{"level": 0, "heading": "", "text": text, "section_path": []}]

    pages: list[ParsedPage] = []
    for i, section in enumerate(sections):
        text = section["text"].strip()
        if text:
            # Text under no heading has the page title as its section path
            untitled = [title] if title else []
            section_path = section["section_path"] if section["level"] else untitled
            pages.append(ParsedPage(
                text=text,
                page_number=i + 1,
                source_file=source,
                section_path=section_path,
            ))

    logger.info(
        "html_parsed",
        file=source,
        sections=len(pages),
    )

    return ParsedDocument(source_file=source, pages=pages, total_pages=len(pages))


class _SectionParser(HTMLParser):
    """HTML parser that splits content by h1-h3 headings.

    The page title (the first <title> outside an <svg>) is not section text; it
    is kept in self.title. A <title> inside an <svg> is ordinary text, except
    inside a heading, where it is left out of the heading text (an icon's label).
    An SVG title left open ends at </svg> or at the end of its heading.
    """

    _HEADING_TAGS = {"h1", "h2", "h3"}

    def __init__(self):
        super().__init__()
        self.title: str = ""
        self._sections: list[dict] = []
        self._current_text: list[str] = []
        self._current_heading: str = ""
        self._current_level: int = 0
        self._in_heading: bool = False
        self._heading_text: list[str] = []
        self._skip_tags = {"script", "style", "nav", "footer", "header"}
        self._skip_depth: int = 0
        self._svg_depth: int = 0
        self._in_page_title: bool = False
        self._in_svg_title: bool = False
        self._title_parts: list[str] = []
        self._h1_context = ""
        self._h2_context = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag = tag.lower()
        # Count <svg> nesting before the skip check, so it stays right inside skipped tags
        if tag == "svg":
            self._svg_depth += 1
        elif tag == "title" and self._svg_depth == 0:
            self._in_page_title = True
            self._title_parts = []
            return
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        if tag in self._HEADING_TAGS:
            # Flush current section
            self._flush_section()
            self._in_heading = True
            self._heading_text = []
            self._current_level = int(tag[1])

        if tag == "title":
            self._in_svg_title = True
            self._current_text.append("\n")
        elif tag in ("br", "p", "div", "li", "tr"):
            self._current_text.append("\n")

    def set_cdata_mode(self, elem: str, **kwargs):
        # A <title> inside an <svg> holds markup, as in browsers. Read as raw text,
        # an unclosed one would swallow the rest of the page up to a </title>
        if elem.lower() == "title" and self._svg_depth > 0:
            return
        super().set_cdata_mode(elem, **kwargs)

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == "svg":
            self._svg_depth = max(0, self._svg_depth - 1)
            # </svg> also ends an SVG title left open
            if self._in_svg_title and self._skip_depth == 0:
                self._in_svg_title = False
                self._current_text.append("\n")
        elif tag == "title" and self._in_page_title:
            self._in_page_title = False
            if not self.title:
                self.title = re.sub(r"\s+", " ", "".join(self._title_parts)).strip()
            return
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return

        if tag == "title":
            # An SVG title is text: keep it apart from the text after it
            self._in_svg_title = False
            self._current_text.append("\n")
        elif tag in self._HEADING_TAGS and self._in_heading:
            self._in_heading = False
            self._in_svg_title = False
            self._current_heading = " ".join(self._heading_text).strip()

            if self._current_level == 1:
                self._h1_context = self._current_heading
                self._h2_context = ""
            elif self._current_level == 2:
                self._h2_context = self._current_heading

    def handle_data(self, data: str):
        if self._in_page_title:
            self._title_parts.append(data)
            return
        if self._skip_depth > 0:
            return
        if self._in_heading and self._in_svg_title:
            # An icon's label is not heading text
            return
        (self._heading_text if self._in_heading else self._current_text).append(data)

    def _flush_section(self):
        text = "".join(self._current_text).strip()
        if text or self._current_heading:
            # Parent headings go into section_path, not into the section's text
            # (text under no heading gets its path, the page title, in parse_html)
            path = []
            if self._current_level >= 2 and self._h1_context:
                path.append(self._h1_context)
            if self._current_level >= 3 and self._h2_context:
                path.append(self._h2_context)
            if self._current_heading:
                path.append(self._current_heading)

            lines = []
            if self._current_heading:
                prefix = "#" * self._current_level if self._current_level > 0 else "#"
                lines.append(f"{prefix} {self._current_heading}")
            if text:
                lines.append(text)

            if lines:
                self._sections.append({
                    "level": self._current_level,
                    "heading": self._current_heading,
                    "text": "\n\n".join(lines),
                    "section_path": path,
                })

        self._current_text = []
        self._current_heading = ""

    def get_sections(self) -> list[dict]:
        """Flush remaining content and return all sections."""
        self._flush_section()
        return self._sections


def _strip_tags(html: str) -> str:
    """Remove all HTML tags and return clean text."""
    # Remove script/style/title content
    html = re.sub(r"<(script|style|title)[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
