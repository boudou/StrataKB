"""HTML parser that strips tags and chunks by heading sections.

Converts HTML documents into clean text, splitting by heading elements
(h1-h3) into separate pages for searchable chunks.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()


def parse_html(file_path: str | Path) -> ParsedDocument:
    """Parse an HTML file, stripping tags and chunking by headings.

    Args:
        file_path: Path to the HTML file.

    Returns:
        ParsedDocument with one page per heading section.
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
        return ParsedDocument(source_file=source, error="HTML file is empty")

    # Extract title from <title> tag
    title = _extract_title(raw)

    # Parse HTML to structured sections
    parser = _SectionParser()
    parser.feed(raw)
    sections = parser.get_sections()

    if not sections:
        # No headings — return as single page with stripped text
        text = _strip_tags(raw).strip()
        if not text:
            return ParsedDocument(source_file=source, error="HTML file has no text content")
        if title:
            text = f"# {title}\n\n{text}"
        page = ParsedPage(text=text, page_number=1, source_file=source)
        return ParsedDocument(source_file=source, pages=[page], total_pages=1)

    pages: list[ParsedPage] = []
    for i, section in enumerate(sections):
        text = section["text"].strip()
        if text:
            pages.append(ParsedPage(
                text=text,
                page_number=i + 1,
                source_file=source,
            ))

    logger.info(
        "html_parsed",
        file=source,
        sections=len(pages),
    )

    return ParsedDocument(source_file=source, pages=pages, total_pages=len(pages))


class _SectionParser(HTMLParser):
    """HTML parser that splits content by h1-h3 headings."""

    _HEADING_TAGS = {"h1", "h2", "h3"}

    def __init__(self):
        super().__init__()
        self._sections: list[dict] = []
        self._current_text: list[str] = []
        self._current_heading: str = ""
        self._current_level: int = 0
        self._in_heading: bool = False
        self._heading_text: list[str] = []
        self._skip_tags = {"script", "style", "nav", "footer", "header"}
        self._skip_depth: int = 0
        self._h1_context = ""
        self._h2_context = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag = tag.lower()
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

        if tag == "br":
            self._current_text.append("\n")
        elif tag in ("p", "div", "li", "tr"):
            self._current_text.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return

        if tag in self._HEADING_TAGS and self._in_heading:
            self._in_heading = False
            self._current_heading = " ".join(self._heading_text).strip()

            if self._current_level == 1:
                self._h1_context = self._current_heading
                self._h2_context = ""
            elif self._current_level == 2:
                self._h2_context = self._current_heading

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        if self._in_heading:
            self._heading_text.append(data)
        else:
            self._current_text.append(data)

    def _flush_section(self):
        text = "".join(self._current_text).strip()
        if text or self._current_heading:
            lines = []
            if self._current_level >= 2 and self._h1_context:
                lines.append(f"# {self._h1_context}")
            if self._current_level >= 3 and self._h2_context:
                lines.append(f"## {self._h2_context}")
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
                })

        self._current_text = []
        self._current_heading = ""

    def get_sections(self) -> list[dict]:
        """Flush remaining content and return all sections."""
        self._flush_section()
        return self._sections


def _extract_title(html: str) -> str:
    """Extract text from <title> tag."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _strip_tags(html: str) -> str:
    """Remove all HTML tags and return clean text."""
    # Remove script/style content
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
