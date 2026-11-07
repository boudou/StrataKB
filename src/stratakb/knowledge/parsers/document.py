"""The contract every parser returns: a parsed document, its pages and its flag."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from stratakb.knowledge.structure import Heading

FlagReason = Literal["probably_scanned", "truncated"]


@dataclass(frozen=True)
class ParserFlag:
    """Why a parsed file is stored but flagged, and how many items a limit left
    out (None when nothing was counted)."""

    reason: FlagReason
    count: int | None = None


@dataclass
class ParsedPage:
    """One piece of a parsed file: a PDF page, or the unit another format's
    parser yields (a Markdown section, a CSV row)."""

    text: str
    page_number: int
    source_file: str
    # Heading chain of a section, outermost first (Markdown and HTML); empty for other formats
    section_path: list[str] = field(default_factory=list)
    # Headings the file marks outside its text (a PDF outline, Word heading
    # styles), in offset order; empty for other formats
    headings: list[Heading] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Result of parsing one file."""

    source_file: str
    pages: list[ParsedPage] = field(default_factory=list)
    total_pages: int = 0
    error: str | None = None
    # Set by a parser that stored only part of a file or found it probably scanned; None otherwise
    flag: ParserFlag | None = None
