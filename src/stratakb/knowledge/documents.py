"""Stored document text, its versions, and positions inside it.

Each ingested file's extracted text (the parser's pages joined by a blank
line) is stored once per version in strata_vectors.documents. Every chunk
records the span of that text it holds, so text[char_start:char_end] is the
chunk's content, and locate() turns a span into a page and a line range.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import asyncpg
import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument

logger = structlog.get_logger()

# Joins the parser's pages into the document text
PAGE_SEPARATOR = "\n\n"

# Formats whose page numbers are real pages; for the others a "page" is a
# section or a record, so no page map is stored
_PAGED_FORMATS = {".pdf"}


def page_starts(doc: ParsedDocument) -> list[int]:
    """Offset of each parsed page's text in the document text."""
    starts: list[int] = []
    pos = 0
    for page in doc.pages:
        starts.append(pos)
        pos += len(page.text) + len(PAGE_SEPARATOR)
    return starts


def build_document_text(doc: ParsedDocument) -> tuple[str, list[list[int]] | None]:
    """The document text and, for PDFs, its page map [[char_start, page_number], ...].

    Pages the parser dropped (a PDF's empty pages) are absent, so the map
    holds each page's own number, not its index.
    """
    text = PAGE_SEPARATOR.join(page.text for page in doc.pages)
    page_map = None
    if Path(doc.source_file).suffix.lower() in _PAGED_FORMATS:
        starts = page_starts(doc)
        page_map = [
            [start, page.page_number] for start, page in zip(starts, doc.pages, strict=True)
        ]
    return text, page_map


async def store_document_version(
    conn: asyncpg.Connection,
    *,
    layer_id: str,
    file_id: str | None,
    source_root: str,
    relative_path: str,
    text: str,
    page_map: list[list[int]] | None,
    text_origin: str = "text",
) -> tuple[int, int]:
    """Store the text as the file's current version; return (document_id, version).

    The latest version of (layer, source root, relative path), the file's
    identity as in ingested_files, is reused when its text is the same,
    otherwise version + 1 is inserted. Versions are keyed by root and path,
    not by file row, so a layer that was cleared and ingested again continues
    its numbering while older versions are still stored, and same-named files
    from different roots keep separate versions.
    """
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    latest = await conn.fetchrow(
        """
        SELECT id, version, sha256 FROM strata_vectors.documents
        WHERE layer_id = $1 AND source_root = $2 AND relative_path = $3
        ORDER BY version DESC LIMIT 1
        """,
        layer_id,
        source_root,
        relative_path,
    )
    if latest is not None and latest["sha256"] == sha256:
        await conn.execute(
            "UPDATE strata_vectors.documents SET file_id = $2 WHERE id = $1",
            latest["id"],
            file_id,
        )
        return latest["id"], latest["version"]

    version = 1 if latest is None else latest["version"] + 1
    document_id = await conn.fetchval(
        """
        INSERT INTO strata_vectors.documents
            (layer_id, file_id, source_root, relative_path, version, text, sha256,
             page_map, text_origin)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        """,
        layer_id,
        file_id,
        source_root,
        relative_path,
        version,
        text,
        sha256,
        json.dumps(page_map) if page_map is not None else None,
        text_origin,
    )
    logger.info("document_version_stored", relative_path=relative_path, version=version)
    return document_id, version


async def prune_documents(conn: asyncpg.Connection, layer_id: str) -> int:
    """Delete the layer's documents that no chunk references; return the count."""
    result = await conn.execute(
        """
        DELETE FROM strata_vectors.documents d
        WHERE d.layer_id = $1
          AND NOT EXISTS (
              SELECT 1 FROM strata_vectors.chunks c WHERE c.document_id = d.id
          )
        """,
        layer_id,
    )
    return int(result.split()[-1])


def locate(
    text: str,
    page_map: list[list[int]] | str | None,
    start: int,
    end: int,
) -> dict:
    """Page and 1-based line range of the span text[start:end].

    Lines count from the page's start when there is a page map (PDFs), from
    the document's start otherwise; page and page_end are None without one.
    page_map may be the JSON text stored in the documents table.
    """
    if isinstance(page_map, str):
        page_map = json.loads(page_map)
    # The span's last character; an empty span sits at start
    last = max(start, end - 1)

    page = page_end = None
    base = base_end = 0
    if page_map:
        offsets = [entry[0] for entry in page_map]
        i = max(bisect_right(offsets, start) - 1, 0)
        j = max(bisect_right(offsets, last) - 1, 0)
        page, base = page_map[i][1], page_map[i][0]
        page_end, base_end = page_map[j][1], page_map[j][0]

    return {
        "page": page,
        "page_end": page_end,
        "line_start": text.count("\n", base, start) + 1,
        "line_end": text.count("\n", base_end, last) + 1,
    }
