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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import asyncpg
import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument
from stratakb.knowledge.structure import Heading, Section

logger = structlog.get_logger()

# Joins the parser's pages into the document text
PAGE_SEPARATOR = "\n\n"

# Formats whose page numbers are real pages; for the others a "page" is a
# section or a record, so no page map is stored
_PAGED_FORMATS = {".pdf"}

# First key of the two-key advisory lock that serializes version numbering
# per file; Postgres keeps it apart from the one-key migration lock
_VERSION_LOCK_NAMESPACE = 0x53544B44


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


def document_headings(doc: ParsedDocument) -> list[Heading]:
    """The headings the parser read outside the text (a PDF outline, Word
    heading styles), with their offsets in the document text.

    Offsets are those of build_document_text's text; the list is empty for
    formats whose parser reports none.
    """
    return [
        heading._replace(offset=start + heading.offset)
        for start, page in zip(page_starts(doc), doc.pages, strict=True)
        for heading in page.headings
    ]


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

    Numbering is serialized per file: a transaction-scoped advisory lock on
    (layer, source root, relative path) makes a concurrent call for the same
    file wait until the first caller's transaction ends, and then read the
    version it committed. The function opens its own transaction (a savepoint
    inside the caller's), so the lock holds until the outermost one ends.
    A version that a concurrent prune deletes while this call waits on it is
    not reused: the call reads the latest version again.
    """
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    async with conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock($1, $2)",
            _VERSION_LOCK_NAMESPACE,
            _version_lock_key(layer_id, source_root, relative_path),
        )
        while True:
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
            if latest is None or latest["sha256"] != sha256:
                break
            # No row back: a prune deleted this version while we waited on it
            if await conn.fetchval(
                "UPDATE strata_vectors.documents SET file_id = $2 WHERE id = $1 RETURNING id",
                latest["id"],
                file_id,
            ) is not None:
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


async def store_sections(
    conn: asyncpg.Connection, document_id: int, sections: Sequence[Section]
) -> None:
    """Store a document version's structure map, unless it already has one.

    A reused version keeps the sections stored with it (its text, and so its
    map, is the same); a version stored before the map existed gets one.
    """
    if not sections:
        return
    if await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM strata_vectors.document_sections WHERE document_id = $1)",
        document_id,
    ):
        return
    await conn.executemany(
        """
        INSERT INTO strata_vectors.document_sections
            (document_id, label, title, path, depth, char_start, char_end)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        [
            (document_id, s.label, s.title, s.path, s.depth, s.char_start, s.char_end)
            for s in sections
        ],
    )


def _version_lock_key(layer_id: str, source_root: str, relative_path: str) -> int:
    """One file's advisory lock key: signed 32 bits, the same in every process."""
    identity = f"{layer_id}\x00{source_root}\x00{relative_path}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big", signed=True)


@dataclass(frozen=True)
class FileScope:
    """Some files of one layer: their source root and their paths relative to it."""

    source_root: str
    relative_paths: Sequence[str]


async def prune_documents(
    conn: asyncpg.Connection,
    layer_ids: Sequence[str | UUID],
    *,
    files: FileScope | None = None,
) -> int:
    """Delete documents that no chunk and no receipt references; return the count.

    Without files the whole of each layer in layer_ids is swept; with them,
    only those files' versions, and layer_ids must then hold exactly one layer
    (a file belongs to one). Several layers are swept in one statement, so
    their versions are locked in one id order across layers, the order a
    receipt locks the versions it cites (sweeping them one by one could
    deadlock).

    Two statements, because a lone DELETE that waits on an uncommitted
    reference still deletes once it commits: its checks keep the snapshot taken
    before the wait, and the foreign key's cascade would remove the new
    reference. So the candidates are locked first, in id order (FOR UPDATE
    conflicts with the FOR KEY SHARE a reference takes), and the delete checks
    again with a new snapshot. Nothing can start referencing a locked row in
    between. The function opens its own transaction (a savepoint inside the
    caller's), so the locks hold until the outermost one ends.
    """
    if files is not None and len(layer_ids) != 1:
        raise ValueError("A file scope needs exactly one layer")
    scope = ""
    args: list = [list(layer_ids)]
    if files is not None:
        scope = "AND d.source_root = $2 AND d.relative_path = ANY($3::text[])"
        args += [files.source_root, list(files.relative_paths)]
    unreferenced = """
          AND NOT EXISTS (
              SELECT 1 FROM strata_vectors.chunks c WHERE c.document_id = d.id
          )
          AND NOT EXISTS (
              SELECT 1 FROM strata_core.receipt_documents r WHERE r.document_id = d.id
          )
    """
    async with conn.transaction():
        rows = await conn.fetch(
            f"""
            SELECT d.id FROM strata_vectors.documents d
            WHERE d.layer_id = ANY($1::uuid[]) {scope} {unreferenced}
            ORDER BY d.id
            FOR UPDATE OF d
            """,
            *args,
        )
        if not rows:
            return 0
        result = await conn.execute(
            f"""
            DELETE FROM strata_vectors.documents d
            WHERE d.id = ANY($1::bigint[]) {unreferenced}
            """,
            [row["id"] for row in rows],
        )
    return int(result.split()[-1])


async def fetch_cited_documents(
    conn: asyncpg.Connection, ends: dict[int, int]
) -> dict[int, dict]:
    """Path, page map and text up to the given end of each cited document.

    ends maps a document id to the furthest char_end cited in it; only the
    text before it is read, which is all locate() needs.
    """
    if not ends:
        return {}
    rows = await conn.fetch(
        """
        SELECT d.id, d.relative_path, d.page_map, left(d.text, x.max_end) AS text
        FROM strata_vectors.documents d
        JOIN unnest($1::bigint[], $2::int[]) AS x(id, max_end) ON x.id = d.id
        """,
        list(ends),
        list(ends.values()),
    )
    return {row["id"]: dict(row) for row in rows}


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
