"""The date recorded with each stored document version, and where it came from.

The date is the front matter's updated or date key when a text format has one,
otherwise the file's modification time. Kept apart from documents.py, which the
retrieval code imports, because reading a date needs the parser registry.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from stratakb.knowledge.parsers.registry import FORMATS
from stratakb.knowledge.structure import frontmatter_end, frontmatter_source

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

logger = structlog.get_logger()

# A front matter key that gives the document's date. A time may follow the
# date; more digits or a hyphen (2026-03-041, 2026-03-04-draft) make it no date.
_DATE_KEY_RE = re.compile(
    r"""^(updated|date)[ \t]*:[ \t]*["']?(\d{4}-\d{2}-\d{2})(?![\d-])""",
    re.ASCII | re.IGNORECASE | re.MULTILINE,
)


def frontmatter_date(text: str) -> str | None:
    """The date (YYYY-MM-DD) in the text's leading front matter, or None.

    Looks at top-level updated and date keys, in any letter case. updated wins:
    the later of the two is the one that tells which document is newer. A value
    that is not a real date is ignored, as if the key were absent. text must
    have \\n line ends.
    """
    found: dict[str, str] = {}
    for match in _DATE_KEY_RE.finditer(text[:frontmatter_end(text)]):
        try:
            date.fromisoformat(match[2])
        except ValueError:
            continue
        found.setdefault(match[1].lower(), match[2])
    return found.get("updated") or found.get("date")


def document_date(file_path: Path, data: bytes) -> tuple[str | None, str | None]:
    """A file's date and where it came from: (YYYY-MM-DD, "front_matter" or "file").

    A text format's front matter comes first, read from the file's bytes
    because the parser strips it from the text. Otherwise the file's
    modification time, as a UTC date; (None, None) when it cannot be read.
    """
    fmt = FORMATS.get(file_path.suffix.lower())
    if fmt is not None and not fmt.binary:
        found = frontmatter_date(frontmatter_source(data))
        if found is not None:
            return found, "front_matter"
    try:
        modified = datetime.fromtimestamp(file_path.stat().st_mtime, UTC)
    except (OSError, OverflowError, ValueError):
        return None, None
    return modified.date().isoformat(), "file"


async def backfill_dates(
    pool: asyncpg.Pool,
    *,
    layer_id: UUID,
    layer_name: str,
    source_root: str,
    unchanged: dict[str, Path],
) -> None:
    """Date the unchanged files whose latest document version has no date yet.

    A version stored before dates were recorded gets one here, since a skipped
    file never reaches the store path; older versions stay as they are, because
    their text is not what the file holds now. Only these files are opened. One
    that cannot be read keeps no date and is tried again by the next ingest: the
    modification time would mark a document with front matter as "file" for good.
    """
    if not unchanged:
        return
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (relative_path) id, relative_path, doc_date_source
        FROM strata_vectors.documents
        WHERE layer_id = $1 AND source_root = $2 AND relative_path = ANY($3::text[])
        ORDER BY relative_path, version DESC
        """,
        layer_id,
        source_root,
        list(unchanged),
    )
    for row in rows:
        if row["doc_date_source"] is not None:
            continue
        relative_path = row["relative_path"]
        file_path = unchanged[relative_path]
        try:
            data = await asyncio.to_thread(file_path.read_bytes)
        except OSError as e:
            logger.warning(
                "file_read_failed", layer=layer_name, file=relative_path,
                error_type=type(e).__name__,
            )
            continue
        doc_date, doc_date_source = await asyncio.to_thread(document_date, file_path, data)
        if doc_date is not None:
            await pool.execute(
                """
                UPDATE strata_vectors.documents SET doc_date = $2, doc_date_source = $3
                WHERE id = $1 AND doc_date IS NULL
                """,
                row["id"],
                doc_date,
                doc_date_source,
            )
