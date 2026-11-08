"""Tests for the date recorded with each stored document version.

The rules for reading a date are pure; the others ingest through the real
pipeline against the database with embeddings mocked.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from structlog.testing import capture_logs

from stratakb.knowledge.document_dates import document_date, frontmatter_date
from tests.helpers import (
    file_outcomes,
    ingest,
    ingest_with_texts,
    locked_read_bytes,
    write_file,
)


@pytest.fixture
def settings():
    from stratakb.config import Settings, get_settings

    return Settings(
        _env_file=None,
        database_url=get_settings().database_url,
        learning_enabled=True,
        reranking_enabled=False,
        strata_user="test-user",
        strata_role="viewer",
    )


@pytest.fixture
def server(settings):
    """A StrataKB server on the test database, its settings patched in everywhere."""
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
        patch("stratakb.learning.tools.write_audit_log", AsyncMock()),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(side_effect=lambda question, **_: [question]),
        ),
    ):
        yield build_server(settings)


@pytest.fixture
async def layer(db_url):
    """A fresh layer, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_dates_{uuid4().hex[:10]}"
    await create_layer(db_url, name=name, display_name="Test Dates")
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


async def _dates(db_url: str, layer: str, relative_path: str) -> dict[int, tuple]:
    """A file's stored versions: {version: (doc_date, doc_date_source)}."""
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        """
        SELECT d.version, d.doc_date, d.doc_date_source FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1 AND d.relative_path = $2
        """,
        layer,
        relative_path,
    )
    return {r["version"]: (r["doc_date"], r["doc_date_source"]) for r in rows}


async def _clear_dates(db_url: str, layer: str) -> None:
    """Make the layer's documents look stored before dates were recorded."""
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    await pool.execute(
        """
        UPDATE strata_vectors.documents SET doc_date = NULL, doc_date_source = NULL
        WHERE layer_id = (SELECT id FROM strata_core.knowledge_layers WHERE name = $1)
        """,
        layer,
    )


async def _chunk_ids(db_url: str, layer: str) -> list:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        "SELECT id FROM strata_vectors.chunks WHERE layer_name = $1 ORDER BY id", layer
    )
    return [r["id"] for r in rows]


def _set_modified(path: Path, year: int, month: int, day: int) -> None:
    moment = datetime(year, month, day, 12, tzinfo=UTC).timestamp()
    os.utime(path, (moment, moment))


def _entries(root: Path, *relative_paths: str) -> list:
    """FileEntry objects for files under root, with their real SHA-256."""
    from stratakb.knowledge.selection import FileEntry

    entries = []
    for rel in relative_paths:
        path = root / rel
        data = path.read_bytes()
        entries.append(FileEntry(
            relative_path=rel,
            size=len(data),
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
        ))
    return entries


async def test_two_runbooks_store_both_kinds_of_date(db_url, layer, tmp_path):
    old = write_file(tmp_path / "failover-2024.md", "# Failover\n\nSwitch by hand.\n")
    _set_modified(old, 2024, 5, 6)
    new = write_file(
        tmp_path / "failover-2026.md",
        "---\ndate: 2026-03-04\n---\n# Failover\n\nSwitch with the script.\n",
    )
    _set_modified(new, 2025, 1, 1)

    await ingest(db_url, layer, tmp_path)

    assert await _dates(db_url, layer, "failover-2024.md") == {1: ("2024-05-06", "file")}
    assert await _dates(db_url, layer, "failover-2026.md") == {
        1: ("2026-03-04", "front_matter")
    }


async def _ask_about_layer(server, db_url: str, layer: str) -> dict:
    """strata_ask with the search patched to return every chunk of the layer."""
    from stratakb.core.database import get_pool
    from stratakb.rag.retriever import SearchResult

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        """
        SELECT id, document_id, char_start, char_end, content, source_file
        FROM strata_vectors.chunks WHERE layer_name = $1 ORDER BY id
        """,
        layer,
    )
    results = [
        SearchResult(
            chunk_id=str(row["id"]),
            content=row["content"],
            score=1.0,
            layer_name=layer,
            source_file=row["source_file"],
            source_section="section",
            doc_type="markdown",
            metadata={},
            similarity=0.9,
            document_id=row["document_id"],
            char_start=row["char_start"],
            char_end=row["char_end"],
        )
        for row in rows
    ]
    with patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=results)):
        result = await server.call_tool("strata_ask", {"question": "q", "layers": [layer]})
    if isinstance(result, tuple):
        result = result[0]
    response = result if isinstance(result, dict) else json.loads(result[0].text)
    if "interaction_id" in response:
        await pool.execute(
            "DELETE FROM strata_learning.interaction_events WHERE id = $1",
            response["interaction_id"],
        )
    return response


async def test_two_runbooks_cite_their_dates(server, db_url, layer, tmp_path):
    old = write_file(tmp_path / "failover-2024.md", "# Failover\n\nSwitch by hand.\n")
    _set_modified(old, 2024, 5, 6)
    new = write_file(
        tmp_path / "failover-2026.md",
        "---\ndate: 2026-03-04\n---\n# Failover\n\nSwitch with the script.\n",
    )
    _set_modified(new, 2025, 1, 1)
    await ingest(db_url, layer, tmp_path)

    response = await _ask_about_layer(server, db_url, layer)

    assert response["status"] == "found"
    dates = {
        passage["citation"]["file"]: (
            passage["citation"]["date"], passage["citation"]["date_source"]
        )
        for passage in response["passages"]
    }
    assert dates == {
        "failover-2024.md": ("2024-05-06", "file"),
        "failover-2026.md": ("2026-03-04", "front_matter"),
    }


async def test_document_without_a_date_cites_null(server, db_url, layer, tmp_path):
    write_file(tmp_path / "a.md", "---\ndate: 2026-03-04\n---\n# A\n\nThe text.\n")
    await ingest(db_url, layer, tmp_path)
    await _clear_dates(db_url, layer)

    response = await _ask_about_layer(server, db_url, layer)

    assert response["status"] != "error"
    citation = response["passages"][0]["citation"]
    assert citation["file"] == "a.md"
    assert citation["lines"] is not None
    assert (citation["date"], citation["date_source"]) == (None, None)


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("date: 2026-03-04", "2026-03-04"),
        ("updated: 2026-05-06\ndate: 2026-03-04", "2026-05-06"),
        ("date: 2026-03-04\nupdated: 2026-05-06", "2026-05-06"),
        ("Date: 2026-03-04", "2026-03-04"),
        ("UPDATED: 2026-05-06", "2026-05-06"),
        ('date: "2026-03-04"', "2026-03-04"),
        ("date: '2026-03-04'", "2026-03-04"),
        ("date: 2026-03-04T10:00:00Z", "2026-03-04"),
        ("date: 2026-03-04 10:00", "2026-03-04"),
        # An invalid value counts as no key, so the other one is used
        ("updated: soon\ndate: 2026-03-04", "2026-03-04"),
        ("updated: 2026-02-30\ndate: 2026-03-04", "2026-03-04"),
        ("date: soon", None),
        ("date: 2026-02-30", None),
        ("date: March 4, 2026", None),
        ("date: 2026-03-041", None),
        ("date: 2026-03-04-draft", None),
        ("date created: 2026-01-01", None),
        ("mydate: 2026-01-01", None),
        ("meta:\n  date: 2026-01-01", None),
        ("title: x", None),
    ],
)
def test_frontmatter_date_rules(block, expected):
    assert frontmatter_date(f"---\n{block}\n---\n# Title\n\nBody.\n") == expected


@pytest.mark.parametrize(
    "text",
    [
        "# Title\n\ndate: 2026-03-04\n",
        "title: x\n---\ndate: 2026-03-04\n---\n",
        # A thematic break, not front matter: the middle line is no key
        "---\nJust a sentence.\ndate: 2026-03-04\n---\n",
        # A date in the body, after front matter without one
        "---\ntitle: x\n---\ndate: 2026-03-04\n",
        "",
    ],
)
def test_frontmatter_date_looks_only_in_front_matter(text):
    assert frontmatter_date(text) is None


@pytest.mark.parametrize(
    "data",
    [
        b"---\r\ndate: 2026-03-04\r\n---\r\n# Title\r\n",
        b"---\rdate: 2026-03-04\r---\r# Title\r",
        b"\xef\xbb\xbf---\ndate: 2026-03-04\n---\n# Title\n",
        b"---\ndate: 2026-03-04\n---\n# Caf\xe9, not UTF-8\n",
    ],
)
def test_crlf_and_byte_order_mark_front_matter(tmp_path, data):
    path = tmp_path / "doc.md"
    path.write_bytes(data)
    assert document_date(path, data) == ("2026-03-04", "front_matter")


def test_binary_format_is_not_decoded(tmp_path):
    data = b"---\ndate: 2026-03-04\n---\n"
    path = tmp_path / "doc.docx"
    path.write_bytes(data)
    _set_modified(path, 2024, 5, 6)
    assert document_date(path, data) == ("2024-05-06", "file")


def test_unreadable_modification_time_gives_no_date(tmp_path):
    missing = tmp_path / "gone.md"
    assert document_date(missing, b"# No front matter\n") == (None, None)


async def test_front_matter_edit_reuses_version_and_updates_date(db_url, layer, tmp_path):
    path = write_file(tmp_path / "a.md", "---\ndate: 2026-03-04\n---\n# A\n\nThe text.\n")
    await ingest(db_url, layer, tmp_path)
    assert await _dates(db_url, layer, "a.md") == {1: ("2026-03-04", "front_matter")}

    write_file(path, "---\ndate: 2026-06-07\n---\n# A\n\nThe text.\n")
    await ingest(db_url, layer, tmp_path)
    assert await _dates(db_url, layer, "a.md") == {1: ("2026-06-07", "front_matter")}


async def test_changed_text_gets_new_version_with_its_own_date(db_url, layer, tmp_path):
    path = write_file(tmp_path / "a.md", "---\ndate: 2026-03-04\n---\n# A\n\nFirst text.\n")
    await ingest(db_url, layer, tmp_path)
    # No prune: the old version stays, as it does while a receipt cites it
    with patch("stratakb.knowledge.ingestion.prune_documents"):
        write_file(path, "---\ndate: 2026-06-07\n---\n# A\n\nSecond text.\n")
        await ingest(db_url, layer, tmp_path)

    assert await _dates(db_url, layer, "a.md") == {
        1: ("2026-03-04", "front_matter"),
        2: ("2026-06-07", "front_matter"),
    }


async def test_storing_without_a_date_keeps_the_stored_one(db_url, layer, tmp_path):
    from stratakb.core.database import get_pool
    from stratakb.knowledge.documents import store_document_version

    write_file(tmp_path / "a.md", "---\ndate: 2026-03-04\n---\n# A\n\nThe text.\n")
    await ingest(db_url, layer, tmp_path)

    pool = await get_pool(db_url)
    row = await pool.fetchrow(
        """
        SELECT d.layer_id, d.source_root, d.text FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1
        """,
        layer,
    )
    async with pool.acquire() as conn:
        await store_document_version(
            conn, layer_id=row["layer_id"], file_id=None, source_root=row["source_root"],
            relative_path="a.md", text=row["text"], page_map=None,
        )
    assert await _dates(db_url, layer, "a.md") == {1: ("2026-03-04", "front_matter")}


async def test_unchanged_file_is_backfilled_without_embedding(db_url, layer, tmp_path):
    path = write_file(tmp_path / "a.md", "---\ndate: 2026-03-04\n---\n# A\n\nFirst text.\n")
    other = write_file(tmp_path / "b.md", "# B\n\nThe other file.\n")
    _set_modified(other, 2024, 5, 6)
    await ingest(db_url, layer, tmp_path)
    # No prune: the old version stays, as it does while a receipt cites it
    with patch("stratakb.knowledge.ingestion.prune_documents"):
        write_file(path, "---\ndate: 2026-06-07\n---\n# A\n\nSecond text.\n")
        await ingest(db_url, layer, tmp_path)
    await _clear_dates(db_url, layer)
    chunks = await _chunk_ids(db_url, layer)

    result, embedded = await ingest_with_texts(db_url, layer, tmp_path)

    assert result.status == "completed"
    assert file_outcomes(result) == {"a.md": ("unchanged", None), "b.md": ("unchanged", None)}
    assert embedded == []
    assert await _chunk_ids(db_url, layer) == chunks
    # Only the latest version is dated: the older one's text is not in the file now
    assert await _dates(db_url, layer, "a.md") == {
        1: (None, None),
        2: ("2026-06-07", "front_matter"),
    }
    assert await _dates(db_url, layer, "b.md") == {1: ("2024-05-06", "file")}


async def test_carried_hash_file_without_a_date_is_backfilled(db_url, layer, tmp_path):
    write_file(tmp_path / "a.md", "---\ndate: 2026-03-04\n---\n# A\n\nThe text.\n")
    entries = _entries(tmp_path, "a.md")
    await ingest(db_url, layer, tmp_path, files=entries)
    await _clear_dates(db_url, layer)

    result, embedded = await ingest_with_texts(db_url, layer, tmp_path, files=entries)

    assert file_outcomes(result) == {"a.md": ("unchanged", None)}
    assert embedded == []
    assert await _dates(db_url, layer, "a.md") == {1: ("2026-03-04", "front_matter")}


async def test_dated_carried_file_is_not_read(db_url, layer, tmp_path):
    write_file(tmp_path / "a.md", "# A\n\nThe text.\n")
    entries = _entries(tmp_path, "a.md")
    await ingest(db_url, layer, tmp_path, files=entries)

    read: list[str] = []
    original_read = Path.read_bytes

    def spy_read(self):
        read.append(self.name)
        return original_read(self)

    with patch.object(Path, "read_bytes", spy_read):
        result = await ingest(db_url, layer, tmp_path, files=entries)

    assert file_outcomes(result) == {"a.md": ("unchanged", None)}
    assert read == []


async def test_backfill_read_failure_leaves_no_date(db_url, layer, tmp_path):
    write_file(tmp_path / "a.md", "---\ndate: 2026-03-04\n---\n# A\n\nThe text.\n")
    entries = _entries(tmp_path, "a.md")
    await ingest(db_url, layer, tmp_path, files=entries)
    await _clear_dates(db_url, layer)

    with (
        patch.object(Path, "read_bytes", locked_read_bytes("a.md")),
        capture_logs() as logs,
    ):
        result = await ingest(db_url, layer, tmp_path, files=entries)

    assert result.status == "completed"
    assert result.errors == []
    assert file_outcomes(result) == {"a.md": ("unchanged", None)}
    assert await _dates(db_url, layer, "a.md") == {1: (None, None)}
    failures = [e for e in logs if e["event"] == "file_read_failed"]
    assert [(e["file"], e["error_type"]) for e in failures] == [("a.md", "PermissionError")]
    assert not any("locked" in str(value) for e in logs for value in e.values())

    # The next ingest tries again, and the front matter date is the one stored
    await ingest(db_url, layer, tmp_path, files=entries)
    assert await _dates(db_url, layer, "a.md") == {1: ("2026-03-04", "front_matter")}
