"""Tests for stored document text, versions and chunk offsets.

The locate and text-building tests are pure; the others ingest through the
real pipeline against the database with embeddings mocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.helpers import wait_until_blocked_by

FIXTURES =Path(__file__).parent / "fixtures"
FAKE_EMBEDDING = [0.01] * 768


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def layer(db_url):
    """A fresh layer with small, overlapping chunks, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_docs_{uuid4().hex[:10]}"
    await create_layer(
        db_url,
        name=name,
        display_name="Test Documents",
        chunk_config={"chunk_size": 100, "chunk_overlap": 40},
    )
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


async def _ingest(db_url: str, layer: str, source: Path, **kwargs):
    from stratakb.knowledge.ingestion import ingest_documents

    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda texts, **_: [FAKE_EMBEDDING for _ in texts]
        return await ingest_documents(
            db_url, "fake-key", layer_name=layer, source_path=str(source), **kwargs
        )


async def _documents(db_url: str, layer: str) -> list:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetch(
        """
        SELECT d.id, d.relative_path, d.version, d.text, d.sha256, d.page_map, d.text_origin
        FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1
        ORDER BY d.relative_path, d.version
        """,
        layer,
    )


async def _chunks(db_url: str, layer: str) -> list:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetch(
        """
        SELECT c.content, c.metadata, c.char_start, c.char_end, c.document_id, c.source_file
        FROM strata_vectors.chunks c
        WHERE c.layer_name = $1
        ORDER BY c.source_file, c.char_start
        """,
        layer,
    )


async def _assert_offsets_match(db_url: str, layer: str) -> dict:
    """Every chunk's span of its document text is its content; return docs by path."""
    docs = {d["id"]: d for d in await _documents(db_url, layer)}
    chunks = await _chunks(db_url, layer)
    assert chunks
    for c in chunks:
        assert c["document_id"] in docs, c["source_file"]
        assert c["char_start"] is not None and c["char_end"] is not None, c["source_file"]
        text = docs[c["document_id"]]["text"]
        assert text[c["char_start"]:c["char_end"]] == c["content"], c["source_file"]
    return {d["relative_path"]: d for d in docs.values()}


async def _layer_id(db_url: str, layer: str):
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetchval(
        "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", layer
    )


async def _store_bare_document(
    db_url: str, layer: str, source_root: str, relative_path: str, text: str
) -> None:
    """Store a document version that no chunk references."""
    from stratakb.core.database import get_pool
    from stratakb.knowledge.documents import store_document_version

    layer_id = await _layer_id(db_url, layer)
    pool = await get_pool(db_url)
    async with pool.acquire() as conn, conn.transaction():
        await store_document_version(
            conn, layer_id=layer_id, file_id=None, source_root=source_root,
            relative_path=relative_path, text=text, page_map=None,
        )


async def _file_rows(db_url: str, layer: str) -> dict:
    """The layer's ingested_files ids, keyed by relative path."""
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        """
        SELECT f.relative_path, f.id FROM strata_core.ingested_files f
        JOIN strata_core.knowledge_layers kl ON kl.id = f.layer_id
        WHERE kl.name = $1
        """,
        layer,
    )
    return {r["relative_path"]: r["id"] for r in rows}


def _copy(name: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy(FIXTURES / name, dest / name))


class TestOffsets:
    async def test_offsets_match_markdown(self, db_url, layer, tmp_path):
        _copy("test_document.md", tmp_path)
        result = await _ingest(db_url, layer, tmp_path)
        assert result.status == "completed"

        docs = await _assert_offsets_match(db_url, layer)
        assert docs["test_document.md"]["page_map"] is None

    async def test_offsets_match_repeated_overlapping_text(self, db_url, layer, tmp_path):
        # The same sentence many times: each overlapping chunk's text occurs at
        # several places, and the running search must pick the one after the last
        (tmp_path / "loop.md").write_text(
            "# Loop\n\n" + "Retry the failover drill. " * 150 + "\n", encoding="utf-8"
        )
        await _ingest(db_url, layer, tmp_path)

        docs = await _assert_offsets_match(db_url, layer)
        spans = [(c["char_start"], c["char_end"]) for c in await _chunks(db_url, layer)]
        assert len(spans) > 3
        # Each chunk starts after the last and leaves no gap (only the blank
        # line that splitting strips), so the spans cover the whole text
        for prev, nxt in zip(spans, spans[1:], strict=False):
            assert prev[0] < nxt[0] <= prev[1] + 2
        assert spans[-1][1] == len(docs["loop.md"]["text"])

    async def test_offsets_match_pdf(self, db_url, layer, tmp_path):
        _copy("test_doc.pdf", tmp_path)
        result = await _ingest(db_url, layer, tmp_path)
        assert result.status == "completed"

        docs = await _assert_offsets_match(db_url, layer)
        page_map = json.loads(docs["test_doc.pdf"]["page_map"])
        assert [page for _, page in page_map] == [1, 2]
        assert page_map[0][0] == 0

    async def test_offsets_match_every_parser(self, db_url, layer, tmp_path):
        for name in (
            "test_document.md", "test_doc.pdf", "test_page.html", "test_schema.xsd",
            "test_openapi.yaml", "reason_codes.csv",
        ):
            _copy(name, tmp_path)
        (tmp_path / "notes.txt").write_text(
            "Plain notes.\n\nA second paragraph about restores.\n", encoding="utf-8"
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"service": "orders", "retries": 3}), encoding="utf-8"
        )
        result = await _ingest(db_url, layer, tmp_path)
        assert result.status == "completed"

        docs = await _assert_offsets_match(db_url, layer)
        assert len(docs) == 8

    async def test_text_origin_is_text(self, db_url, layer, tmp_path):
        for name in ("test_document.md", "test_doc.pdf", "test_page.html", "reason_codes.csv"):
            _copy(name, tmp_path)
        await _ingest(db_url, layer, tmp_path)

        docs = await _documents(db_url, layer)
        assert len(docs) == 4
        assert {d["text_origin"] for d in docs} == {"text"}
        for d in docs:
            assert d["sha256"] == hashlib.sha256(d["text"].encode("utf-8")).hexdigest()


class TestLocate:
    async def test_locate_pdf_pages(self, db_url, layer, tmp_path):
        from stratakb.knowledge.documents import locate

        _copy("test_doc.pdf", tmp_path)
        await _ingest(db_url, layer, tmp_path)
        doc = (await _documents(db_url, layer))[0]

        on_page_2 = [
            c for c in await _chunks(db_url, layer) if json.loads(c["metadata"])["page"] == 2
        ]
        assert on_page_2
        for c in on_page_2:
            loc = locate(doc["text"], doc["page_map"], c["char_start"], c["char_end"])
            assert (loc["page"], loc["page_end"]) == (2, 2)

        page_2_start = json.loads(doc["page_map"])[1][0]
        # A span from page 1 into page 2; lines count from each page's start
        loc = locate(doc["text"], doc["page_map"], 0, page_2_start + 3)
        assert (loc["page"], loc["page_end"]) == (1, 2)
        assert (loc["line_start"], loc["line_end"]) == (1, 1)

    async def test_locate_markdown_lines(self, db_url, layer, tmp_path):
        from stratakb.knowledge.documents import locate

        (tmp_path / "guide.md").write_text(
            "# Title\n\nline one\nline two\n\n## Sub\n\nline three\n", encoding="utf-8"
        )
        await _ingest(db_url, layer, tmp_path)
        doc = (await _documents(db_url, layer))[0]
        text = doc["text"]

        start = text.index("line two")
        loc = locate(text, doc["page_map"], start, start + len("line two"))
        assert loc == {"page": None, "page_end": None, "line_start": 4, "line_end": 4}

        start = text.index("line one")
        end = text.index("line three") + len("line three")
        assert locate(text, None, start, end)["line_start"] == 3
        assert locate(text, None, start, end)["line_end"] == 8

    def test_locate_span_ending_at_newline(self):
        from stratakb.knowledge.documents import locate

        text = "a\nb\nc"
        # "b\n" ends on line 2: the newline is the span's last character
        assert locate(text, None, 2, 4)["line_end"] == 2
        assert locate(text, None, 4, 4)["line_start"] == 3


class TestBuildText:
    def test_pages_joined_and_page_map_uses_page_numbers(self):
        from stratakb.knowledge.documents import build_document_text, page_starts
        from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

        # Page 2 was empty and dropped by the parser
        doc = ParsedDocument(
            source_file="scan.pdf",
            pages=[ParsedPage("first", 1, "scan.pdf"), ParsedPage("third", 3, "scan.pdf")],
            total_pages=3,
        )
        text, page_map = build_document_text(doc)
        assert text == "first\n\nthird"
        assert page_map == [[0, 1], [7, 3]]
        assert page_starts(doc) == [0, 7]

    def test_no_page_map_for_other_formats(self):
        from stratakb.knowledge.documents import build_document_text
        from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

        doc = ParsedDocument(source_file="a.md", pages=[ParsedPage("# A", 1, "a.md")])
        assert build_document_text(doc) == ("# A", None)


class TestVersions:
    async def test_reingest_unchanged_keeps_version(self, db_url, layer, tmp_path):
        from stratakb.knowledge.selection import FileEntry

        path = _copy("test_document.md", tmp_path)
        await _ingest(db_url, layer, tmp_path)
        before = await _documents(db_url, layer)

        result = await _ingest(db_url, layer, tmp_path)
        assert result.files_unchanged == 1
        # A carried hash that differs from the tracked one forces a re-parse
        # and re-store; the text is the same, so the version is kept
        entry = FileEntry(
            relative_path="test_document.md", size=path.stat().st_size, path=path,
            sha256="0" * 64,
        )
        result = await _ingest(db_url, layer, tmp_path, files=[entry])
        assert result.files_updated == 1

        after = await _documents(db_url, layer)
        assert [(d["id"], d["version"], d["sha256"]) for d in after] == [
            (d["id"], d["version"], d["sha256"]) for d in before
        ]
        assert after[0]["version"] == 1
        await _assert_offsets_match(db_url, layer)

    async def test_reingest_changed_bumps_version(self, db_url, layer, tmp_path):
        path = tmp_path / "guide.md"
        path.write_text("# Guide\n\nFirst text about failover.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        (v1,) = await _documents(db_url, layer)

        path.write_text("# Guide\n\nSecond text about restores.\n", encoding="utf-8")
        result = await _ingest(db_url, layer, tmp_path)
        assert result.files_updated == 1

        # Version 1 has no chunks left, so it is pruned
        (v2,) = await _documents(db_url, layer)
        assert v2["version"] == 2
        assert v2["sha256"] != v1["sha256"]
        assert "restores" in v2["text"]
        await _assert_offsets_match(db_url, layer)

    async def test_same_name_from_different_roots_keeps_own_versions(
        self, db_url, layer, tmp_path
    ):
        # A single-file ingest stores the bare file name below its parent, so
        # both files have relative path README.md in the same layer
        a = tmp_path / "a" / "README.md"
        b = tmp_path / "b" / "README.md"
        a.parent.mkdir()
        b.parent.mkdir()
        a.write_text("# A\n\nAlpha text about failover.\n", encoding="utf-8")
        b.write_text("# B\n\nBeta text about restores.\n", encoding="utf-8")
        await _ingest(db_url, layer, a)
        await _ingest(db_url, layer, b)

        async def versions() -> dict:
            from stratakb.core.database import get_pool

            pool = await get_pool(db_url)
            rows = await pool.fetch(
                """
                SELECT d.source_root, d.version, d.text, f.source_root AS file_root
                FROM strata_vectors.documents d
                JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
                LEFT JOIN strata_core.ingested_files f ON f.id = d.file_id
                WHERE kl.name = $1
                ORDER BY d.source_root, d.version
                """,
                layer,
            )
            for r in rows:
                assert r["file_root"] == r["source_root"]
            return {r["source_root"]: (r["version"], r["text"]) for r in rows}

        root_a, root_b = a.parent.resolve().as_posix(), b.parent.resolve().as_posix()
        assert {k: v[0] for k, v in (await versions()).items()} == {root_a: 1, root_b: 1}

        a.write_text("# A\n\nAlpha text, now about backups.\n", encoding="utf-8")
        await _ingest(db_url, layer, a)
        after = await versions()
        assert after[root_a][0] == 2 and "backups" in after[root_a][1]
        assert after[root_b][0] == 1 and "restores" in after[root_b][1]

        # A's new text equal to B's: A gets its own version 3, B keeps its row
        a.write_text(b.read_text(encoding="utf-8"), encoding="utf-8")
        await _ingest(db_url, layer, a)
        after = await versions()
        assert after[root_a][0] == 3
        assert after[root_b][0] == 1
        await _assert_offsets_match(db_url, layer)

    async def test_removed_file_prunes_its_document(self, db_url, layer, tmp_path):
        (tmp_path / "a.md").write_text("# A\n\nAlpha.\n", encoding="utf-8")
        b = tmp_path / "b.md"
        b.write_text("# B\n\nBeta.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        assert len(await _documents(db_url, layer)) == 2

        b.unlink()
        await _ingest(db_url, layer, tmp_path)
        assert [d["relative_path"] for d in await _documents(db_url, layer)] == ["a.md"]

    async def test_store_prunes_only_its_own_file(self, db_url, layer, tmp_path):
        root = tmp_path.resolve().as_posix()
        # A document no chunk references, as another path may leave behind
        await _store_bare_document(db_url, layer, root, "a.md", "Alpha.")

        b = tmp_path / "b.md"
        b.write_text("# B\n\nFirst text about failover.\n", encoding="utf-8")
        await _ingest(db_url, layer, b)
        b.write_text("# B\n\nSecond text about restores.\n", encoding="utf-8")
        await _ingest(db_url, layer, b)

        docs = await _documents(db_url, layer)
        assert [(d["relative_path"], d["version"]) for d in docs] == [("a.md", 1), ("b.md", 2)]

    async def test_removed_file_keeps_other_files_documents(self, db_url, layer, tmp_path):
        root = tmp_path.resolve().as_posix()
        (tmp_path / "a.md").write_text("# A\n\nAlpha.\n", encoding="utf-8")
        b = tmp_path / "b.md"
        b.write_text("# B\n\nBeta.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        await _store_bare_document(db_url, layer, root, "c.md", "Gamma.")

        # An emptied file is removed as having no text
        b.write_text("", encoding="utf-8")
        result = await _ingest(db_url, layer, tmp_path)
        assert ("b.md", "removed", "no_text") in [
            (f.relative_path, f.outcome, f.reason) for f in result.files
        ]
        assert [d["relative_path"] for d in await _documents(db_url, layer)] == ["a.md", "c.md"]

    @pytest.mark.parametrize("same_text", [False, True])
    async def test_concurrent_versions_numbered(self, db_url, layer, same_text):
        import asyncio

        from stratakb.core.database import get_pool
        from stratakb.knowledge.documents import store_document_version

        pool = await get_pool(db_url)
        layer_id = await _layer_id(db_url, layer)

        def store(conn, text):
            return store_document_version(
                conn, layer_id=layer_id, file_id=None, source_root="/race",
                relative_path="race.md", text=text, page_map=None,
            )

        async with pool.acquire() as conn_a, pool.acquire() as conn_b:
            tx_a = conn_a.transaction()
            await tx_a.start()
            tx_b = conn_b.transaction()
            await tx_b.start()
            try:
                first = await store(conn_a, "First text.")
                task = asyncio.create_task(
                    store(conn_b, "First text." if same_text else "Second text.")
                )
                # B must be waiting on A before A commits
                pid = conn_b.get_server_pid()
                async with asyncio.timeout(5):
                    while await pool.fetchval(
                        "SELECT wait_event_type FROM pg_stat_activity WHERE pid = $1", pid
                    ) != "Lock":
                        await asyncio.sleep(0.05)
                await tx_a.commit()
                second = await task
                await tx_b.commit()
            except BaseException:
                with contextlib.suppress(Exception):
                    await tx_a.rollback()
                with contextlib.suppress(Exception):
                    await tx_b.rollback()
                raise

        assert first[1] == 1
        if same_text:
            assert second == first
        else:
            assert second[1] == 2
        versions = [
            r["version"] for r in await pool.fetch(
                "SELECT version FROM strata_vectors.documents"
                " WHERE layer_id = $1 AND relative_path = 'race.md' ORDER BY version",
                layer_id,
            )
        ]
        assert versions == ([1] if same_text else [1, 2])

    async def test_clear_layer_prunes_documents(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import clear_layer

        (tmp_path / "a.md").write_text("# A\n\nAlpha.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        await clear_layer(db_url, layer)
        assert await _documents(db_url, layer) == []

    async def test_ingest_reuses_no_version_a_prune_deleted(self, db_url, layer, tmp_path):
        from stratakb.core.database import get_pool
        from stratakb.knowledge.documents import prune_documents
        from stratakb.rag.receipts import new_receipt_id

        src = tmp_path / "src"
        src.mkdir()
        a = src / "a.md"
        a.write_text("# A\n\nAlpha about failover.\n", encoding="utf-8")
        (src / "b.md").write_text("# B\n\nBeta.\n", encoding="utf-8")
        await _ingest(db_url, layer, src)
        pruned = {d["relative_path"]: d["id"] for d in await _documents(db_url, layer)}["a.md"]

        pool = await get_pool(db_url)
        layer_id = await _layer_id(db_url, layer)
        receipt_id = new_receipt_id()
        await pool.execute(
            """
            INSERT INTO strata_core.receipts (id, issued_by, kind, mode, layers, body)
            VALUES ($1, 'alice', 'verify', 'grounded', '[]', '{}')
            """,
            receipt_id,
        )
        await pool.execute(
            "INSERT INTO strata_core.receipt_documents (receipt_id, document_id) VALUES ($1, $2)",
            receipt_id,
            pruned,
        )
        # Removed from the folder: the receipt keeps its version
        away = tmp_path / "away.md"
        a.rename(away)
        await _ingest(db_url, layer, src)
        assert pruned in [d["id"] for d in await _documents(db_url, layer)]

        async with pool.acquire() as held:
            tx = held.transaction()
            await tx.start()
            try:
                await held.execute("DELETE FROM strata_core.receipts WHERE id = $1", receipt_id)
                assert await prune_documents(held, [layer_id]) == 1
                # Back with the same text while the prune is not committed yet
                away.rename(a)
                ingest = asyncio.create_task(_ingest(db_url, layer, src))
                await wait_until_blocked_by(pool, held.get_server_pid(), ingest)
                await tx.commit()
            except BaseException:
                with contextlib.suppress(Exception):
                    await tx.rollback()
                raise
        result = await ingest

        assert ("a.md", "added", None) in [
            (f.relative_path, f.outcome, f.reason) for f in result.files
        ]
        assert result.errors == []
        docs = await _assert_offsets_match(db_url, layer)
        a_docs = [d for d in await _documents(db_url, layer) if d["relative_path"] == "a.md"]
        assert len(a_docs) == 1
        assert docs["a.md"]["id"] != pruned

    async def test_clear_layer_locks_file_rows_before_chunks(self, db_url, layer, tmp_path):
        import asyncpg

        from stratakb.core.database import get_pool
        from stratakb.knowledge.ingestion import clear_layer

        (tmp_path / "a.md").write_text("# A\n\nAlpha.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        file_id = (await _file_rows(db_url, layer))["a.md"]

        pool = await get_pool(db_url)
        async with pool.acquire() as held, pool.acquire() as probe:
            tx = held.transaction()
            await tx.start()
            try:
                await held.execute(
                    "SELECT id FROM strata_core.ingested_files WHERE id = $1 FOR KEY SHARE",
                    file_id,
                )
                clearing = asyncio.create_task(clear_layer(db_url, layer))
                await wait_until_blocked_by(pool, held.get_server_pid(), clearing)
                # The clear waits on the file row without holding its chunks
                probe_tx = probe.transaction()
                await probe_tx.start()
                try:
                    await probe.execute(
                        "SELECT id FROM strata_vectors.chunks WHERE file_id = $1"
                        " FOR UPDATE NOWAIT",
                        file_id,
                    )
                except asyncpg.LockNotAvailableError:
                    pytest.fail("clear_layer locked the chunks before the file row")
                finally:
                    await probe_tx.rollback()
            finally:
                await tx.rollback()
        await clearing

        assert await _chunks(db_url, layer) == []
        assert await _documents(db_url, layer) == []
        assert await _file_rows(db_url, layer) == {}

    @pytest.mark.parametrize("how", ["remove", "clear"])
    async def test_removal_waits_for_ingest_of_same_file(self, db_url, layer, tmp_path, how):
        from stratakb.core.database import get_pool
        from stratakb.knowledge import ingestion

        a = tmp_path / "a.md"
        a.write_text("# A\n\nFirst text about failover.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        row_id = (await _file_rows(db_url, layer))["a.md"]
        layer_id = await _layer_id(db_url, layer)
        a.write_text("# A\n\nSecond text about restores.\n", encoding="utf-8")

        real = ingestion.store_document_version
        picked, release = asyncio.Event(), asyncio.Event()
        pids = []

        async def paused(conn, **kwargs):
            stored = await real(conn, **kwargs)
            pids.append(conn.get_server_pid())
            picked.set()
            await release.wait()
            return stored

        pool = await get_pool(db_url)
        with patch("stratakb.knowledge.ingestion.store_document_version", paused):
            ingest = asyncio.create_task(_ingest(db_url, layer, tmp_path))
            try:
                async with asyncio.timeout(30):
                    await picked.wait()
                    if how == "remove":
                        removal = asyncio.create_task(ingestion._remove_files(
                            pool, layer_id=str(layer_id),
                            source_root=tmp_path.resolve().as_posix(),
                            files=[("a.md", row_id)],
                        ))
                    else:
                        removal = asyncio.create_task(ingestion.clear_layer(db_url, layer))
                    await wait_until_blocked_by(pool, pids[0], removal)
            finally:
                release.set()
            result = await ingest
        await removal

        assert ("a.md", "updated", None) in [
            (f.relative_path, f.outcome, f.reason) for f in result.files
        ]
        assert result.errors == []
        assert await _file_rows(db_url, layer) == {}
        assert await _chunks(db_url, layer) == []
        assert await _documents(db_url, layer) == []

    async def test_overlapping_removers_do_not_deadlock(self, db_url, layer, tmp_path):
        from stratakb.core.database import get_pool
        from stratakb.knowledge.ingestion import _remove_files, clear_layer

        for name in ("a", "b", "c"):
            (tmp_path / f"{name}.md").write_text(f"# {name}\n\nText {name}.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        ids = await _file_rows(db_url, layer)
        layer_id = str(await _layer_id(db_url, layer))
        root = tmp_path.resolve().as_posix()
        pool = await get_pool(db_url)

        def remove(*names):
            return _remove_files(
                pool, layer_id=layer_id, source_root=root,
                files=[(f"{n}.md", ids[f"{n}.md"]) for n in names],
            )

        async with asyncio.timeout(30):
            await asyncio.gather(
                remove("a", "b"), remove("c", "b", "a"), clear_layer(db_url, layer)
            )

        assert await _file_rows(db_url, layer) == {}
        assert await _chunks(db_url, layer) == []
        assert await _documents(db_url, layer) == []

    async def test_delete_layer_removes_documents(self, db_url, tmp_path):
        from stratakb.core.database import get_pool
        from stratakb.knowledge.layers import create_layer, delete_layer

        name = f"test_docs_{uuid4().hex[:10]}"
        await create_layer(db_url, name=name, display_name="Doomed")
        (tmp_path / "a.md").write_text("# A\n\nAlpha.\n", encoding="utf-8")
        await _ingest(db_url, name, tmp_path)
        pool = await get_pool(db_url)
        layer_id = await pool.fetchval(
            "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", name
        )

        await delete_layer(db_url, name=name)
        assert await pool.fetchval(
            "SELECT count(*) FROM strata_vectors.documents WHERE layer_id = $1", layer_id
        ) == 0


class TestLegacyChunks:
    async def test_chunk_without_offsets_still_searchable(self, db_url, layer):
        from stratakb.knowledge.vector_store import ChunkRecord, insert_chunks
        from stratakb.rag.retriever import search_chunks

        await insert_chunks(db_url, [ChunkRecord(
            content="Legacy chunk about failover drills.",
            embedding=FAKE_EMBEDDING,
            layer_name=layer,
            source_file="old.md",
        )])
        rows = await _chunks(db_url, layer)
        assert [(r["document_id"], r["char_start"], r["char_end"]) for r in rows] == [
            (None, None, None)
        ]

        with (
            patch("stratakb.rag.retriever._get_cache", return_value=None),
            patch(
                "stratakb.rag.retriever.embed_texts", new_callable=AsyncMock,
                return_value=[FAKE_EMBEDDING],
            ),
        ):
            results = await search_chunks(db_url, "fake-key", "failover", layers=[layer])
        assert [r.content for r in results] == ["Legacy chunk about failover drills."]
