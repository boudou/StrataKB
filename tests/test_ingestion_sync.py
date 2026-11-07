"""Tests for per-file ingestion tracking: sync by default, relative paths, recounted stats.

All tests need the database; embeddings are mocked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tests.helpers import (
    FAKE_EMBEDDING,
    as_admin,
    call_tool,
    file_outcomes,
    ingest,
    ingest_with_texts,
    locked_read_bytes,
    write_file,
)


async def _chunks(db_url: str, layer: str) -> list:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetch(
        "SELECT id, source_file, file_id FROM strata_vectors.chunks WHERE layer_name = $1",
        layer,
    )


async def _files(db_url: str, layer: str) -> list:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetch(
        """
        SELECT f.source_root, f.relative_path, f.content_hash, f.chunk_count
        FROM strata_core.ingested_files f
        JOIN strata_core.knowledge_layers kl ON kl.id = f.layer_id
        WHERE kl.name = $1
        """,
        layer,
    )


def _ids_by_file(rows) -> dict[str, set]:
    out: dict[str, set] = {}
    for r in rows:
        out.setdefault(r["source_file"], set()).add(r["id"])
    return out


class TestSync:
    async def test_unchanged_second_run_embeds_nothing(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content about failover.\n")
        write_file(tmp_path / "b.md", "# Beta\n\nBeta content about backups.\n")

        first, embedded = await ingest_with_texts(db_url, layer, tmp_path)
        assert first.files_added == 2
        assert embedded

        second, embedded = await ingest_with_texts(db_url, layer, tmp_path)
        assert embedded == []
        assert second.status == "completed"
        assert second.files_unchanged == 2
        assert (second.files_added, second.files_updated, second.files_removed) == (0, 0, 0)
        assert second.docs_processed == 0
        assert len(await _chunks(db_url, layer)) == first.chunks_created

    async def test_tracked_files_by_relative_path(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import tracked_files
        from stratakb.knowledge.layers import get_layer

        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        write_file(tmp_path / "sub" / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)
        layer_id = (await get_layer(db_url, name=layer))["id"]
        source_root = tmp_path.resolve().as_posix()

        tracked = await tracked_files(db_url, layer_id, source_root)

        stored = {r["relative_path"]: r["content_hash"] for r in await _files(db_url, layer)}
        assert set(tracked) == {"a.md", "sub/b.md"}
        assert {rel: r["content_hash"] for rel, r in tracked.items()} == stored
        assert all(tracked[rel]["relative_path"] == rel for rel in tracked)
        assert await tracked_files(db_url, layer_id, source_root + "/other") == {}

    async def test_edit_replaces_only_that_file(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = write_file(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)
        before = _ids_by_file(await _chunks(db_url, layer))

        write_file(b, "# Beta\n\nBeta content, edited with new restore steps.\n")
        result, embedded = await ingest_with_texts(db_url, layer, tmp_path)

        assert result.files_updated == 1
        assert result.files_unchanged == 1
        assert all("restore steps" in t or "Beta" in t for t in embedded)
        after = _ids_by_file(await _chunks(db_url, layer))
        assert after["a.md"] == before["a.md"]
        assert after["b.md"].isdisjoint(before["b.md"])

    async def test_deleted_file_is_removed(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = write_file(tmp_path / "sub" / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)

        b.unlink()
        result = await ingest(db_url, layer, tmp_path)

        assert result.files_removed == 1
        assert {r["source_file"] for r in await _chunks(db_url, layer)} == {"a.md"}
        assert [r["relative_path"] for r in await _files(db_url, layer)] == ["a.md"]

    async def test_same_name_in_two_folders(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a" / "readme.md", "# A\n\nReadme of service A.\n")
        write_file(tmp_path / "b" / "readme.md", "# B\n\nReadme of service B.\n")

        result = await ingest(db_url, layer, tmp_path)

        assert result.files_added == 2
        assert {r["source_file"] for r in await _chunks(db_url, layer)} == {
            "a/readme.md",
            "b/readme.md",
        }
        files = await _files(db_url, layer)
        assert {f["relative_path"] for f in files} == {"a/readme.md", "b/readme.md"}
        assert {f["source_root"] for f in files} == {tmp_path.resolve().as_posix()}

    async def test_second_folder_keeps_first(self, db_url, layer, tmp_path):
        x = tmp_path / "x"
        y = tmp_path / "y"
        write_file(x / "readme.md", "# X\n\nFolder X content.\n")
        write_file(y / "readme.md", "# Y\n\nFolder Y content.\n")

        await ingest(db_url, layer, x)
        result = await ingest(db_url, layer, y)

        assert result.files_added == 1
        assert result.files_removed == 0
        files = await _files(db_url, layer)
        assert {f["source_root"] for f in files} == {x.resolve().as_posix(), y.resolve().as_posix()}
        assert len(_ids_by_file(await _chunks(db_url, layer))["readme.md"]) >= 2

    async def test_parse_failure_keeps_old_chunks(self, db_url, layer, tmp_path):
        doc = write_file(
            tmp_path / "codes.json", '{"code": "AC01", "meaning": "Incorrect account"}'
        )
        await ingest(db_url, layer, tmp_path)
        before = _ids_by_file(await _chunks(db_url, layer))["codes.json"]

        write_file(doc, '{"code": "AC01", "meaning": ')
        result, embedded = await ingest_with_texts(db_url, layer, tmp_path)

        assert embedded == []
        assert result.docs_failed == 1
        assert result.files_removed == 0
        assert [e["file"] for e in result.errors] == ["codes.json"]
        assert _ids_by_file(await _chunks(db_url, layer))["codes.json"] == before
        assert len(await _files(db_url, layer)) == 1

    async def test_single_file_never_removes(self, db_url, layer, tmp_path):
        a = write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = write_file(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)

        b.unlink()
        result = await ingest(db_url, layer, a)

        assert result.files_unchanged == 1
        assert result.files_removed == 0
        assert {r["source_file"] for r in await _chunks(db_url, layer)} == {"a.md", "b.md"}

    async def test_replace_all_rebuilds(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        other = write_file(
            tmp_path.parent / f"{tmp_path.name}_other" / "o.md", "# Other\n\nOther.\n"
        )
        await ingest(db_url, layer, tmp_path)
        await ingest(db_url, layer, other.parent)
        before = {r["id"] for r in await _chunks(db_url, layer)}

        result, embedded = await ingest_with_texts(db_url, layer, tmp_path, replace_all=True)

        assert embedded
        assert result.files_added == 1
        assert result.files_unchanged == 0
        rows = await _chunks(db_url, layer)
        assert {r["source_file"] for r in rows} == {"a.md"}
        assert {r["id"] for r in rows}.isdisjoint(before)
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["a.md"]

    async def test_layer_counts_after_append(self, db_url, layer, tmp_path):
        from stratakb.knowledge.layers import get_layer_status

        x = tmp_path / "x"
        y = tmp_path / "y"
        write_file(x / "one.md", "# One\n\nFirst folder.\n")
        write_file(x / "two.md", "# Two\n\nFirst folder again.\n")
        write_file(y / "three.md", "# Three\n\nSecond folder.\n")

        await ingest(db_url, layer, x)
        result = await ingest(db_url, layer, y)

        status = await get_layer_status(db_url, name=layer)
        assert result.docs_processed == 1
        assert status["doc_count"] == len(await _files(db_url, layer)) == 3
        assert status["chunk_count"] == len(await _chunks(db_url, layer))

    async def test_legacy_chunks_rebuilt_once(self, db_url, layer, tmp_path):
        from stratakb.knowledge.vector_store import ChunkRecord, insert_chunks

        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        await insert_chunks(db_url, [
            ChunkRecord(
                content="Old chunk", embedding=FAKE_EMBEDDING, layer_name=layer,
                source_file="a.md", doc_type="markdown",
            )
        ])

        first = await ingest(db_url, layer, tmp_path)
        assert first.rebuilt_legacy is True
        assert first.files_added == 1
        rows = await _chunks(db_url, layer)
        assert rows and all(r["file_id"] is not None for r in rows)

        second = await ingest(db_url, layer, tmp_path)
        assert second.rebuilt_legacy is False
        assert second.files_unchanged == 1


class TestPolicyAndEmpty:
    async def test_secret_names_never_opened(self, db_url, layer, tmp_path):
        from stratakb.knowledge import ingestion

        write_file(tmp_path / ".env.local", "API_KEY=do-not-read\n")
        write_file(tmp_path / "client_secret.json", '{"client_secret": "do-not-read"}')
        write_file(tmp_path / ".env.json", '{"token": "do-not-read"}')
        write_file(tmp_path / "notes.md", "# Notes\n\nPlain runbook notes.\n")

        read: list[str] = []
        parsed: list[str] = []
        original_read = Path.read_bytes
        original_parse = ingestion._parse_file

        def spy_read(self):
            read.append(self.name)
            return original_read(self)

        def spy_parse(file_path):
            parsed.append(file_path.name)
            return original_parse(file_path)

        with (
            patch.object(Path, "read_bytes", spy_read),
            patch("stratakb.knowledge.ingestion._parse_file", side_effect=spy_parse),
        ):
            result, embedded = await ingest_with_texts(db_url, layer, tmp_path)

        assert read == ["notes.md"]
        assert parsed == ["notes.md"]
        assert not any("do-not-read" in t for t in embedded)
        outcomes = file_outcomes(result)
        assert outcomes["client_secret.json"] == ("skipped", "secret")
        assert outcomes[".env.json"] == ("skipped", "secret")
        assert outcomes["notes.md"] == ("added", None)
        assert ".env.local" not in outcomes  # not a supported format, never discovered
        assert result.files_skipped == 2
        assert result.status == "completed"
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["notes.md"]

    async def test_tracked_secret_removed(self, db_url, layer, tmp_path):
        write_file(tmp_path / "secrets.yaml", "token: abc\nowner: ops\n")
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        with patch("stratakb.knowledge.ingestion.is_secret_name", return_value=False):
            first = await ingest(db_url, layer, tmp_path)
        assert first.files_added == 2
        assert "secrets.yaml" in _ids_by_file(await _chunks(db_url, layer))

        result = await ingest(db_url, layer, tmp_path)

        assert file_outcomes(result)["secrets.yaml"] == ("removed", "secret")
        assert result.files_removed == 1
        assert result.files_skipped == 0
        assert {r["source_file"] for r in await _chunks(db_url, layer)} == {"a.md"}
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["a.md"]

    async def test_empty_files_are_no_text(self, db_url, layer, tmp_path):
        write_file(tmp_path / "empty.md", "")
        (tmp_path / "zero.csv").write_bytes(b"")
        write_file(tmp_path / "blank.json", "  \n\t\n")
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        result = await ingest(db_url, layer, tmp_path)

        outcomes = file_outcomes(result)
        for name in ("empty.md", "zero.csv", "blank.json"):
            assert outcomes[name] == ("skipped", "no_text")
        assert outcomes["a.md"] == ("added", None)
        assert result.files_skipped == 3
        assert result.docs_failed == 0
        assert result.errors == []
        assert result.status == "completed"

    async def test_parses_without_text_are_no_text(self, db_url, layer, tmp_path):
        write_file(tmp_path / "codes.csv", "Code,Name\n")
        write_file(tmp_path / "blank.html", "<html><body><p> </p></body></html>")
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        result = await ingest(db_url, layer, tmp_path)

        outcomes = file_outcomes(result)
        assert outcomes["codes.csv"] == ("skipped", "no_text")
        assert outcomes["blank.html"] == ("skipped", "no_text")
        assert outcomes["a.md"] == ("added", None)
        assert result.files_skipped == 2
        assert result.docs_failed == 0
        assert result.errors == []
        assert result.status == "completed"
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["a.md"]

    async def test_tracked_csv_losing_its_rows_removed(self, db_url, layer, tmp_path):
        codes = write_file(tmp_path / "codes.csv", "Code,Name\nA1,Alpha\n")
        write_file(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)

        write_file(codes, "Code,Name\n")
        result = await ingest(db_url, layer, tmp_path)

        assert file_outcomes(result)["codes.csv"] == ("removed", "no_text")
        assert result.files_removed == 1
        assert result.errors == []
        assert {r["source_file"] for r in await _chunks(db_url, layer)} == {"b.md"}
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["b.md"]

    async def test_tracked_file_becoming_empty_removed(self, db_url, layer, tmp_path):
        a = write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        write_file(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)

        write_file(a, "")
        result = await ingest(db_url, layer, tmp_path)

        assert file_outcomes(result)["a.md"] == ("removed", "no_text")
        assert result.files_removed == 1
        assert result.errors == []
        assert {r["source_file"] for r in await _chunks(db_url, layer)} == {"b.md"}
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["b.md"]

    async def test_parser_exception_fails_one_file(self, db_url, layer, tmp_path):
        from stratakb.knowledge import ingestion

        bad = write_file(tmp_path / "bad.md", "# Bad\n\nFirst version.\n")
        await ingest(db_url, layer, tmp_path)
        before = _ids_by_file(await _chunks(db_url, layer))["bad.md"]

        write_file(bad, "# Bad\n\nSecond version.\n")
        write_file(tmp_path / "good.md", "# Good\n\nGood content.\n")
        original_parse = ingestion._parse_file

        def raising_parse(file_path):
            if file_path.name == "bad.md":
                raise RuntimeError("parser exploded")
            return original_parse(file_path)

        with patch("stratakb.knowledge.ingestion._parse_file", side_effect=raising_parse):
            result = await ingest(db_url, layer, tmp_path)

        outcomes = file_outcomes(result)
        assert outcomes["bad.md"] == ("failed", "parse_error")
        assert outcomes["good.md"] == ("added", None)
        assert result.errors == [
            {"file": "bad.md", "error": "parser exploded", "reason": "parse_error"}
        ]
        assert result.docs_failed == 1
        assert result.status == "partial"
        assert _ids_by_file(await _chunks(db_url, layer))["bad.md"] == before

    async def test_unreadable_file_is_read_error(self, db_url, layer, tmp_path):
        write_file(tmp_path / "locked.md", "# Locked\n\nLocked content.\n")
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        with patch.object(Path, "read_bytes", locked_read_bytes("locked.md")):
            result = await ingest(db_url, layer, tmp_path)

        outcomes = file_outcomes(result)
        assert outcomes["locked.md"] == ("failed", "read_error")
        assert outcomes["a.md"] == ("added", None)
        assert result.errors == [{"file": "locked.md", "error": "locked", "reason": "read_error"}]
        assert result.docs_failed == 1
        assert result.status == "partial"

    async def test_tracked_file_unreadable_keeps_chunks(self, db_url, layer, tmp_path):
        locked = write_file(tmp_path / "locked.md", "# Locked\n\nFirst version.\n")
        await ingest(db_url, layer, tmp_path)
        before = _ids_by_file(await _chunks(db_url, layer))["locked.md"]

        write_file(locked, "# Locked\n\nSecond version.\n")
        with patch.object(Path, "read_bytes", locked_read_bytes("locked.md")):
            result = await ingest(db_url, layer, tmp_path)

        assert file_outcomes(result)["locked.md"] == ("failed", "read_error")
        assert _ids_by_file(await _chunks(db_url, layer))["locked.md"] == before

    async def test_deleted_file_outcome(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import FileOutcome

        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = write_file(tmp_path / "sub" / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)

        b.unlink()
        result = await ingest(db_url, layer, tmp_path)

        assert FileOutcome("sub/b.md", "removed", "deleted") in result.files
        assert FileOutcome("a.md", "unchanged") in result.files
        assert result.files_removed == 1


def _pdf_doc(page_lengths: list[int], total_pages: int, scanned_pages: set[int]):
    """A parsed PDF with one text page per length, numbered from 1; the parser
    drops pages without text. scanned_pages are the pages that count as scanned."""
    from stratakb.knowledge.parsers.document import ParsedDocument, ParsedPage
    from stratakb.knowledge.parsers.pdf_parser import scanned_flag

    doc = ParsedDocument(
        source_file="scan.pdf",
        pages=[
            ParsedPage(text="x" * n, page_number=i + 1, source_file="scan.pdf")
            for i, n in enumerate(page_lengths)
        ],
        total_pages=total_pages,
        scanned_pages=scanned_pages,
    )
    doc.flag = scanned_flag(doc)
    return doc


class TestScannedPdf:
    async def _ingest_pdf(
        self,
        db_url,
        layer,
        root: Path,
        page_lengths: list[int],
        total_pages: int,
        scanned_pages: set[int],
    ):
        """Ingest root with scan.pdf parsed as the given pages; other files parse as usual."""
        from stratakb.knowledge import ingestion

        original_parse = ingestion._parse_file

        def fake_parse(file_path):
            if file_path.name == "scan.pdf":
                return _pdf_doc(page_lengths, total_pages, scanned_pages)
            return original_parse(file_path)

        with patch("stratakb.knowledge.ingestion._parse_file", side_effect=fake_parse):
            result = await ingest(db_url, layer, root)
        return result

    async def test_one_of_four_pages_flagged(self, db_url, layer, tmp_path):
        (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 test")
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        result = await self._ingest_pdf(db_url, layer, tmp_path, [1000], 4, {2, 3, 4})

        outcomes = file_outcomes(result)
        assert outcomes["scan.pdf"] == ("flagged", "probably_scanned")
        assert outcomes["a.md"] == ("added", None)
        assert result.files_flagged == 1
        assert result.files_added == 2
        assert result.docs_processed == 2
        assert result.status == "completed"
        files = {f["relative_path"]: f for f in await _files(db_url, layer)}
        assert files["scan.pdf"]["chunk_count"] > 0
        assert "scan.pdf" in _ids_by_file(await _chunks(db_url, layer))

    async def test_two_of_four_pages_not_flagged(self, db_url, layer, tmp_path):
        (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 test")

        result = await self._ingest_pdf(db_url, layer, tmp_path, [400, 400], 4, {3, 4})

        assert file_outcomes(result)["scan.pdf"] == ("added", None)
        assert result.files_flagged == 0
        assert result.files_added == 1

    async def test_short_pages_without_images_not_flagged(self, db_url, layer, tmp_path):
        (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 test")

        result = await self._ingest_pdf(db_url, layer, tmp_path, [100, 100], 2, set())

        assert file_outcomes(result)["scan.pdf"] == ("added", None)
        assert result.files_flagged == 0
        assert result.files_added == 1

    async def test_pdf_without_text_pages_is_no_text(self, db_url, layer, tmp_path):
        (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 test")
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        result = await self._ingest_pdf(db_url, layer, tmp_path, [], 3, {1, 2, 3})

        assert file_outcomes(result)["scan.pdf"] == ("skipped", "no_text")
        assert result.files_flagged == 0
        assert result.files_skipped == 1
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["a.md"]

    async def test_flagged_update_counts_as_updated(self, db_url, layer, tmp_path):
        scan = tmp_path / "scan.pdf"
        scan.write_bytes(b"%PDF-1.4 first")
        first = await self._ingest_pdf(db_url, layer, tmp_path, [1000], 4, {2, 3, 4})
        assert file_outcomes(first)["scan.pdf"] == ("flagged", "probably_scanned")
        assert first.files_added == 1

        scan.write_bytes(b"%PDF-1.4 second")
        second = await self._ingest_pdf(db_url, layer, tmp_path, [1000], 4, {2, 3, 4})
        assert file_outcomes(second)["scan.pdf"] == ("flagged", "probably_scanned")
        assert second.files_updated == 1
        assert second.files_flagged == 1
        assert second.docs_processed == 1

        third = await self._ingest_pdf(db_url, layer, tmp_path, [1000], 4, {2, 3, 4})
        assert file_outcomes(third)["scan.pdf"] == ("unchanged", None)
        assert third.files_flagged == 0
        assert third.files_unchanged == 1


def _csv(rows: list[str]) -> str:
    return "Code,Name\n" + "".join(f"{r},Name of {r}\n" for r in rows)


class TestTruncatedCsv:
    """A CSV over the parser's row limit (patched to 2) is stored and flagged."""

    async def _ingest_capped(self, db_url, layer, root: Path):
        with patch("stratakb.knowledge.parsers.csv_parser.MAX_ROWS", 2):
            result = await ingest(db_url, layer, root)
        return result

    async def test_over_limit_flagged_and_stored(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import FileOutcome

        write_file(tmp_path / "codes.csv", _csv(["A1", "A2", "A3"]))
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        result = await self._ingest_capped(db_url, layer, tmp_path)

        outcomes = file_outcomes(result)
        assert outcomes["codes.csv"] == ("flagged", "truncated")
        assert FileOutcome("codes.csv", "flagged", "truncated", 1, stored="added") in result.files
        assert outcomes["a.md"] == ("added", None)
        assert result.files_flagged == 1
        assert result.files_added == 2
        assert result.status == "completed"
        files = {f["relative_path"]: f for f in await _files(db_url, layer)}
        assert files["codes.csv"]["chunk_count"] == 2

    async def test_at_limit_not_flagged(self, db_url, layer, tmp_path):
        write_file(tmp_path / "codes.csv", _csv(["A1", "A2"]))

        result = await self._ingest_capped(db_url, layer, tmp_path)

        assert file_outcomes(result)["codes.csv"] == ("added", None)
        assert result.files_flagged == 0
        assert result.files_added == 1

    async def test_flagged_update_and_unchanged_rerun(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import FileOutcome

        codes = tmp_path / "codes.csv"
        write_file(codes, _csv(["A1", "A2", "A3"]))
        first = await self._ingest_capped(db_url, layer, tmp_path)
        assert file_outcomes(first)["codes.csv"] == ("flagged", "truncated")

        write_file(codes, _csv(["B1", "B2", "B3", "B4"]))
        second = await self._ingest_capped(db_url, layer, tmp_path)
        assert FileOutcome("codes.csv", "flagged", "truncated", 2, stored="updated") in second.files
        assert second.files_updated == 1
        assert second.files_flagged == 1

        third = await self._ingest_capped(db_url, layer, tmp_path)
        assert file_outcomes(third)["codes.csv"] == ("unchanged", None)
        assert third.files_flagged == 0
        assert third.files_unchanged == 1


async def test_parser_flag_reaches_the_outcome(db_url, layer, tmp_path):
    """The ingest loop copies any parser's flag into the outcome, whatever the format."""
    from stratakb.knowledge.ingestion import FileOutcome
    from stratakb.knowledge.parsers.document import ParsedDocument, ParsedPage, ParserFlag

    write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
    doc = ParsedDocument(
        source_file="a.md",
        pages=[ParsedPage(text="Alpha content.", page_number=1, source_file="a.md")],
        total_pages=1,
        flag=ParserFlag("truncated", 7),
    )

    with patch("stratakb.knowledge.ingestion._parse_file", return_value=doc):
        result = await ingest(db_url, layer, tmp_path)

    assert FileOutcome("a.md", "flagged", "truncated", 7, stored="added") in result.files
    assert result.files_flagged == 1
    assert result.files_added == 1


async def test_tool_returns_files_skipped(server):
    from stratakb.knowledge.ingestion import FileOutcome, IngestionResult

    fake = IngestionResult(
        layer_name="x",
        status="completed",
        files=[
            FileOutcome("a.md", "skipped", "secret"),
            FileOutcome("b.md", "skipped", "no_text"),
        ],
    )
    with (
        patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
        patch("stratakb.knowledge.tools.get_current_user", return_value=("tester", "admin")),
        patch(
            "stratakb.knowledge.ingestion.ingest_documents",
            new_callable=AsyncMock,
            return_value=fake,
        ),
        patch("stratakb.knowledge.tools.write_audit_log", new_callable=AsyncMock),
    ):
        result = await call_tool(
            server, "strata_ingest_documents", layer_name="x", source_path="y"
        )

    assert result["status"] == "completed"
    assert result["files_skipped"] == 2


async def test_tool_returns_files_flagged(server):
    from stratakb.knowledge.ingestion import FileOutcome, IngestionResult

    fake = IngestionResult(
        layer_name="x",
        status="completed",
        files=[FileOutcome("a.csv", "flagged", "truncated", stored="added")],
    )
    with (
        patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
        patch("stratakb.knowledge.tools.get_current_user", return_value=("tester", "admin")),
        patch(
            "stratakb.knowledge.ingestion.ingest_documents",
            new_callable=AsyncMock,
            return_value=fake,
        ),
        patch("stratakb.knowledge.tools.write_audit_log", new_callable=AsyncMock),
    ):
        result = await call_tool(
            server, "strata_ingest_documents", layer_name="x", source_path="y"
        )

    assert result["status"] == "completed"
    assert result["files_flagged"] == 1
    assert result["files_added"] == 1
    assert result["flagged"] == [{"file": "a.csv", "reason": "truncated", "count": None}]


async def test_tool_lists_flagged_csv_with_the_rows_left_out(server, db_url, layer, tmp_path):
    write_file(tmp_path / "codes.csv", _csv(["A1", "A2", "A3", "A4", "A5"]))
    write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

    with as_admin(), patch("stratakb.knowledge.parsers.csv_parser.MAX_ROWS", 2):
        result = await call_tool(
            server, "strata_ingest_documents", layer_name=layer, source_path=str(tmp_path)
        )

    assert result["flagged"] == [{"file": "codes.csv", "reason": "truncated", "count": 3}]


async def test_tool_flagged_is_none_without_flagged_files(server, db_url, layer, tmp_path):
    write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

    with as_admin():
        result = await call_tool(
            server, "strata_ingest_documents", layer_name=layer, source_path=str(tmp_path)
        )

    assert result["status"] == "completed"
    assert result["flagged"] is None


def test_tool_description_names_the_csv_row_limit():
    from mcp.server.fastmcp import FastMCP

    from stratakb.knowledge.parsers import csv_parser
    from stratakb.knowledge.tools import register_knowledge_tools

    def description() -> str:
        mcp = FastMCP("test")
        register_knowledge_tools(mcp, "postgresql://unused")
        return mcp._tool_manager.get_tool("strata_ingest_documents").description

    assert f"A CSV over {csv_parser.MAX_ROWS:,} rows" in description()
    with patch.object(csv_parser, "MAX_ROWS", 1_234):
        assert "A CSV over 1,234 rows" in description()


def _entries(root: Path, *relative_paths: str, carry_hash: bool = False) -> list:
    """FileEntry objects for files under root, optionally with their real SHA-256."""
    from stratakb.knowledge.selection import FileEntry

    entries = []
    for rel in relative_paths:
        path = root / rel
        data = path.read_bytes()
        entries.append(FileEntry(
            relative_path=rel,
            size=len(data),
            path=path,
            sha256=hashlib.sha256(data).hexdigest() if carry_hash else None,
        ))
    return entries


class TestFileList:
    async def test_excluded_existing_file_removed(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import FileOutcome

        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        write_file(tmp_path / "sub" / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)

        result = await ingest(db_url, layer, tmp_path, files=_entries(tmp_path, "a.md"))

        assert result.status == "completed"
        assert FileOutcome("sub/b.md", "removed", "not_selected") in result.files
        assert FileOutcome("a.md", "unchanged") in result.files
        assert result.files_removed == 1
        assert (tmp_path / "sub" / "b.md").exists()
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["a.md"]
        assert set(_ids_by_file(await _chunks(db_url, layer))) == {"a.md"}

    async def test_unlisted_missing_file_is_deleted(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import FileOutcome

        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = write_file(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)
        b.unlink()

        result = await ingest(db_url, layer, tmp_path, files=_entries(tmp_path, "a.md"))

        assert FileOutcome("b.md", "removed", "deleted") in result.files

    async def test_empty_list_removes_all(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        write_file(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        await ingest(db_url, layer, tmp_path)

        result = await ingest(db_url, layer, tmp_path, files=[])

        assert result.status == "completed"
        assert result.files_removed == 2
        assert set(file_outcomes(result).values()) == {("removed", "not_selected")}
        assert await _files(db_url, layer) == []
        assert await _chunks(db_url, layer) == []

    async def test_empty_list_nothing_tracked_completes(self, db_url, layer, tmp_path):
        result = await ingest(db_url, layer, tmp_path, files=[])

        assert result.status == "completed"
        assert result.errors == []
        assert result.files == []

    async def test_list_ingests_only_given_files(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        write_file(tmp_path / "other.md", "# Other\n\nNot listed.\n")

        result = await ingest(db_url, layer, tmp_path, files=_entries(tmp_path, "a.md"))

        assert file_outcomes(result) == {"a.md": ("added", None)}
        assert [f["relative_path"] for f in await _files(db_url, layer)] == ["a.md"]

    async def test_carried_hash_skips_hashing(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        write_file(tmp_path / "sub" / "b.md", "# Beta\n\nBeta content.\n")
        entries = _entries(tmp_path, "a.md", "sub/b.md", carry_hash=True)

        # Only the module's name: asyncpg's login uses hashlib.sha256 too
        with patch("stratakb.knowledge.ingestion.hashlib") as mock_hashlib:
            first = await ingest(db_url, layer, tmp_path, files=entries)
            second = await ingest(db_url, layer, tmp_path, files=entries)

        mock_hashlib.sha256.assert_not_called()
        assert first.files_added == 2
        assert second.files_unchanged == 2
        stored = {f["relative_path"]: f["content_hash"] for f in await _files(db_url, layer)}
        assert stored == {e.relative_path: e.sha256 for e in entries}

    async def test_carried_unchanged_hash_never_opens(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        entries = _entries(tmp_path, "a.md", carry_hash=True)
        await ingest(db_url, layer, tmp_path, files=entries)

        with patch.object(Path, "read_bytes", side_effect=AssertionError("opened")):
            result = await ingest(db_url, layer, tmp_path, files=entries)

        assert file_outcomes(result) == {"a.md": ("unchanged", None)}

    async def test_files_with_replace_all_raises(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        with pytest.raises(ValueError, match="replace_all"):
            await ingest(
                db_url, layer, tmp_path, files=_entries(tmp_path, "a.md"), replace_all=True
            )

    @pytest.mark.parametrize("bad", ["../x.md", "sub/../../x.md", "/etc/x.md"])
    async def test_dotdot_path_raises(self, db_url, layer, tmp_path, bad):
        from stratakb.knowledge.selection import FileEntry

        entry = FileEntry(relative_path=bad, size=1, path=tmp_path / "x.md")
        with pytest.raises(ValueError, match="relative"):
            await ingest(db_url, layer, tmp_path, files=[entry])

    async def test_file_source_path_raises(self, db_url, layer, tmp_path):
        a = write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        with pytest.raises(ValueError, match="folder"):
            await ingest(db_url, layer, a, files=_entries(tmp_path, "a.md"))

    async def test_progress_calls(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = write_file(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        calls: list[tuple[int, int]] = []

        await ingest(
            db_url, layer, tmp_path,
            files=_entries(tmp_path, "a.md", "b.md", carry_hash=True),
            progress=lambda done, total: calls.append((done, total)),
        )
        assert calls == [(1, 2), (2, 2)]

        calls.clear()
        write_file(b, "# Beta\n\nBeta content, edited.\n")
        await ingest(
            db_url, layer, tmp_path,
            files=_entries(tmp_path, "a.md", "b.md", carry_hash=True),
            progress=lambda done, total: calls.append((done, total)),
        )
        assert calls == [(1, 1)]

    async def test_progress_in_discovery_mode(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        calls: list[tuple[int, int]] = []

        await ingest(db_url, layer, tmp_path, progress=lambda d, t: calls.append((d, t)))

        assert calls == [(1, 1)]

    async def test_connection_error_raises(self, db_url, layer, tmp_path):
        import asyncpg

        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        with (
            patch(
                "stratakb.knowledge.ingestion._store_file",
                new_callable=AsyncMock,
                side_effect=asyncpg.InterfaceError("connection lost"),
            ),
            pytest.raises(asyncpg.InterfaceError),
        ):
            await ingest(db_url, layer, tmp_path, files=_entries(tmp_path, "a.md"))

    async def test_store_runtime_error_is_embed_error(self, db_url, layer, tmp_path):
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        with patch(
            "stratakb.knowledge.ingestion._store_file",
            new_callable=AsyncMock,
            side_effect=RuntimeError("constraint"),
        ):
            result = await ingest(db_url, layer, tmp_path, files=_entries(tmp_path, "a.md"))

        assert file_outcomes(result)["a.md"] == ("failed", "embed_error")
        assert result.status == "failed"

    async def test_root_form_matches_mcp_route(
        self, server, db_url, layer, tmp_path, monkeypatch
    ):
        from stratakb.knowledge.selection import select_files

        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        write_file(tmp_path / "sub" / "b.md", "# Beta\n\nBeta content.\n")

        with as_admin():
            first = await call_tool(
                server, "strata_ingest_documents", layer_name=layer, source_path=str(tmp_path)
            )
        assert first["files_added"] == 2

        result, embedded = await ingest_with_texts(
            db_url, layer, str(tmp_path) + "/", files=select_files(tmp_path).entries
        )

        assert embedded == []
        assert set(file_outcomes(result).values()) == {("unchanged", None)}
        assert (result.files_added, result.files_removed, result.files_unchanged) == (0, 0, 2)
