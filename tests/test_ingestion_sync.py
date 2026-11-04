"""Tests for per-file ingestion tracking: sync by default, relative paths, recounted stats.

All tests need the database; embeddings are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

FAKE_EMBEDDING = [0.01] * 768


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def layer(db_url):
    """A fresh layer with a unique name, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_sync_{uuid4().hex[:10]}"
    await create_layer(
        db_url,
        name=name,
        display_name="Test Sync",
        chunk_config={"chunk_size": 500, "chunk_overlap": 50},
    )
    yield name
    try:
        await delete_layer(db_url, name=name)
    except LayerNotFoundError:
        pass


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def _ingest(db_url: str, layer: str, source: Path, **kwargs):
    """Run ingest_documents with mocked embeddings; return (result, texts embedded)."""
    from stratakb.knowledge.ingestion import ingest_documents

    embedded: list[str] = []

    def fake_embed(texts, **_):
        embedded.extend(texts)
        return [FAKE_EMBEDDING for _ in texts]

    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = fake_embed
        result = await ingest_documents(
            db_url, "fake-key", layer_name=layer, source_path=str(source), **kwargs
        )
    return result, embedded


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
        _write(tmp_path / "a.md", "# Alpha\n\nAlpha content about failover.\n")
        _write(tmp_path / "b.md", "# Beta\n\nBeta content about backups.\n")

        first, embedded = await _ingest(db_url, layer, tmp_path)
        assert first.files_added == 2
        assert embedded

        second, embedded = await _ingest(db_url, layer, tmp_path)
        assert embedded == []
        assert second.status == "completed"
        assert second.files_unchanged == 2
        assert (second.files_added, second.files_updated, second.files_removed) == (0, 0, 0)
        assert second.docs_processed == 0
        assert len(await _chunks(db_url, layer)) == first.chunks_created

    async def test_edit_replaces_only_that_file(self, db_url, layer, tmp_path):
        _write(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = _write(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        await _ingest(db_url, layer, tmp_path)
        before = _ids_by_file(await _chunks(db_url, layer))

        _write(b, "# Beta\n\nBeta content, edited with new restore steps.\n")
        result, embedded = await _ingest(db_url, layer, tmp_path)

        assert result.files_updated == 1
        assert result.files_unchanged == 1
        assert all("restore steps" in t or "Beta" in t for t in embedded)
        after = _ids_by_file(await _chunks(db_url, layer))
        assert after["a.md"] == before["a.md"]
        assert after["b.md"].isdisjoint(before["b.md"])

    async def test_deleted_file_is_removed(self, db_url, layer, tmp_path):
        _write(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = _write(tmp_path / "sub" / "b.md", "# Beta\n\nBeta content.\n")
        await _ingest(db_url, layer, tmp_path)

        b.unlink()
        result, _ = await _ingest(db_url, layer, tmp_path)

        assert result.files_removed == 1
        assert {r["source_file"] for r in await _chunks(db_url, layer)} == {"a.md"}
        assert [r["relative_path"] for r in await _files(db_url, layer)] == ["a.md"]

    async def test_same_name_in_two_folders(self, db_url, layer, tmp_path):
        _write(tmp_path / "a" / "readme.md", "# A\n\nReadme of service A.\n")
        _write(tmp_path / "b" / "readme.md", "# B\n\nReadme of service B.\n")

        result, _ = await _ingest(db_url, layer, tmp_path)

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
        _write(x / "readme.md", "# X\n\nFolder X content.\n")
        _write(y / "readme.md", "# Y\n\nFolder Y content.\n")

        await _ingest(db_url, layer, x)
        result, _ = await _ingest(db_url, layer, y)

        assert result.files_added == 1
        assert result.files_removed == 0
        files = await _files(db_url, layer)
        assert {f["source_root"] for f in files} == {x.resolve().as_posix(), y.resolve().as_posix()}
        assert len(_ids_by_file(await _chunks(db_url, layer))["readme.md"]) >= 2

    async def test_parse_failure_keeps_old_chunks(self, db_url, layer, tmp_path):
        doc = _write(tmp_path / "codes.json", '{"code": "AC01", "meaning": "Incorrect account"}')
        await _ingest(db_url, layer, tmp_path)
        before = _ids_by_file(await _chunks(db_url, layer))["codes.json"]

        _write(doc, '{"code": "AC01", "meaning": ')
        result, embedded = await _ingest(db_url, layer, tmp_path)

        assert embedded == []
        assert result.docs_failed == 1
        assert result.files_removed == 0
        assert [e["file"] for e in result.errors] == ["codes.json"]
        assert _ids_by_file(await _chunks(db_url, layer))["codes.json"] == before
        assert len(await _files(db_url, layer)) == 1

    async def test_single_file_never_removes(self, db_url, layer, tmp_path):
        a = _write(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        b = _write(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        await _ingest(db_url, layer, tmp_path)

        b.unlink()
        result, _ = await _ingest(db_url, layer, a)

        assert result.files_unchanged == 1
        assert result.files_removed == 0
        assert {r["source_file"] for r in await _chunks(db_url, layer)} == {"a.md", "b.md"}

    async def test_replace_all_rebuilds(self, db_url, layer, tmp_path):
        _write(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        other = _write(tmp_path.parent / f"{tmp_path.name}_other" / "o.md", "# Other\n\nOther.\n")
        await _ingest(db_url, layer, tmp_path)
        await _ingest(db_url, layer, other.parent)
        before = {r["id"] for r in await _chunks(db_url, layer)}

        result, embedded = await _ingest(db_url, layer, tmp_path, replace_all=True)

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
        _write(x / "one.md", "# One\n\nFirst folder.\n")
        _write(x / "two.md", "# Two\n\nFirst folder again.\n")
        _write(y / "three.md", "# Three\n\nSecond folder.\n")

        await _ingest(db_url, layer, x)
        result, _ = await _ingest(db_url, layer, y)

        status = await get_layer_status(db_url, name=layer)
        assert result.docs_processed == 1
        assert status["doc_count"] == len(await _files(db_url, layer)) == 3
        assert status["chunk_count"] == len(await _chunks(db_url, layer))

    async def test_legacy_chunks_rebuilt_once(self, db_url, layer, tmp_path):
        from stratakb.knowledge.vector_store import ChunkRecord, insert_chunks

        _write(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        await insert_chunks(db_url, [
            ChunkRecord(
                content="Old chunk", embedding=FAKE_EMBEDDING, layer_name=layer,
                source_file="a.md", doc_type="markdown",
            )
        ])

        first, _ = await _ingest(db_url, layer, tmp_path)
        assert first.rebuilt_legacy is True
        assert first.files_added == 1
        rows = await _chunks(db_url, layer)
        assert rows and all(r["file_id"] is not None for r in rows)

        second, _ = await _ingest(db_url, layer, tmp_path)
        assert second.rebuilt_legacy is False
        assert second.files_unchanged == 1


class TestMigration:
    async def test_sql_file_applies_twice(self, db_url):
        from stratakb.core.database import get_pool
        from stratakb.core.migrations import INGESTED_FILES_SQL

        pool = await get_pool(db_url)
        sql = INGESTED_FILES_SQL.read_text(encoding="utf-8")
        await pool.execute(sql)
        await pool.execute(sql)
        assert await pool.fetchval("SELECT to_regclass('strata_core.ingested_files')::text") == (
            "strata_core.ingested_files"
        )

    async def test_ensure_runs_twice(self, db_url):
        from stratakb.core import migrations

        migrations._checked.discard(db_url)
        await migrations.ensure_ingestion_schema(db_url)
        await migrations.ensure_ingestion_schema(db_url)
        assert db_url in migrations._checked

    async def test_permission_error_names_the_file(self):
        import asyncpg

        from stratakb.core import migrations

        url = "postgresql://nobody@127.0.0.1:1/none"
        pool = AsyncMock()
        pool.fetchval.return_value = True
        pool.execute.side_effect = asyncpg.InsufficientPrivilegeError("must be owner")
        with (
            patch("stratakb.core.migrations.get_pool", new_callable=AsyncMock, return_value=pool),
            pytest.raises(migrations.MigrationError, match="04_ingested_files.sql"),
        ):
            await migrations.ensure_ingestion_schema(url)
        assert url not in migrations._checked
