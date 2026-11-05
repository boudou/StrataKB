"""Tests for scripts/backup_restore.py layer export and import.

psql is not installed on the test machines, so the database tests read the
export with asyncpg COPY and apply the import file with a small stand-in for
psql (_apply). Embeddings are mocked.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib.util
import io
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import asyncpg
import pytest

from stratakb.knowledge import add_folder
from stratakb.knowledge.add_folder import Options

_spec = importlib.util.spec_from_file_location(
    "backup_restore", Path(__file__).parent.parent / "scripts" / "backup_restore.py"
)
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)

FAKE_EMBEDDING = [0.01] * 768


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch, tmp_path):
    """Keep the machine's git config and any repository above tmp_path out of the tests."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))


# --- pure functions -----------------------------------------------------------


def test_split_copy_output():
    out = b"a\tx\\\\.\nb\n\\.\n\\.\nc\n\\.\n"
    assert br.split_copy_output(out, 3) == [b"a\tx\\\\.\nb\n", b"", b"c\n"]
    with pytest.raises(ValueError):
        br.split_copy_output(out, 2)
    with pytest.raises(ValueError):
        br.split_copy_output(out + b"stray\n", 3)


def test_queries_quote_the_layer_name():
    queries = br.export_queries("it's")
    assert len(queries) == len(br.EXPORT_TABLES)
    assert all("'it''s'" in q for q in queries)
    script = br.copy_out_script(queries)
    assert script.startswith("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;")
    assert script.count("\\echo '\\\\.'") == len(queries)
    assert "\\pset" not in script


def _import_args(tmp_path, data: bytes) -> SimpleNamespace:
    src = tmp_path / "layer.sql.gz"
    with gzip.open(src, "wb") as f:
        f.write(data)
    return SimpleNamespace(input=str(src), yes=True)


def test_import_exit_codes(tmp_path, monkeypatch, capsys):
    calls = []

    def failing(sql="", args=None, stdin=None):
        calls.append(args)
        return subprocess.CompletedProcess(args, 3, b"", b"ERROR:  boom")

    monkeypatch.setattr(br, "_psql", failing)
    data = br.build_layer_export("x", "0", [b""] * len(br.EXPORT_TABLES))
    with pytest.raises(SystemExit) as e:
        br.cmd_import_layer(_import_args(tmp_path, data))
    assert e.value.code == 1
    assert "ON_ERROR_STOP=1" in calls[-1]
    out = capsys.readouterr()
    assert "Import complete." not in out.out
    assert "Import failed: nothing was changed." in out.err

    def ok(sql="", args=None, stdin=None):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(br, "_psql", ok)
    old = b"-- StrataKB Layer Export\n-- Layer: x\nBEGIN;\nCOMMIT;\n"
    br.cmd_import_layer(_import_args(tmp_path, old))
    assert "ON_ERROR_STOP=0" in calls[-1]
    out = capsys.readouterr().out
    assert "made by an older version" in out and "Import complete." in out


# --- database -----------------------------------------------------------------


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    with patch("stratakb.knowledge.add_folder.broad_reason", return_value=None):
        yield get_settings().database_url


@pytest.fixture
async def cleanup(db_url):
    """Layer names to delete after the test."""
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer

    names: list[str] = []
    yield names
    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    _write(root / "README.md", "# Team Docs\n\nHow the team works.\n")
    _write(root / "guide" / "setup.md",
           "# Setup\n\n## 1. Install\n\nRun the installer.\n\n## 2. Configure\n\nSet it up.\n")
    _write(root / "notes.md", "# Notes\n\nMeeting notes.\n")
    return root


async def _add(db_url, root, name):
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.refusal is None, p.refusal
    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda texts, **_: [FAKE_EMBEDDING for _ in texts]
        settings = SimpleNamespace(strata_user="tester", model_fields_set={"strata_user"},
                                   openai_api_key="fake-key")
        await add_folder.apply(db_url, settings, p, confirmed=set())


async def _export(conn, layer: str) -> bytes:
    """What cmd_export_layer writes, with asyncpg COPY in place of psql."""
    count = await conn.fetchval(
        "SELECT chunk_count FROM strata_core.knowledge_layers WHERE name = $1", layer)
    parts = []
    async with conn.transaction(isolation="repeatable_read", readonly=True):
        for q in br.export_queries(layer):
            buf = io.BytesIO()
            await conn.copy_from_query(q, output=buf, format="text")
            parts.append(buf.getvalue())
    return br.build_layer_export(layer, str(count), parts)


_COPY_LINE = re.compile(r"COPY (\S+) \(([^)]*)\) FROM stdin;")


async def _apply(conn, data: bytes) -> None:
    """Run an import file the way psql does: SQL text, and COPY blocks ended by `\\.`."""
    pending: list[str] = []
    lines = iter(data.split(b"\n"))
    for raw in lines:
        line = raw.decode("utf-8")
        m = _COPY_LINE.fullmatch(line)
        if not m:
            pending.append(line)
            continue
        if "".join(pending).strip():
            await conn.execute("\n".join(pending))
        pending = []
        block = []
        for row in lines:
            if row == b"\\.":
                break
            block.append(row + b"\n")
        schema, _, table = m.group(1).rpartition(".")
        await conn.copy_to_table(
            table, schema_name=schema or None, columns=m.group(2).split(", "),
            source=io.BytesIO(b"".join(block)), format="text")
    if "".join(pending).strip():
        await conn.execute("\n".join(pending))


async def _snapshot(conn, layer: str) -> dict:
    layer_row = await conn.fetchrow(
        "SELECT * FROM strata_core.knowledge_layers WHERE name = $1", layer)
    lid = layer_row["id"]
    doc_key = ("d.source_root, d.relative_path, d.version, d.sha256, d.text, d.file_id, "
               "d.page_map, d.text_origin, d.ingested_at")
    return {
        "layer": dict(layer_row),
        "runs": [dict(r) for r in await conn.fetch(
            "SELECT * FROM strata_core.ingestion_runs WHERE layer_id = $1 ORDER BY id", lid)],
        "files": [dict(r) for r in await conn.fetch(
            "SELECT * FROM strata_core.ingested_files WHERE layer_id = $1 ORDER BY id", lid)],
        "documents": [tuple(r) for r in await conn.fetch(
            f"SELECT {doc_key} FROM strata_vectors.documents d WHERE layer_id = $1 "
            "ORDER BY d.relative_path, d.version", lid)],
        "sections": [tuple(r) for r in await conn.fetch(
            "SELECT d.relative_path, d.version, s.label, s.title, s.path, s.depth, "
            "s.char_start, s.char_end FROM strata_vectors.document_sections s "
            "JOIN strata_vectors.documents d ON d.id = s.document_id WHERE d.layer_id = $1 "
            "ORDER BY 1, 2, s.char_start, s.label", lid)],
        "chunks": [tuple(r) for r in await conn.fetch(
            "SELECT c.id, c.layer_name, c.content, c.embedding::text, c.metadata, "
            "c.source_file, c.source_section, c.doc_type, c.version, c.created_at, "
            "c.file_id, c.char_start, c.char_end, c.clause_path, "
            f"{doc_key} FROM strata_vectors.chunks c "
            "LEFT JOIN strata_vectors.documents d ON d.id = c.document_id "
            "WHERE c.layer_name = $1 ORDER BY c.id", layer)],
    }


def _name() -> str:
    return f"test_br_{uuid4().hex[:10]}"


async def test_round_trip_keeps_tracking_and_text(db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import delete_layer

    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    await _add(db_url, root, name)

    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "UPDATE strata_core.knowledge_layers SET answer_mode = 'extractive' WHERE name = $1",
            name)
        before = await _snapshot(conn, name)
        assert before["layer"]["origin"] == "folder" and before["layer"]["root_path"]
        assert len(before["files"]) == 3 and before["documents"] and before["sections"]
        assert all(c[10] is not None and c[14] is not None for c in before["chunks"])
        data = await _export(conn, name)
        await delete_layer(db_url, name=name)
        await _apply(conn, data)
        after = await _snapshot(conn, name)
    finally:
        await conn.close()

    assert after == before
    assert after["layer"]["answer_mode"] == "extractive"

    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.action == "sync"
    assert p.planned == {"added": 0, "updated": 0, "unchanged": 3, "removed": 0}
    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as embed:
        settings = SimpleNamespace(strata_user="tester", model_fields_set={"strata_user"},
                                   openai_api_key="fake-key")
        report = await add_folder.apply(db_url, settings, p, confirmed=set())
    embed.assert_not_called()
    assert report.counts["unchanged"] == 3


async def test_import_remaps_colliding_document_ids(db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import create_layer, delete_layer

    root = _docs(tmp_path)
    name_a, name_b = _name(), _name()
    cleanup.extend([name_a, name_b])
    await _add(db_url, root, name_a)

    conn = await asyncpg.connect(db_url)
    try:
        before = await _snapshot(conn, name_a)
        old_ids = [r["id"] for r in await conn.fetch(
            "SELECT d.id FROM strata_vectors.documents d JOIN strata_core.knowledge_layers l "
            "ON l.id = d.layer_id WHERE l.name = $1", name_a)]
        data = await _export(conn, name_a)
        await delete_layer(db_url, name=name_a)

        layer_b = await create_layer(db_url, name=name_b, display_name="B")
        await conn.execute(
            "INSERT INTO strata_vectors.documents (id, layer_id, source_root, relative_path, "
            "version, text, sha256) OVERRIDING SYSTEM VALUE "
            "VALUES ($1, $2, 'C:/b', 'b.md', 1, 'bee', 'x')",
            old_ids[0], layer_b["id"])

        await _apply(conn, data)
        after = await _snapshot(conn, name_a)
        b_doc = await conn.fetchrow(
            "SELECT layer_id, text FROM strata_vectors.documents WHERE id = $1", old_ids[0])
        new_ids = {r["document_id"] for r in await conn.fetch(
            "SELECT document_id FROM strata_vectors.chunks WHERE layer_name = $1", name_a)}
    finally:
        await conn.close()

    assert after == before
    assert b_doc["layer_id"] == layer_b["id"] and b_doc["text"] == "bee"
    assert old_ids[0] not in new_ids


async def test_export_covers_every_column(db_url):
    skipped = {("strata_vectors.chunks", "tsv"), ("strata_vectors.document_sections", "id")}
    conn = await asyncpg.connect(db_url)
    try:
        for table, cols in br.EXPORT_TABLES:
            schema, _, name = table.partition(".")
            actual = {r["column_name"] for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = $1 AND table_name = $2", schema, name)}
            expected = actual - {c for t, c in skipped if t == table}
            assert set(cols) == expected, table
    finally:
        await conn.close()
