"""Tests for scripts/backup_restore.py layer export and import.

Most database tests read the export with asyncpg COPY and run the import steps
through asyncpg (_apply), since psql is not on every machine. The tests at the
end run the commands through real psql; they skip without it, and fail in CI.
Embeddings are mocked.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from stratakb.knowledge import add_folder
from stratakb.knowledge.add_folder import Options
from tests.helpers import (
    docs_folder,
    fake_embed,
    folder_settings,
    unique_layer_name,
    write_file,
)

_spec = importlib.util.spec_from_file_location(
    "backup_restore", Path(__file__).parent.parent / "scripts" / "backup_restore.py"
)
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)

pytestmark = pytest.mark.usefixtures("isolated_git")


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


def _import_args(tmp_path, data: bytes, root=None) -> SimpleNamespace:
    src = tmp_path / "layer.sql.gz"
    with gzip.open(src, "wb") as f:
        f.write(data)
    return SimpleNamespace(input=str(src), yes=True, root=root)


def test_export_format_reads_the_header():
    assert br.export_format(b"-- StrataKB Layer Export\n-- Format: 2\nBEGIN;\n") == 2
    assert br.export_format(b"-- StrataKB Layer Export\n-- Format: 12\n") == 12
    assert br.export_format(b"-- StrataKB Layer Export\n-- Layer: x\nBEGIN;\n") is None
    assert br.export_format(b"1\n2\n3\n4\n5\n6\n-- Format: 2\n") is None
    with pytest.raises(ValueError):
        br.export_format(b"-- Format: two\n")


def _layer_part(name="x", root="/old/docs") -> bytes:
    """One knowledge_layers row in COPY text."""
    values = {c: "\\N" for c in br.LAYERS.columns}
    values.update(name=name, root_path=root if root is not None else "\\N")
    return "\t".join(values[c] for c in br.LAYERS.columns).encode() + b"\n"


def _export_file(layer_part: bytes = b"", name="x") -> bytes:
    return br.build_layer_export(name, "0", [layer_part] + [b""] * (len(br.EXPORT_TABLES) - 1))


def test_export_layer_row():
    layer = br.export_layer_row(_export_file(_layer_part(root="/old/a\\\\b\\tc")))
    assert layer == br.ExportedLayer("x", "/old/a\\b\tc")
    assert br.export_layer_row(_export_file(_layer_part(root=None))).root_path is None
    assert br.export_layer_row(_export_file(_layer_part(name="\\N"))) is None
    assert br.export_layer_row(_export_file()) is None
    assert br.export_layer_row(b"-- Layer: x\nBEGIN;\nCOMMIT;\n") is None


@pytest.mark.parametrize("root", ["/plain/docs", "/a\\b\tc\nd\re", "\\N", "/ελληνικά/docs"])
def test_row_with_root_writes_copy_text(root):
    data = _export_file(_layer_part())
    columns, start, end = br._layer_row_span(data)
    moved = data[:start] + br.row_with_root(data[start:end], columns, root) + data[end:]
    assert br.export_layer_row(moved) == br.ExportedLayer("x", root)


def test_steps_serialize_to_what_psql_reads():
    data = br.serialize_steps(["BEGIN;", br.CopyBlock("t", ("a", "b"), b"1\t2\n"), "COMMIT;"])
    assert data == b"BEGIN;\nCOPY t (a, b) FROM stdin;\n1\t2\n\\.\nCOMMIT;\n"


def _recording_psql(monkeypatch, stdout=b"", code=0):
    """Patch _psql; the returned list collects (sql, args, stdin) of every call."""
    calls = []

    def fake(sql="", args=None, stdin=None):
        calls.append((sql, args, stdin))
        return subprocess.CompletedProcess(args, code, stdout, b"ERROR:  boom" if code else b"")

    monkeypatch.setattr(br, "_psql", fake)
    return calls


@pytest.mark.parametrize("header", [b"-- Format: 3", b"-- Format: three"])
def test_import_refuses_a_format_it_cannot_read(tmp_path, monkeypatch, capsys, header):
    calls = _recording_psql(monkeypatch)
    data = b"-- StrataKB Layer Export\n" + header + b"\n-- Layer: x\nBEGIN;\nCOMMIT;\n"
    with pytest.raises(SystemExit) as e:
        br.cmd_import_layer(_import_args(tmp_path, data))
    assert e.value.code == 1
    assert calls == []
    out = capsys.readouterr()
    assert "older version" not in out.out
    assert "Import failed: nothing was changed." in out.err
    assert ("export format 3" in out.err) if header.endswith(b"3") else ("format" in out.err)


def test_import_names_the_layer_holding_the_root(tmp_path, monkeypatch, capsys):
    calls = _recording_psql(monkeypatch, stdout=b"other_layer\n")
    with pytest.raises(SystemExit) as e:
        br.cmd_import_layer(_import_args(tmp_path, _export_file(_layer_part())))
    assert e.value.code == 1
    assert len(calls) == 1 and calls[0][2] is None
    assert "'/old/docs'" in calls[0][0] and "name <> 'x'" in calls[0][0]
    err = capsys.readouterr().err
    assert "'other_layer'" in err and "/old/docs" in err
    assert "Import failed: nothing was changed." in err


def test_import_root_moves_the_layer_row_and_its_files(tmp_path, monkeypatch):
    calls = _recording_psql(monkeypatch)
    new_root = tmp_path / "moved"
    new_root.mkdir()
    br.cmd_import_layer(_import_args(tmp_path, _export_file(_layer_part()), root=str(new_root)))
    (check_sql, _, _), (_, args, stdin) = calls
    expected = new_root.resolve().as_posix()
    assert f"'{expected}'" in check_sql and "'/old/docs'" not in check_sql
    assert args == br.PSQL_STRICT
    assert br.export_layer_row(stdin) == br.ExportedLayer("x", expected)
    assert stdin.endswith(b"COMMIT;\n")
    assert stdin.count(b"COMMIT;") == 1
    assert stdin.count(f"SET source_root = '{expected}'".encode()) == 2
    assert stdin.count(b"source_root = '/old/docs'") == 2


@pytest.mark.parametrize("data", [
    b"-- StrataKB Layer Export\n-- Layer: x\nBEGIN;\nCOMMIT;\n",
    _export_file(_layer_part(root=None)),
    _export_file(_layer_part()) + b"SELECT 1;\n",
])
def test_import_root_is_refused_where_it_cannot_apply(tmp_path, monkeypatch, capsys, data):
    calls = _recording_psql(monkeypatch)
    with pytest.raises(SystemExit):
        br.cmd_import_layer(_import_args(tmp_path, data, root=str(tmp_path)))
    assert calls == []
    assert "--root" in capsys.readouterr().err


def test_import_root_must_be_a_folder(tmp_path, monkeypatch, capsys):
    calls = _recording_psql(monkeypatch)
    missing = tmp_path / "missing"
    with pytest.raises(SystemExit):
        br.cmd_import_layer(_import_args(tmp_path, _export_file(_layer_part()), root=str(missing)))
    assert calls == []
    assert "must be a folder" in capsys.readouterr().err


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


async def _add(db_url, root, name):
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.refusal is None, p.refusal
    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = fake_embed
        await add_folder.apply(db_url, folder_settings("tester"), p, confirmed=set())


async def _export(conn, layer: str) -> list:
    """The steps cmd_export_layer writes, with asyncpg COPY in place of psql."""
    parts = []
    async with conn.transaction(isolation="repeatable_read", readonly=True):
        for q in br.export_queries(layer):
            buf = io.BytesIO()
            await conn.copy_from_query(q, output=buf, format="text")
            parts.append(buf.getvalue())
    return br.layer_import_steps(layer, parts)


async def _apply(conn, steps: list) -> None:
    """Run import steps through asyncpg."""
    for step in steps:
        if isinstance(step, br.CopyBlock):
            schema, _, table = step.table.rpartition(".")
            await conn.copy_to_table(
                table, schema_name=schema or None, columns=list(step.columns),
                source=io.BytesIO(step.rows), format="text")
        else:
            await conn.execute(step)


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


async def test_round_trip_keeps_tracking_and_text(folder_db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import delete_layer

    root = docs_folder(tmp_path)
    name = unique_layer_name()
    cleanup.append(name)
    await _add(folder_db_url, root, name)

    conn = await asyncpg.connect(folder_db_url)
    try:
        await conn.execute(
            "UPDATE strata_core.knowledge_layers SET answer_mode = 'extractive', "
            "structure_patterns = '[\"^(?P<label>AT-\\\\d{2})\"]' WHERE name = $1",
            name)
        before = await _snapshot(conn, name)
        assert before["layer"]["origin"] == "folder" and before["layer"]["root_path"]
        assert len(before["files"]) == 3 and before["documents"] and before["sections"]
        assert all(c[10] is not None and c[14] is not None for c in before["chunks"])
        data = await _export(conn, name)
        await delete_layer(folder_db_url, name=name)
        await _apply(conn, data)
        after = await _snapshot(conn, name)
    finally:
        await conn.close()

    assert after == before
    assert after["layer"]["answer_mode"] == "extractive"
    assert json.loads(after["layer"]["structure_patterns"]) == ["^(?P<label>AT-\\d{2})"]

    p = await add_folder.plan(folder_db_url, root, Options(name=name))
    assert p.action == "sync"
    assert p.planned == {"added": 0, "updated": 0, "unchanged": 3, "removed": 0}
    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as embed:
        report = await add_folder.apply(
            folder_db_url, folder_settings("tester"), p, confirmed=set()
        )
    embed.assert_not_called()
    assert report.counts["unchanged"] == 3


async def test_import_remaps_colliding_document_ids(folder_db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import create_layer, delete_layer

    root = docs_folder(tmp_path)
    name_a, name_b = unique_layer_name(), unique_layer_name()
    cleanup.extend([name_a, name_b])
    await _add(folder_db_url, root, name_a)

    conn = await asyncpg.connect(folder_db_url)
    try:
        before = await _snapshot(conn, name_a)
        old_ids = [r["id"] for r in await conn.fetch(
            "SELECT d.id FROM strata_vectors.documents d JOIN strata_core.knowledge_layers l "
            "ON l.id = d.layer_id WHERE l.name = $1", name_a)]
        data = await _export(conn, name_a)
        await delete_layer(folder_db_url, name=name_a)

        layer_b = await create_layer(folder_db_url, name=name_b, display_name="B")
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


async def test_export_covers_every_column(folder_db_url):
    skipped = {("strata_vectors.chunks", "tsv"), ("strata_vectors.document_sections", "id")}
    conn = await asyncpg.connect(folder_db_url)
    try:
        for table in br.EXPORT_TABLES:
            schema, _, name = table.name.partition(".")
            actual = {r["column_name"] for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = $1 AND table_name = $2", schema, name)}
            expected = actual - {c for t, c in skipped if t == table.name}
            assert set(table.columns) == expected, table.name
    finally:
        await conn.close()


async def _assert_folder_unchanged(db_url, root: Path, name: str) -> None:
    """Adding the folder again recognizes every file and embeds nothing."""
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.refusal is None, p.refusal
    assert p.action == "sync"
    assert p.planned == {"added": 0, "updated": 0, "unchanged": 3, "removed": 0}
    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as embed:
        report = await add_folder.apply(db_url, folder_settings("tester"), p, confirmed=set())
    embed.assert_not_called()
    assert report.counts["unchanged"] == 3


def _moved_steps(steps: list, name: str, old_root: str, new_root: str) -> list:
    """Import steps with the row and the statements _with_new_root writes into the file."""
    moved = [step._replace(rows=br.row_with_root(step.rows[:-1], step.columns, new_root) + b"\n")
             if isinstance(step, br.CopyBlock) and step.table == br.LAYERS.name else step
             for step in steps]
    return [*moved[:-1], *br.source_root_statements(name, old_root, new_root), moved[-1]]


async def test_new_root_moves_the_layer(folder_db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import create_layer, delete_layer, get_layer

    root = docs_folder(tmp_path)
    old_root = root.resolve().as_posix()
    moved = tmp_path / "elsewhere" / "docs"
    moved.parent.mkdir()
    name, holder = unique_layer_name(), unique_layer_name()
    cleanup.extend([name, holder])
    await _add(folder_db_url, root, name)

    conn = await asyncpg.connect(folder_db_url)
    try:
        steps = await _export(conn, name)
        await delete_layer(folder_db_url, name=name)
        # The folder the file records belongs to another layer by now
        await create_layer(folder_db_url, name=holder, display_name="Holder", root_path=old_root)
        root.rename(moved)
        new_root = moved.resolve().as_posix()
        await _apply(conn, _moved_steps(steps, name, old_root, new_root))
        for table in ("strata_core.ingested_files", "strata_vectors.documents"):
            roots = {r[0] for r in await conn.fetch(f"SELECT DISTINCT source_root FROM {table} "
                                                    "WHERE layer_id = (SELECT id FROM "
                                                    "strata_core.knowledge_layers WHERE name = $1)",
                                                    name)}
            assert roots == {new_root}, table
    finally:
        await conn.close()

    assert (await get_layer(folder_db_url, name=name))["root_path"] == new_root
    await _assert_folder_unchanged(folder_db_url, moved, name)


async def test_root_holder_query(folder_db_url, tmp_path, cleanup):
    root = docs_folder(tmp_path)
    name = unique_layer_name()
    cleanup.append(name)
    await _add(folder_db_url, root, name)
    root_path = root.resolve().as_posix()

    conn = await asyncpg.connect(folder_db_url)
    try:
        assert await conn.fetchval(br.root_holder_query(root_path, "another_layer")) == name
        assert await conn.fetchval(br.root_holder_query(root_path, name)) is None
    finally:
        await conn.close()


# --- real psql ----------------------------------------------------------------


@pytest.fixture
def real_psql(db_url, tmp_path, monkeypatch):
    """Point the script's psql calls at the test database, without docker exec."""
    psql = shutil.which("psql")
    if psql is None:
        if os.environ.get("CI"):
            pytest.fail("CI must run the real psql tests: psql is not on the PATH")
        pytest.skip("psql is not on the PATH")

    def run(cmd, capture=False, stdin_data=None):
        return subprocess.run([psql, *cmd[1:]], capture_output=capture, input=stdin_data)

    monkeypatch.setattr(br, "_run_docker", run)
    monkeypatch.setattr(br, "DB_NAME", db_url)
    monkeypatch.setattr(br, "BACKUP_DIR", tmp_path / "backups")


def _export_through_psql(tmp_path, name: str) -> Path:
    out = tmp_path / f"{name}.sql.gz"
    br.cmd_export_layer(SimpleNamespace(layer_name=name, output=str(out)))
    return out


async def test_real_psql_round_trip(folder_db_url, real_psql, tmp_path, cleanup):
    from stratakb.knowledge.layers import delete_layer

    root = docs_folder(tmp_path)
    name = unique_layer_name()
    cleanup.append(name)
    await _add(folder_db_url, root, name)

    conn = await asyncpg.connect(folder_db_url)
    try:
        # Characters COPY text escapes, and a line that looks like the end of a block
        await conn.execute(
            "UPDATE strata_vectors.chunks "
            "SET content = E'tab\\there\\nback\\\\slash\\n\\\\.\\nend' "
            "WHERE layer_name = $1", name)
        before = await _snapshot(conn, name)
        src = _export_through_psql(tmp_path, name)
        await delete_layer(folder_db_url, name=name)
        br.cmd_import_layer(SimpleNamespace(input=str(src), yes=True, root=None))
        after = await _snapshot(conn, name)
    finally:
        await conn.close()

    assert after == before
    assert "back\\slash\n\\.\nend" in after["chunks"][0][2]
    await _assert_folder_unchanged(folder_db_url, root, name)


async def test_real_psql_import_onto_a_moved_folder(folder_db_url, real_psql, tmp_path, cleanup):
    root = docs_folder(tmp_path)
    moved = tmp_path / "elsewhere" / "docs"
    moved.parent.mkdir()
    name = unique_layer_name()
    cleanup.append(name)
    await _add(folder_db_url, root, name)

    src = _export_through_psql(tmp_path, name)
    root.rename(moved)
    br.cmd_import_layer(SimpleNamespace(input=str(src), yes=True, root=str(moved)))
    await _assert_folder_unchanged(folder_db_url, moved, name)


async def test_real_psql_root_conflict_names_the_layer(
    folder_db_url, real_psql, tmp_path, cleanup, capsys
):
    root = docs_folder(tmp_path)
    other_root = tmp_path / "other"
    write_file(other_root / "a.md", "# Other\n\nOther text.\n")
    name, other = unique_layer_name(), unique_layer_name()
    cleanup.extend([name, other])
    await _add(folder_db_url, root, name)
    await _add(folder_db_url, other_root, other)

    src = _export_through_psql(tmp_path, name)
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        br.cmd_import_layer(SimpleNamespace(input=str(src), yes=True, root=str(other_root)))
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert f"'{other}'" in err and "already holds" in err

    # Nothing was changed: the file's own folder still imports
    br.cmd_import_layer(SimpleNamespace(input=str(src), yes=True, root=None))
    await _assert_folder_unchanged(folder_db_url, root, name)


async def test_real_psql_root_moves_off_a_folder_another_layer_holds(
    folder_db_url, real_psql, tmp_path, cleanup, capsys
):
    from stratakb.knowledge.layers import delete_layer, get_layer

    root = docs_folder(tmp_path)
    moved = tmp_path / "elsewhere" / "docs"
    name, holder = unique_layer_name(), unique_layer_name()
    cleanup.extend([name, holder])
    await _add(folder_db_url, root, name)
    src = _export_through_psql(tmp_path, name)
    await delete_layer(folder_db_url, name=name)
    await _add(folder_db_url, root, holder)
    shutil.copytree(root, moved)

    capsys.readouterr()
    with pytest.raises(SystemExit):
        br.cmd_import_layer(SimpleNamespace(input=str(src), yes=True, root=None))
    err = capsys.readouterr().err
    assert f"'{holder}'" in err and "--root" in err and "duplicate key" not in err

    br.cmd_import_layer(SimpleNamespace(input=str(src), yes=True, root=str(moved)))
    assert (await get_layer(folder_db_url, name=name))["root_path"] == moved.resolve().as_posix()
    assert (await get_layer(folder_db_url, name=holder))["root_path"] == root.resolve().as_posix()
    await _assert_folder_unchanged(folder_db_url, moved, name)


async def test_real_psql_failed_import_changes_nothing(
    folder_db_url, real_psql, tmp_path, cleanup, capsys
):
    root = docs_folder(tmp_path)
    name = unique_layer_name()
    cleanup.append(name)
    await _add(folder_db_url, root, name)

    conn = await asyncpg.connect(folder_db_url)
    try:
        before = await _snapshot(conn, name)
        src = _export_through_psql(tmp_path, name)
        with gzip.open(src, "rb") as f:
            data = f.read()
        broken = data[:-len(b"COMMIT;\n")] + b"SELECT 1 / 0;\nCOMMIT;\n"
        with pytest.raises(SystemExit):
            br.cmd_import_layer(_import_args(tmp_path, broken))
        assert await _snapshot(conn, name) == before
    finally:
        await conn.close()
    assert "Import failed: nothing was changed." in capsys.readouterr().err
