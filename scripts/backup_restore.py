#!/usr/bin/env python3
"""StrataKB database backup & restore tool.

All knowledge (schemas, layers, embeddings, configs) lives in PostgreSQL.
This tool handles full backups and per-layer exports using pg_dump/psql
inside the Docker container.

Usage:
    uv run python scripts/backup_restore.py backup
    uv run python scripts/backup_restore.py backup -o my_backup.sql.gz
    uv run python scripts/backup_restore.py restore
    uv run python scripts/backup_restore.py restore -i backups/strata_2026-03-29.sql.gz
    uv run python scripts/backup_restore.py list
    uv run python scripts/backup_restore.py stats
    uv run python scripts/backup_restore.py export-layer sepa
    uv run python scripts/backup_restore.py import-layer -i backups/layer_sepa_2026-03-29.sql.gz
    uv run python scripts/backup_restore.py import-layer -i layer_sepa.sql.gz --root /new/docs

restore and import-layer feed the file to psql with the database owner's rights,
so a crafted file can do anything in the database. Use only files you made or trust.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, NoReturn

from stratakb.knowledge.layers import LAYER_COLUMN_NAMES

ROOT = Path(__file__).parent.parent
BACKUP_DIR = ROOT / "backups"

DB_CONTAINER = os.environ.get("DB_CONTAINER", "strata-postgres")
DB_NAME = os.environ.get("POSTGRES_DB", "strata")
DB_USER = os.environ.get("POSTGRES_USER", "strata")

# psql for a script that must run whole: -X reads no psqlrc that could change it,
# -q keeps command tags off stdout (the export reads its data from there), and
# ON_ERROR_STOP stops at the first error.
PSQL_STRICT = ["-q", "-X", "-v", "ON_ERROR_STOP=1"]


def _run_docker(cmd: list[str], capture=False, stdin_data: bytes | None = None):
    """Execute command inside the postgres container."""
    full = ["docker", "exec"]
    if stdin_data is not None:
        full.append("-i")
    full.append(DB_CONTAINER)
    full.extend(cmd)
    return subprocess.run(
        full,
        capture_output=capture,
        input=stdin_data,
    )


def _pg_dump(args: list[str]) -> subprocess.CompletedProcess:
    return _run_docker(["pg_dump", "-U", DB_USER, "-d", DB_NAME] + args, capture=True)


def _psql(sql: str = "", args: list[str] | None = None, stdin: bytes | None = None):
    cmd = ["psql", "-U", DB_USER, "-d", DB_NAME]
    if args:
        cmd.extend(args)
    if sql:
        cmd.extend(["-c", sql])
    return _run_docker(cmd, capture=True, stdin_data=stdin)


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


# ── backup ────────────────────────────────────────────────────────────

def cmd_backup(args):
    """Full database backup (pg_dump, gzipped)."""
    BACKUP_DIR.mkdir(exist_ok=True)
    out = Path(args.output) if args.output else BACKUP_DIR / f"strata_{_timestamp()}.sql.gz"

    print(f"Backing up '{DB_NAME}' from container '{DB_CONTAINER}'...")

    r = _pg_dump(["--no-owner", "--no-acl", "--clean", "--if-exists"])
    if r.returncode != 0:
        print(f"pg_dump failed:\n{r.stderr.decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)

    raw = len(r.stdout)
    with gzip.open(out, "wb", compresslevel=6) as f:
        f.write(r.stdout)
    compressed = out.stat().st_size

    print(f"Done: {out}")
    print(f"  {raw / 1024 / 1024:.1f} MB raw -> {compressed / 1024 / 1024:.1f} MB compressed "
          f"({(1 - compressed / raw) * 100:.0f}% reduction)")


# ── restore ───────────────────────────────────────────────────────────

def cmd_restore(args):
    """Restore full database from backup."""
    if args.input:
        src = Path(args.input)
    else:
        BACKUP_DIR.mkdir(exist_ok=True)
        candidates = sorted(BACKUP_DIR.glob("strata_*.sql.gz"))
        if not candidates:
            print("No backups found in backups/"); sys.exit(1)
        src = candidates[-1]
        print(f"Using latest: {src.name}")

    if not src.exists():
        print(f"Not found: {src}", file=sys.stderr); sys.exit(1)

    if not args.yes:
        if input(f"REPLACE current database with {src.name}? [y/N] ").lower() != "y":
            print("Aborted."); return

    with gzip.open(src, "rb") as f:
        sql = f.read()
    print(f"Restoring {len(sql) / 1024 / 1024:.1f} MB...")

    r = _psql(args=["-v", "ON_ERROR_STOP=0"], stdin=sql)
    _report_errors(r)
    print("Restore complete.")


# ── list ──────────────────────────────────────────────────────────────

def cmd_list(args):
    """List available backups."""
    BACKUP_DIR.mkdir(exist_ok=True)
    files = sorted(BACKUP_DIR.glob("*.sql.gz"))
    if not files:
        print("No backups found."); return
    print(f"Backups ({len(files)}):\n")
    for f in files:
        sz = f.stat().st_size / 1024 / 1024
        mt = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {f.name:50s} {sz:7.1f} MB  {mt}")


# ── stats ─────────────────────────────────────────────────────────────

def cmd_stats(args):
    """Show database size and layer statistics."""
    r = _psql("SELECT name, status, doc_count, chunk_count FROM strata_core.knowledge_layers ORDER BY name")
    if r.returncode == 0:
        print("Layers:")
        print(r.stdout.decode(errors="replace"))

    r = _psql("SELECT count(*) AS total_chunks FROM strata_vectors.chunks")
    if r.returncode == 0:
        print(r.stdout.decode(errors="replace").strip())

    r = _psql(f"SELECT pg_size_pretty(pg_database_size('{DB_NAME}')) AS db_size")
    if r.returncode == 0:
        print(r.stdout.decode(errors="replace").strip())

    r = _psql("SELECT pg_size_pretty(pg_total_relation_size('strata_vectors.chunks')) AS vectors_size")
    if r.returncode == 0:
        print(r.stdout.decode(errors="replace").strip())

    r = _psql(
        "SELECT layer_name, count(*) as chunks, "
        "pg_size_pretty(sum(pg_column_size(embedding))) as embedding_size "
        "FROM strata_vectors.chunks GROUP BY layer_name ORDER BY layer_name"
    )
    if r.returncode == 0:
        print("\nPer-layer storage:")
        print(r.stdout.decode(errors="replace"))


# ── export-layer ──────────────────────────────────────────────────────

EXPORT_FORMAT = 4


class ExportTable(NamedTuple):
    """A table a layer export copies.

    columns are the ones the import writes back. where limits the table to the
    layer, with {name} the quoted layer name and {layer_id} a subquery for its id.
    """

    name: str
    columns: tuple[str, ...]
    where: str


# In an order the foreign keys allow. chunks.tsv (generated) and
# document_sections.id (identity) are left out: the target regenerates them.
LAYERS = ExportTable("strata_core.knowledge_layers", LAYER_COLUMN_NAMES, "name = {name}")
RUNS = ExportTable(
    "strata_core.ingestion_runs",
    ("id", "layer_id", "source_path", "status", "docs_processed", "docs_failed",
     "chunks_created", "error_log", "started_at", "completed_at"),
    "layer_id = {layer_id}")
FILES = ExportTable(
    "strata_core.ingested_files",
    ("id", "layer_id", "source_root", "relative_path", "content_hash", "size_bytes",
     "chunk_count", "ingested_at"),
    "layer_id = {layer_id}")
DOCUMENTS = ExportTable(
    "strata_vectors.documents",
    ("id", "layer_id", "file_id", "source_root", "relative_path", "version", "text",
     "sha256", "page_map", "text_origin", "ingested_at", "doc_date", "doc_date_source",
     "supersedes"),
    "layer_id = {layer_id}")
SECTIONS = ExportTable(
    "strata_vectors.document_sections",
    ("document_id", "label", "title", "path", "depth", "char_start", "char_end"),
    "document_id IN (SELECT id FROM strata_vectors.documents WHERE layer_id = {layer_id})")
CHUNKS = ExportTable(
    "strata_vectors.chunks",
    ("id", "layer_name", "content", "embedding", "metadata", "source_file",
     "source_section", "doc_type", "version", "created_at", "file_id", "document_id",
     "char_start", "char_end", "clause_path"),
    "layer_name = {name}")
EXPORT_TABLES = (LAYERS, RUNS, FILES, DOCUMENTS, SECTIONS, CHUNKS)

# The last statement of an export; import-layer --root goes in front of it
COMMIT = "COMMIT;"


class CopyBlock(NamedTuple):
    """Rows in COPY text format, loaded into a table."""

    table: str
    columns: tuple[str, ...]
    rows: bytes


def _sql_str(value: str) -> str:
    """A SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _layer_id_sql(name: str) -> str:
    """A subquery for the id of the layer with this quoted name."""
    return f"(SELECT id FROM {LAYERS.name} WHERE name = {name})"


def export_queries(layer: str) -> list[str]:
    """One SELECT per EXPORT_TABLES entry, limited to the layer."""
    name = _sql_str(layer)
    layer_id = _layer_id_sql(name)
    return [f"SELECT {', '.join(t.columns)} FROM {t.name} "
            f"WHERE {t.where.format(name=name, layer_id=layer_id)}"
            for t in EXPORT_TABLES]


def copy_out_script(queries: list[str]) -> str:
    """A psql script printing each query as COPY text, each part ended by a line `\\.`.

    All reads share one read-only snapshot, so an ingestion running meanwhile
    cannot leave chunks whose file or document is missing from the export.
    """
    lines = ["BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;"]
    for q in queries:
        lines.append(f"COPY ({q}) TO STDOUT;")
        lines.append("\\echo '\\\\.'")
    lines.append("COMMIT;")
    return "\n".join(lines) + "\n"


def split_copy_output(stdout: bytes, n: int) -> list[bytes]:
    """Split copy_out_script output into its n COPY parts.

    Splits on whole lines equal to `\\.`: COPY text escapes a backslash, so a
    data line can end with `\\.` but never equal it.
    """
    parts: list[bytes] = []
    current: list[bytes] = []
    lines = stdout.split(b"\n")
    tail = lines.pop()
    for line in lines:
        if line == b"\\.":
            parts.append(b"".join(row + b"\n" for row in current))
            current = []
        else:
            current.append(line)
    if current or tail or len(parts) != n:
        raise ValueError(f"expected {n} COPY parts, got {len(parts)} and trailing data "
                         f"{bool(current or tail)}")
    return parts


def layer_import_steps(layer: str, parts: list[bytes]) -> list[str | CopyBlock]:
    """The import as SQL statements and COPY blocks: replaces the layer in one transaction.

    parts holds one COPY text part per EXPORT_TABLES entry. Document ids are
    numbered per database, so documents are staged and inserted with new ids,
    and sections and chunks are remapped to them through the documents' natural
    key. Layer, file and chunk ids are UUIDs and are kept.
    """
    rows = dict(zip(EXPORT_TABLES, parts, strict=True))
    name = _sql_str(layer)

    def cols(names):
        return ", ".join(names)

    def staged(temp: str, table: ExportTable) -> list[str | CopyBlock]:
        return [f"CREATE TEMP TABLE {temp} ON COMMIT DROP AS "
                f"SELECT {cols(table.columns)} FROM {table.name} WITH NO DATA;",
                CopyBlock(temp, table.columns, rows[table])]

    doc_data_cols = [c for c in DOCUMENTS.columns if c != "id"]
    section_select = ", ".join("m.new_id" if c == "document_id" else f"s.{c}"
                               for c in SECTIONS.columns)
    chunk_select = ", ".join("m.new_id" if c == "document_id" else f"c.{c}"
                             for c in CHUNKS.columns)

    return [
        "BEGIN;",
        # Chunks first: chunks.document_id has no ON DELETE. Deleting the layer
        # row cascades to its runs, files, documents and their sections.
        f"DELETE FROM {CHUNKS.name} WHERE layer_name = {name};",
        f"DELETE FROM {LAYERS.name} WHERE name = {name};",
        CopyBlock(LAYERS.name, LAYERS.columns, rows[LAYERS]),
        CopyBlock(RUNS.name, RUNS.columns, rows[RUNS]),
        CopyBlock(FILES.name, FILES.columns, rows[FILES]),
        *staged("_layer_documents", DOCUMENTS),
        f"INSERT INTO {DOCUMENTS.name} ({cols(doc_data_cols)}) "
        f"SELECT {cols(doc_data_cols)} FROM _layer_documents;",
        f"CREATE TEMP TABLE _layer_document_ids ON COMMIT DROP AS "
        f"SELECT o.id AS old_id, d.id AS new_id FROM _layer_documents o "
        f"JOIN {DOCUMENTS.name} d ON d.layer_id = o.layer_id AND d.source_root = o.source_root "
        f"AND d.relative_path = o.relative_path AND d.version = o.version;",
        *staged("_layer_sections", SECTIONS),
        f"INSERT INTO {SECTIONS.name} ({cols(SECTIONS.columns)}) SELECT {section_select} "
        f"FROM _layer_sections s JOIN _layer_document_ids m ON m.old_id = s.document_id;",
        *staged("_layer_chunks", CHUNKS),
        f"INSERT INTO {CHUNKS.name} ({cols(CHUNKS.columns)}) SELECT {chunk_select} "
        f"FROM _layer_chunks c LEFT JOIN _layer_document_ids m ON m.old_id = c.document_id;",
        COMMIT,
    ]


def serialize_steps(steps: list[str | CopyBlock]) -> bytes:
    """The bytes psql reads for these steps."""
    out: list[bytes] = []
    for step in steps:
        if isinstance(step, CopyBlock):
            out.append(f"COPY {step.table} ({', '.join(step.columns)}) FROM stdin;\n".encode())
            out.append(step.rows)
            out.append(b"\\.\n")
        else:
            out.append(step.encode("utf-8") + b"\n")
    return b"".join(out)


def build_layer_export(layer: str, chunk_count: str, parts: list[bytes]) -> bytes:
    """The export file: a header, then the import steps."""
    header = ["-- StrataKB Layer Export",
              f"-- Format: {EXPORT_FORMAT}",
              f"-- Layer: {layer}",
              f"-- Date: {datetime.now().isoformat()}",
              f"-- Chunks: {chunk_count}"]
    return serialize_steps([*header, *layer_import_steps(layer, parts)])


def cmd_export_layer(args):
    """Export a single layer (config, file tracking, stored text, chunks) using COPY."""
    BACKUP_DIR.mkdir(exist_ok=True)
    layer = args.layer_name
    out = Path(args.output) if args.output else BACKUP_DIR / f"layer_{layer}_{_timestamp()}.sql.gz"

    # Verify layer exists and get count
    r = _psql("SELECT chunk_count FROM strata_core.knowledge_layers "
              f"WHERE name = {_sql_str(layer)}", args=["-t", "-A"])
    count = r.stdout.decode().strip() if r.returncode == 0 else ""
    if not count:
        print(f"Layer '{layer}' not found."); sys.exit(1)
    print(f"Exporting layer '{layer}' ({count} chunks)...")

    queries = export_queries(layer)
    r = _psql(args=PSQL_STRICT, stdin=copy_out_script(queries).encode("utf-8"))
    if r.returncode != 0:
        print(f"Export failed:\n{r.stderr.decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    try:
        parts = split_copy_output(r.stdout, len(queries))
    except ValueError as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)
    data = build_layer_export(layer, count, parts)

    raw = len(data)
    with gzip.open(out, "wb", compresslevel=6) as f:
        f.write(data)
    compressed = out.stat().st_size

    print(f"Done: {out}")
    print(f"  {raw / 1024 / 1024:.1f} MB raw -> {compressed / 1024 / 1024:.1f} MB compressed")


# ── import-layer ──────────────────────────────────────────────────────

def export_format(data: bytes) -> int | None:
    """The format version in an export's header; None for a file with no such line.

    Raises ValueError when the line is there but holds no number.
    """
    for line in data[:1024].split(b"\n")[:6]:
        if line.startswith(b"-- Format:"):
            value = line.removeprefix(b"-- Format:").strip()
            if not value.isdigit():
                raise ValueError(f"unreadable format line: {line.decode(errors='replace')}")
            return int(value)
    return None


_COPY_ESCAPES = {b"b": b"\b", b"f": b"\f", b"n": b"\n", b"r": b"\r", b"t": b"\t", b"v": b"\v"}


def _copy_field(field: bytes) -> str | None:
    r"""One field of a COPY text row: \N is NULL, and a backslash starts an escape."""
    if field == b"\\N":
        return None
    unescaped = re.sub(rb"\\(.)", lambda m: _COPY_ESCAPES.get(m[1], m[1]), field, flags=re.S)
    return unescaped.decode("utf-8")


def _copy_text(value: str) -> bytes:
    """A string as one field of a COPY text row."""
    text = value.encode("utf-8").replace(b"\\", b"\\\\")
    for letter, char in _COPY_ESCAPES.items():
        text = text.replace(char, b"\\" + letter)
    return text


class ExportedLayer(NamedTuple):
    """What the checks in front of an import read from an export's layer row."""

    name: str
    root_path: str | None


def _layer_row_span(data: bytes) -> tuple[list[str], int, int] | None:
    """The layer row's columns and byte span in an export file; None when it holds none."""
    start = data.find(f"COPY {LAYERS.name} (".encode())
    if start < 0:
        return None
    columns = data[data.index(b"(", start) + 1:data.index(b")", start)].decode().split(", ")
    row_start = data.find(b"\n", start) + 1
    row_end = data.find(b"\n", row_start)
    if data[row_start:row_end].count(b"\t") != len(columns) - 1:
        return None
    return columns, row_start, row_end


def export_layer_row(data: bytes) -> ExportedLayer | None:
    """The layer row in an export file; None when it holds no readable one.

    For the checks in front of an import only. The file itself decides what
    psql then does.
    """
    span = _layer_row_span(data)
    if span is None:
        return None
    columns, start, end = span
    row = dict(zip(columns, map(_copy_field, data[start:end].split(b"\t")), strict=True))
    name = row.get("name")
    return ExportedLayer(name, row.get("root_path")) if name else None


def row_with_root(row: bytes, columns: Sequence[str], root: str) -> bytes:
    """A knowledge_layers COPY row carrying this root_path."""
    fields = row.split(b"\t")
    fields[columns.index("root_path")] = _copy_text(root)
    return b"\t".join(fields)


def root_holder_query(root: str, layer: str) -> str:
    """The SELECT finding another layer that holds this folder."""
    return (f"SELECT name FROM {LAYERS.name} "
            f"WHERE root_path = {_sql_str(root)} AND name <> {_sql_str(layer)}")


def source_root_statements(layer: str, old_root: str, new_root: str) -> list[str]:
    """Move an imported layer's tracked files and documents, in front of the export's COMMIT.

    Only rows of the layer's own root move. ingestion_runs.source_path stays:
    it is the history of where each run read from. chunks.source_file is
    relative to the root.
    """
    layer_id = _layer_id_sql(_sql_str(layer))
    return [f"UPDATE {table.name} SET source_root = {_sql_str(new_root)} "
            f"WHERE layer_id = {layer_id} AND source_root = {_sql_str(old_root)};"
            for table in (FILES, DOCUMENTS)]


def _with_new_root(sql: bytes, root: str) -> bytes:
    """The export file moved to root; exits when it cannot be moved.

    The layer row itself carries the new root: the unique index on root_path is
    checked as the row is copied, so the file's old root may be another layer's.
    """
    layer = export_layer_row(sql)
    if not layer or not layer.root_path:
        _refuse_import("--root needs a layer added from a folder; this file holds none.")
    if not Path(root).is_dir():
        _refuse_import(f"--root must be a folder on this machine: {root}")
    tail = f"{COMMIT}\n".encode()
    if not sql.endswith(tail):
        _refuse_import(f"--root needs a file ending in {COMMIT}; this one does not.")
    columns, start, end = _layer_row_span(sql)
    statements = source_root_statements(layer.name, layer.root_path, root)
    return (sql[:start] + row_with_root(sql[start:end], columns, root)
            + sql[end:-len(tail)] + serialize_steps(statements) + tail)


def _check_root_free(layer: str, root: str) -> None:
    """Exit when another layer holds the folder this import would give the layer."""
    r = _psql(root_holder_query(root, layer), args=[*PSQL_STRICT, "-t", "-A"])
    if r.returncode != 0:
        _refuse_import(r.stderr.decode(errors="replace"))
    holder = r.stdout.decode(errors="replace").strip()
    if holder:
        _refuse_import(f"The layer '{holder}' already holds the folder {root}. Delete that "
                       "layer, or import with --root and another folder.")


def _refuse_import(reason: str) -> NoReturn:
    print(reason, file=sys.stderr)
    print("Import failed: nothing was changed.", file=sys.stderr)
    sys.exit(1)


def _readable_format(src: Path, sql: bytes) -> int | None:
    """The file's export format, None for a legacy file; exits when this version cannot read it."""
    try:
        version = export_format(sql)
    except ValueError as e:
        _refuse_import(f"{src.name} is not a layer export this version can read: {e}.")
    if version is not None and version > EXPORT_FORMAT:
        _refuse_import(f"{src.name} has export format {version}, and this version reads up to "
                       f"format {EXPORT_FORMAT}. Update StrataKB and import it again.")
    return version


def _confirm_import(src: Path, yes: bool) -> bool:
    if yes:
        return True
    answer = input(f"Import from {src.name}? Replaces existing layer data and runs the file's "
                   "SQL with the database owner's rights, so use only files you made or trust. "
                   "[y/N] ")
    if answer.lower() != "y":
        print("Aborted.")
        return False
    return True


def _import_current(src: Path, sql: bytes, root: str | None, yes: bool) -> None:
    """Import a file with file tracking, in one transaction that stops at the first error."""
    if root:
        sql = _with_new_root(sql, root)
    layer = export_layer_row(sql)
    if layer and layer.root_path:
        _check_root_free(layer.name, layer.root_path)
    if not _confirm_import(src, yes):
        return
    print(f"Importing {len(sql) / 1024 / 1024:.1f} MB...")
    r = _psql(args=PSQL_STRICT, stdin=sql)
    if r.returncode != 0:
        _refuse_import(r.stderr.decode(errors="replace"))
    print(f"Import complete. The layer's folder is now {root}." if root else "Import complete.")


def _import_legacy(src: Path, sql: bytes, root: str | None, yes: bool) -> None:
    """Import a file from before format lines, which tolerates what already exists."""
    if root:
        _refuse_import("--root needs a newer export: this file has no file tracking to move.")
    if not _confirm_import(src, yes):
        return
    print(f"Importing {len(sql) / 1024 / 1024:.1f} MB...")
    print("This export was made by an older version: it has no file tracking, origin or "
          "stored text, so the next ingestion rebuilds the layer.")
    r = _psql(args=["-v", "ON_ERROR_STOP=0"], stdin=sql)
    _report_errors(r)
    print("Import complete.")


def cmd_import_layer(args):
    """Import a layer from an export file."""
    src = Path(args.input)
    if not src.exists():
        print(f"Not found: {src}", file=sys.stderr)
        sys.exit(1)

    with gzip.open(src, "rb") as f:
        sql = f.read()

    root =Path(args.root).resolve().as_posix() if args.root else None
    if _readable_format(src, sql) is None:
        _import_legacy(src, sql, root, args.yes)
    else:
        _import_current(src, sql, root, args.yes)


# ── helpers ───────────────────────────────────────────────────────────

def _report_errors(r: subprocess.CompletedProcess):
    if r.returncode != 0 and r.stderr:
        stderr = r.stderr.decode(errors="replace")
        errors = [l for l in stderr.splitlines()
                  if "ERROR" in l and "already exists" not in l and "does not exist" not in l]
        if errors:
            print("Errors (may be benign):")
            for e in errors[:10]:
                print(f"  {e}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="StrataKB database backup & restore")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("backup", help="Full database backup")
    s.add_argument("-o", "--output", help="Output path (default: backups/strata_<ts>.sql.gz)")

    s = sub.add_parser("restore", help="Restore from backup")
    s.add_argument("-i", "--input", help="Backup file (default: latest)")
    s.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    sub.add_parser("list", help="List backups")
    sub.add_parser("stats", help="Database statistics")

    s = sub.add_parser("export-layer", help="Export one layer")
    s.add_argument("layer_name")
    s.add_argument("-o", "--output")

    s = sub.add_parser("import-layer", help="Import a layer export")
    s.add_argument("-i", "--input", required=True)
    s.add_argument("--root", help="The folder's path on this machine, when it moved")
    s.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    args = p.parse_args()
    {"backup": cmd_backup, "restore": cmd_restore, "list": cmd_list,
     "stats": cmd_stats, "export-layer": cmd_export_layer,
     "import-layer": cmd_import_layer}[args.cmd](args)


if __name__ == "__main__":
    main()
