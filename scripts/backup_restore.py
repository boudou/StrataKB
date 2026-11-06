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
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKUP_DIR = ROOT / "backups"

DB_CONTAINER = os.environ.get("DB_CONTAINER", "strata-postgres")
DB_NAME = os.environ.get("POSTGRES_DB", "strata")
DB_USER = os.environ.get("POSTGRES_USER", "strata")


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

# The tables a layer export copies, in an order the foreign keys allow, with
# every column the import writes back. chunks.tsv (generated) and
# document_sections.id (identity) are left out: the target regenerates them.
EXPORT_FORMAT = "2"
EXPORT_TABLES: list[tuple[str, tuple[str, ...]]] = [
    ("strata_core.knowledge_layers", (
        "id", "name", "display_name", "description", "metadata_schema", "chunk_config",
        "routing_patterns", "status", "doc_count", "chunk_count", "last_ingested",
        "created_at", "updated_at", "origin", "owner", "root_path", "answer_mode",
        "structure_patterns")),
    ("strata_core.ingestion_runs", (
        "id", "layer_id", "source_path", "status", "docs_processed", "docs_failed",
        "chunks_created", "error_log", "started_at", "completed_at")),
    ("strata_core.ingested_files", (
        "id", "layer_id", "source_root", "relative_path", "content_hash", "size_bytes",
        "chunk_count", "ingested_at")),
    ("strata_vectors.documents", (
        "id", "layer_id", "file_id", "source_root", "relative_path", "version", "text",
        "sha256", "page_map", "text_origin", "ingested_at")),
    ("strata_vectors.document_sections", (
        "document_id", "label", "title", "path", "depth", "char_start", "char_end")),
    ("strata_vectors.chunks", (
        "id", "layer_name", "content", "embedding", "metadata", "source_file",
        "source_section", "doc_type", "version", "created_at", "file_id", "document_id",
        "char_start", "char_end", "clause_path")),
]


def _sql_str(value: str) -> str:
    """A SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def export_queries(layer: str) -> list[str]:
    """One SELECT per EXPORT_TABLES entry, limited to the layer."""
    name = _sql_str(layer)
    layer_id = f"(SELECT id FROM strata_core.knowledge_layers WHERE name = {name})"
    where = {
        "strata_core.knowledge_layers": f"name = {name}",
        "strata_core.ingestion_runs": f"layer_id = {layer_id}",
        "strata_core.ingested_files": f"layer_id = {layer_id}",
        "strata_vectors.documents": f"layer_id = {layer_id}",
        "strata_vectors.document_sections": (
            f"document_id IN (SELECT id FROM strata_vectors.documents "
            f"WHERE layer_id = {layer_id})"),
        "strata_vectors.chunks": f"layer_name = {name}",
    }
    return [f"SELECT {', '.join(cols)} FROM {table} WHERE {where[table]}"
            for table, cols in EXPORT_TABLES]


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


def build_layer_export(layer: str, chunk_count: str, parts: list[bytes]) -> bytes:
    """The import script: replaces the layer in one transaction.

    Document ids are numbered per database, so documents are staged and
    inserted with new ids, and sections and chunks are remapped to them
    through the documents' natural key. Layer, file and chunk ids are UUIDs
    and are kept.
    """
    (layers_t, layer_cols), (runs_t, run_cols), (files_t, file_cols), \
        (docs_t, doc_cols), (sections_t, section_cols), (chunks_t, chunk_cols) = EXPORT_TABLES
    name = _sql_str(layer)

    def cols(names, prefix=""):
        return ", ".join(prefix + c for c in names)

    doc_data_cols = [c for c in doc_cols if c != "id"]
    section_select = ", ".join("m.new_id" if c == "document_id" else f"s.{c}"
                               for c in section_cols)
    chunk_select = ", ".join("m.new_id" if c == "document_id" else f"c.{c}"
                             for c in chunk_cols)

    out: list[bytes] = []

    def sql(text: str):
        out.append(text.encode("utf-8") + b"\n")

    def copy_block(table: str, names, part: bytes):
        sql(f"COPY {table} ({cols(names)}) FROM stdin;")
        out.append(part)
        out.append(b"\\.\n")

    sql("-- StrataKB Layer Export")
    sql(f"-- Format: {EXPORT_FORMAT}")
    sql(f"-- Layer: {layer}")
    sql(f"-- Date: {datetime.now().isoformat()}")
    sql(f"-- Chunks: {chunk_count}")
    sql("BEGIN;")
    # Chunks first: chunks.document_id has no ON DELETE. Deleting the layer
    # row cascades to its runs, files, documents and their sections.
    sql(f"DELETE FROM {chunks_t} WHERE layer_name = {name};")
    sql(f"DELETE FROM {layers_t} WHERE name = {name};")
    copy_block(layers_t, layer_cols, parts[0])
    copy_block(runs_t, run_cols, parts[1])
    copy_block(files_t, file_cols, parts[2])
    sql(f"CREATE TEMP TABLE _layer_documents ON COMMIT DROP AS "
        f"SELECT {cols(doc_cols)} FROM {docs_t} WITH NO DATA;")
    copy_block("_layer_documents", doc_cols, parts[3])
    sql(f"INSERT INTO {docs_t} ({cols(doc_data_cols)}) "
        f"SELECT {cols(doc_data_cols)} FROM _layer_documents;")
    sql(f"CREATE TEMP TABLE _layer_document_ids ON COMMIT DROP AS "
        f"SELECT o.id AS old_id, d.id AS new_id FROM _layer_documents o "
        f"JOIN {docs_t} d ON d.layer_id = o.layer_id AND d.source_root = o.source_root "
        f"AND d.relative_path = o.relative_path AND d.version = o.version;")
    sql(f"CREATE TEMP TABLE _layer_sections ON COMMIT DROP AS "
        f"SELECT {cols(section_cols)} FROM {sections_t} WITH NO DATA;")
    copy_block("_layer_sections", section_cols, parts[4])
    sql(f"INSERT INTO {sections_t} ({cols(section_cols)}) SELECT {section_select} "
        f"FROM _layer_sections s JOIN _layer_document_ids m ON m.old_id = s.document_id;")
    sql(f"CREATE TEMP TABLE _layer_chunks ON COMMIT DROP AS "
        f"SELECT {cols(chunk_cols)} FROM {chunks_t} WITH NO DATA;")
    copy_block("_layer_chunks", chunk_cols, parts[5])
    sql(f"INSERT INTO {chunks_t} ({cols(chunk_cols)}) SELECT {chunk_select} "
        f"FROM _layer_chunks c LEFT JOIN _layer_document_ids m ON m.old_id = c.document_id;")
    sql("COMMIT;")
    return b"".join(out)


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
    r = _psql(args=["-q", "-X", "-v", "ON_ERROR_STOP=1"],
              stdin=copy_out_script(queries).encode("utf-8"))
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

def cmd_import_layer(args):
    """Import a layer from an export file."""
    src = Path(args.input)
    if not src.exists():
        print(f"Not found: {src}", file=sys.stderr); sys.exit(1)

    if not args.yes:
        if input(f"Import from {src.name}? Replaces existing layer data. [y/N] ").lower() != "y":
            print("Aborted."); return

    with gzip.open(src, "rb") as f:
        sql = f.read()
    print(f"Importing {len(sql) / 1024 / 1024:.1f} MB...")

    if f"-- Format: {EXPORT_FORMAT}".encode() in sql[:1024].split(b"\n")[:6]:
        r = _psql(args=["-q", "-X", "-v", "ON_ERROR_STOP=1"], stdin=sql)
        if r.returncode != 0:
            print(r.stderr.decode(errors="replace"), file=sys.stderr)
            print("Import failed: nothing was changed.", file=sys.stderr)
            sys.exit(1)
        print("Import complete.")
        return

    print("This export was made by an older version: it has no file tracking, origin or "
          "stored text, so the next ingestion rebuilds the layer.")
    r = _psql(args=["-v", "ON_ERROR_STOP=0"], stdin=sql)
    _report_errors(r)
    print("Import complete.")


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
    s.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    args = p.parse_args()
    {"backup": cmd_backup, "restore": cmd_restore, "list": cmd_list,
     "stats": cmd_stats, "export-layer": cmd_export_layer,
     "import-layer": cmd_import_layer}[args.cmd](args)


if __name__ == "__main__":
    main()
