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

def cmd_export_layer(args):
    """Export a single layer (config + chunks) using COPY for speed."""
    BACKUP_DIR.mkdir(exist_ok=True)
    layer = args.layer_name
    out = Path(args.output) if args.output else BACKUP_DIR / f"layer_{layer}_{_timestamp()}.sql.gz"

    # Verify layer exists and get count
    r = _psql(f"SELECT chunk_count FROM strata_core.knowledge_layers WHERE name = '{layer}'",
              args=["-t", "-A"])
    count = r.stdout.decode().strip() if r.returncode == 0 else ""
    if not count:
        print(f"Layer '{layer}' not found."); sys.exit(1)
    print(f"Exporting layer '{layer}' ({count} chunks)...")

    # Build export SQL that outputs a self-contained restore script
    # Uses COPY TO/FROM for maximum speed with binary-safe data
    export_script = f"""
\\pset tuples_only on
\\pset format unaligned

-- Header
\\echo '-- StrataKB Layer Export'
\\echo '-- Layer: {layer}'
\\echo '-- Date: {datetime.now().isoformat()}'
\\echo '-- Chunks: {count}'
\\echo ''
\\echo 'BEGIN;'
\\echo ''
\\echo '-- Clean existing data for this layer'
\\echo 'DELETE FROM strata_vectors.chunks WHERE layer_name = ''{layer}'';'
\\echo 'DELETE FROM strata_core.ingestion_runs WHERE layer_id IN (SELECT id FROM strata_core.knowledge_layers WHERE name = ''{layer}'');'
\\echo 'DELETE FROM strata_core.knowledge_layers WHERE name = ''{layer}'';'
\\echo ''

-- Layer config as COPY block
\\echo 'COPY strata_core.knowledge_layers (id, name, display_name, description, metadata_schema, chunk_config, routing_patterns, status, doc_count, chunk_count, last_ingested, created_at, updated_at) FROM stdin;'
COPY (SELECT id, name, display_name, description, metadata_schema, chunk_config, routing_patterns, status, doc_count, chunk_count, last_ingested, created_at, updated_at FROM strata_core.knowledge_layers WHERE name = '{layer}') TO STDOUT;
\\echo '\\\\.'
\\echo ''

-- Ingestion runs as COPY block
\\echo 'COPY strata_core.ingestion_runs (id, layer_id, source_path, status, docs_processed, docs_failed, chunks_created, error_log, started_at, completed_at) FROM stdin;'
COPY (SELECT ir.id, ir.layer_id, ir.source_path, ir.status, ir.docs_processed, ir.docs_failed, ir.chunks_created, ir.error_log, ir.started_at, ir.completed_at FROM strata_core.ingestion_runs ir JOIN strata_core.knowledge_layers kl ON ir.layer_id = kl.id WHERE kl.name = '{layer}') TO STDOUT;
\\echo '\\\\.'
\\echo ''

-- Chunks with embeddings as COPY block (this is the big one)
\\echo 'COPY strata_vectors.chunks (id, layer_name, content, embedding, metadata, source_file, source_section, doc_type, version, created_at) FROM stdin;'
COPY (SELECT id, layer_name, content, embedding, metadata, source_file, source_section, doc_type, version, created_at FROM strata_vectors.chunks WHERE layer_name = '{layer}') TO STDOUT;
\\echo '\\\\.'
\\echo ''
\\echo 'COMMIT;'
"""

    r = _psql(stdin=export_script.encode("utf-8"))
    if r.returncode != 0:
        print(f"Export failed:\n{r.stderr.decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)

    raw = len(r.stdout)
    with gzip.open(out, "wb", compresslevel=6) as f:
        f.write(r.stdout)
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
