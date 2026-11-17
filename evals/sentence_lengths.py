"""Report how many sentences are too long to quote, and the named clauses holding them.

An extractive answer never quotes a sentence longer than MAX_SENTENCE_CHARS. For each layer of
the database in DATABASE_URL (--database), a folder of documents (--folder) or the source folders
of a preset (--preset), this prints the share of sentences over that limit and the share of
sections (named clauses) holding one, as JSON on stdout. Sentences and sections are found as an
ingest finds them. Nothing is written but the --out file, and the database is only read.

--folder reads every file the preset dry run would, which may be more than `stratakb add` keeps;
--database is the exact view of what was ingested.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

FROZEN = Path(__file__).resolve().parent / "gold"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--database", action="store_true", help="report the layers of the database in DATABASE_URL"
    )
    parser.add_argument(
        "--layer", action="append", help="with --database: one layer to report (repeatable)"
    )
    parser.add_argument("--folder", type=Path, help="a folder of documents to report")
    parser.add_argument(
        "--preset",
        type=Path,
        help="a preset YAML: its source folders, or with --folder its structure_patterns",
    )
    parser.add_argument("--out", type=Path, help="also write the report to this file")
    args = parser.parse_args(argv)
    if not (args.database or args.folder or args.preset):
        parser.error("no input given: name --database, --folder or --preset")
    if args.database and (args.folder or args.preset):
        parser.error("--database: give it alone, not with --folder or --preset")
    if args.layer and not args.database:
        parser.error("--layer: only with --database")
    if args.folder is not None and not args.folder.exists():
        parser.error(f"--folder: no such folder: {args.folder}")
    if args.preset is not None and not args.preset.is_file():
        parser.error(f"--preset: no such file: {args.preset}")
    if args.out is not None:
        if args.out.resolve().is_relative_to(FROZEN):
            parser.error(f"--out: {FROZEN.name}/ is frozen, write somewhere else")
        if not args.out.parent.is_dir():
            parser.error(f"--out: no such folder: {args.out.parent}")
    return args


def pct(n, d):
    return round(100 * n / d, 2) if d else None


class Tally:
    """Sentence and section counts over the documents of one source."""

    def __init__(self, name):
        self.name = name
        self.documents = self.sentences = self.long_sentences = 0
        self.clauses = self.clauses_with_long = 0
        self.longest = None

    def add(self, file, text, sections):
        """Count one document's sentences, and the sections (start, end) holding long ones."""
        from stratakb.rag.extractive import MAX_SENTENCE_CHARS, split_sentences

        spans = split_sentences(text)
        long_spans = [(s, e) for s, e in spans if e - s > MAX_SENTENCE_CHARS]
        self.documents += 1
        self.sentences += len(spans)
        self.long_sentences += len(long_spans)
        self.clauses += len(sections)
        # A section's span holds its subsections, so the enclosing sections count too
        self.clauses_with_long += sum(
            any(s < end and e > start for s, e in long_spans) for start, end in sections
        )
        for s, e in spans:
            if self.longest is None or e - s > self.longest["chars"]:
                self.longest = {"file": file, "chars": e - s}

    def report(self, **extra):
        return {
            "name": self.name,
            "documents": self.documents,
            **extra,
            "sentences": self.sentences,
            "long_sentences": self.long_sentences,
            "long_pct": pct(self.long_sentences, self.sentences),
            "clauses": self.clauses,
            "clauses_with_long": self.clauses_with_long,
            "clauses_with_long_pct": pct(self.clauses_with_long, self.clauses),
            "longest": self.longest,
        }


def read_folder(name, source, structure_patterns=()):
    """The report of one folder (or file), read as the preset dry run reads it."""
    from stratakb.knowledge.ingestion import _discover_files, _map_document, _parse_file
    from stratakb.knowledge.policy import is_secret_name

    tally = Tally(name)
    failed = 0
    for path in _discover_files(source):
        relative = path.name if source.is_file() else path.relative_to(source).as_posix()
        # Secret-looking names are never opened, as in a real ingest
        if is_secret_name(relative):
            continue
        parsed = _parse_file(path)
        if parsed.error:
            failed += 1
            print(f"{name}: {relative}: {parsed.error}", file=sys.stderr)
            continue
        mapped = _map_document(parsed, structure_patterns)
        tally.add(relative, mapped.text, [(s.char_start, s.char_end) for s in mapped.sections])
    return tally.report(failed=failed)


def read_preset(preset_path, folder=None):
    """One report per source group of the preset, or one for folder with its patterns."""
    from stratakb.knowledge.layers import LayerValidationError
    from stratakb.knowledge.presets import load_preset

    try:
        preset = load_preset(preset_path)
    except LayerValidationError as e:
        sys.exit(f"Stopped: {e}")
    if folder is not None:
        return [read_folder(str(folder), folder, preset.structure_patterns)]
    reports = []
    for group in preset.sources:
        source = preset.root / group.path
        if not source.exists():
            print(f"{group.name}: source folder not found: {source}", file=sys.stderr)
            continue
        reports.append(read_folder(group.name, source, preset.structure_patterns))
    return reports


async def read_database(database_url, layers=None):
    """One report per layer, from the document versions its chunks cite; reads only."""
    from stratakb.core.database import close_pool, get_pool

    try:
        pool = await get_pool(database_url)
        known = {
            r["name"]: r["id"]
            for r in await pool.fetch("SELECT id, name FROM strata_core.knowledge_layers")
        }
        missing = [name for name in layers or [] if name not in known]
        if missing:
            sys.exit(f"Stopped: no such layer: {', '.join(missing)}")
        reports = []
        for name in layers or sorted(known):
            # Versions kept only for receipts are cited by no chunk: left out
            documents = await pool.fetch(
                """
                SELECT d.id, d.relative_path, d.text FROM strata_vectors.documents d
                WHERE d.layer_id = $1 AND EXISTS (
                    SELECT 1 FROM strata_vectors.chunks c WHERE c.document_id = d.id)
                ORDER BY d.relative_path, d.version
                """,
                known[name],
            )
            sections = {}
            for r in await pool.fetch(
                """
                SELECT s.document_id, s.char_start, s.char_end
                FROM strata_vectors.document_sections s
                JOIN strata_vectors.documents d ON d.id = s.document_id
                WHERE d.layer_id = $1
                """,
                known[name],
            ):
                sections.setdefault(r["document_id"], []).append((r["char_start"], r["char_end"]))
            legacy = await pool.fetchval(
                "SELECT count(*) FROM strata_vectors.chunks"
                " WHERE layer_name = $1 AND document_id IS NULL",
                name,
            )
            tally = Tally(name)
            for d in documents:
                tally.add(d["relative_path"], d["text"], sections.get(d["id"], []))
            reports.append(tally.report(chunks_without_document=legacy))
        return reports
    finally:
        await close_pool()


def main(argv=None):
    args = parse_args(argv)

    from stratakb.config import get_settings
    from stratakb.core.logging import configure_logging
    from stratakb.rag.extractive import MAX_SENTENCE_CHARS

    configure_logging("ERROR")
    if args.database:
        try:
            sources = asyncio.run(read_database(get_settings().database_url, args.layer))
        except Exception as e:
            sys.exit(f"Stopped: cannot read the database: {type(e).__name__}: {e}")
    elif args.preset is not None:
        sources = read_preset(args.preset, args.folder)
    else:
        sources = [read_folder(str(args.folder), args.folder)]
    report = json.dumps({"max_sentence_chars": MAX_SENTENCE_CHARS, "sources": sources}, indent=1)
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
