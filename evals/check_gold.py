"""Check a gold.jsonl against the raw corpus, with no server and no database involved.

1. Each evidence set must match inside its own source section, within one window of text the size
   of a chunk, so a pattern that cannot match a passage is caught before scoring.
2. For each small-tier question, nothing to report beyond (1). For each full-tier question, say
   whether its evidence also appears in the small layer's PEPs: if not, it can serve as a refusal
   on the small layer.
3. Each refusal's absent patterns must match nothing in any PEP.

The ids of the questions from (2) go to the --out file, and nothing else is written.
"""
import argparse
import json
import re
from pathlib import Path

WINDOW = 1500
FROZEN = Path(__file__).resolve().parent / "gold"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, required=True, help="the gold.jsonl to check")
    parser.add_argument("--corpus", type=Path, required=True, help="folder holding every pep-*.rst")
    parser.add_argument(
        "--small", type=Path, required=True, help="folder holding the small layer's pep-*.rst"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="file to write the cross-layer refusal ids to"
    )
    args = parser.parse_args(argv)
    if not args.gold.is_file():
        parser.error(f"--gold: no such file: {args.gold}")
    for name, folder in (("--corpus", args.corpus), ("--small", args.small)):
        if not any(folder.glob("pep-*.rst")):
            parser.error(f"{name}: no pep-*.rst in {folder}")
    if args.out.resolve().is_relative_to(FROZEN):
        parser.error(f"--out: {FROZEN.name}/ is frozen, write somewhere else")
    if not args.out.parent.is_dir():
        parser.error(f"--out: no such folder: {args.out.parent}")
    return args


def norm(text):
    return re.sub(r"\s+", " ", text)


def read(path):
    return norm(path.read_text(encoding="utf-8", errors="replace"))


def in_window(text, patterns):
    """True when every pattern matches inside one window around a match of the first."""
    for m in re.finditer(patterns[0], text, re.I):
        chunk = text[max(0, m.start() - WINDOW // 2):m.end() + WINDOW // 2]
        if all(re.search(p, chunk, re.I) for p in patterns[1:]):
            return True
    return False


def main(argv=None):
    args = parse_args(argv)
    gold = [json.loads(x) for x in args.gold.read_text(encoding="utf-8").splitlines()]
    texts = {p.name: read(p) for p in sorted(args.corpus.glob("pep-*.rst"))}
    small = {p.name: read(p) for p in sorted(args.small.glob("pep-*.rst"))}
    bad = weak = 0
    cross = []
    for row in gold:
        if row["kind"] == "refusal":
            hits = [
                (p, n) for p in row["absent"] for n, t in texts.items() if re.search(p, t, re.I)
            ]
            if hits:
                bad += 1
                names = sorted({h[1] for h in hits})[:6]
                print("REFUSAL HIT", row["id"], row["question"], names, len(hits))
            continue
        file = row["source"].split(":")[0]
        if not in_window(texts[file], row["evidence"]):
            bad += 1
            print("NO EVIDENCE IN SOURCE", row["id"], file, row["evidence"])
        others = sum(in_window(t, row["evidence"]) for t in texts.values())
        if others >= 8:
            weak += 1
            print("WEAK (matches", others, "files)", row["id"], row["evidence"])
        if row["tier"] == "full" and not any(in_window(t, row["evidence"]) for t in small.values()):
            cross.append(row["id"])
    print("problems", bad, "weak", weak, "cross-layer refusals", len(cross))
    args.out.write_text(json.dumps(cross), encoding="utf-8")


if __name__ == "__main__":
    main()
