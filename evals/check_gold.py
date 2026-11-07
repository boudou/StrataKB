"""Check gold.jsonl against the raw corpus, with no server involved.

1. Each evidence set must match inside its own source section, within one window of text the size
   of a chunk, so a pattern that cannot match a passage is caught before scoring.
2. For each small-tier question, nothing to report beyond (1). For each full-tier question, say
   whether its evidence also appears in the 8 small PEPs: if not, it can serve as a refusal on the
   small layer.
3. Each refusal's absent patterns must match nothing in any PEP.
argv: gold.jsonl, corpus dir, small-layer dir.
"""
import json
import re
import sys
from pathlib import Path

gold = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()]
ALL = sorted(Path(sys.argv[2]).glob("pep-*.rst"))
SMALL = sorted(Path(sys.argv[3]).glob("pep-*.rst"))
WINDOW = 1500


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


texts = {p.name: read(p) for p in ALL}
small = {p.name: read(p) for p in SMALL}
bad = weak = 0
cross = []
for row in gold:
    if row["kind"] == "refusal":
        hits = [(p, n) for p in row["absent"] for n, t in texts.items() if re.search(p, t, re.I)]
        if hits:
            bad += 1
            print("REFUSAL HIT", row["id"], row["question"], sorted({h[1] for h in hits})[:6], len(hits))
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
Path(sys.argv[1]).with_name("cross_refusals.json").write_text(json.dumps(cross), encoding="utf-8")
