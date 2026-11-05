"""Report ruff findings only on the lines a branch adds or changes.

The tree has older ruff findings that are fixed separately, so a pull request
that touches a file should answer only for its own lines.

    python scripts/ruff_changed_lines.py origin/main [ruff command ...]

The ruff command defaults to `ruff`. Exits 1 when a changed line has a finding.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def changed_lines(diff: str) -> dict[str, set[int]]:
    """Map each file in a `git diff -U0` to the new-side line numbers it adds or changes."""
    lines: dict[str, set[int]] = {}
    current: str | None = None
    for row in diff.splitlines():
        if row.startswith("+++ "):
            path = row[4:]
            current = path[2:] if path.startswith("b/") else None
            if current is not None:
                lines.setdefault(current, set())
            continue
        match = _HUNK.match(row)
        if match and current is not None:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            lines[current].update(range(start, start + count))
    return lines


def findings_on_changed_lines(
    findings: list[dict], changed: dict[str, set[int]], root: str
) -> list[dict]:
    """Keep the ruff JSON findings whose file and row are in `changed`."""
    prefix = root.rstrip("/\\") + "/"
    kept = []
    for f in findings:
        path = f["filename"].replace("\\", "/")
        if path.startswith(prefix.replace("\\", "/")):
            path = path[len(prefix):]
        if f["location"]["row"] in changed.get(path, set()):
            kept.append({**f, "filename": path})
    return kept


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    base, ruff = argv[0], argv[1:] or ["ruff"]
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "-U0", "--diff-filter=ACMR", f"{base}...HEAD", "--", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    changed = changed_lines(diff)
    if not changed:
        print("No Python files changed")
        return 0
    out = subprocess.run(
        [*ruff, "check", "--no-cache", "--output-format=json", *sorted(changed)],
        capture_output=True, text=True,
    )
    if out.returncode not in (0, 1):
        print(out.stderr, file=sys.stderr)
        return out.returncode
    kept = findings_on_changed_lines(json.loads(out.stdout or "[]"), changed, root)
    for f in kept:
        row, col = f["location"]["row"], f["location"]["column"]
        print(f"::error file={f['filename']},line={row},col={col}::{f['code']} {f['message']}")
    print(f"{len(kept)} ruff finding(s) on changed lines in {len(changed)} file(s)")
    return 1 if kept else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
