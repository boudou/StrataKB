"""Check a pull request against the repository rules in CONTRIBUTING.md.

Checks the commit messages since the base, the pull request title and body
(a squash merge builds its message from them) and the lines the pull request
adds:

- no attribution trailers from bots or tools: a ``Co-authored-by:`` whose
  address is a bot or a ``noreply@`` address of a service, or a
  "Generated with/by" line;
- no absolute paths into a home folder (``X:\\Users\\<name>``,
  ``/home/<name>``, ``/Users/<name>``);
- none of the extra terms in the STRATA_PRIVATE_TERMS environment variable
  (one per line or comma-separated). A hit prints only where it is, never the
  term or the line, so the list stays private in a public log.

Usage: python scripts/check_repo_rules.py <base-ref>
PR_TITLE and PR_BODY are read from the environment when set. Exits 1 on any
finding and prints GitHub annotations.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

CO_AUTHOR = re.compile(r"^\s*co-authored-by:\s*(?P<name>[^<]*)<(?P<email>[^>]*)>", re.I | re.M)
GENERATED = re.compile(r"^[^\w\n]*generated (?:with|by)\b", re.I | re.M)
# A home folder is what a pasted local path gives away. Other folder names
# (a projects folder, say) belong in STRATA_PRIVATE_TERMS.
LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]+users[\\/]+(?!runneradmin\b)[a-z0-9._-]+"
    r"|(?<![\w.~])/(?:home|Users)/(?!runner/)[A-Za-z0-9._-]+/",
    re.I,
)
SKIP_PATH_CHECK = {"scripts/check_repo_rules.py"}
# Service bots whose co-author trailer GitHub adds to their own pull requests
ALLOWED_BOTS = {"dependabot[bot]", "github-actions[bot]"}


@dataclass
class Finding:
    where: str
    message: str
    file: str | None = None
    line: int | None = None

    def annotation(self) -> str:
        location = f" file={self.file},line={self.line}" if self.file else ""
        return f"::error{location}::{self.where}: {self.message}"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *args],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout


def private_terms(raw: str | None) -> list[re.Pattern]:
    terms = [t.strip() for t in re.split(r"[\n,]", raw or "") if t.strip()]
    return [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.I) for t in terms]


def check_message(where: str, text: str, terms: list[re.Pattern]) -> list[Finding]:
    """Attribution trailers and private terms in a commit message, title or body."""
    findings = []
    for m in CO_AUTHOR.finditer(text):
        email = m["email"].strip().lower()
        if m["name"].strip().lower() in ALLOWED_BOTS:
            continue
        if "[bot]" in email or email.startswith("noreply@") or re.search(r"\bAI\b", m["name"]):
            findings.append(Finding(where, f"attribution trailer: {m.group(0).strip()}"))
    for m in GENERATED.finditer(text):
        line = text[m.start():].splitlines()[0].strip()
        findings.append(Finding(where, f"generated-by line: {line}"))
    if any(t.search(text) for t in terms):
        findings.append(Finding(where, "contains a private term"))
    return findings


def added_lines(diff: str):
    """Yield (file, line number, text) for every line a unified diff adds."""
    file, number = None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            file = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            number = int(m.group(1)) if m else 0
        elif line.startswith("+") and file:
            yield file, number, line[1:]
            number += 1
        elif not line.startswith("-"):
            number += 1


def check_diff(diff: str, terms: list[re.Pattern]) -> list[Finding]:
    """Local paths and private terms in added lines and in added file names."""
    findings = []
    private_names = set()
    for file, number, text in added_lines(diff):
        if file not in private_names and any(t.search(file) for t in terms):
            # The name itself is private: never print it, not even as a location
            private_names.add(file)
            findings.append(Finding("diff", "an added file name contains a private term"))
        location = (None, None) if file in private_names else (file, number)
        if file not in SKIP_PATH_CHECK and (m := LOCAL_PATH.search(text)):
            findings.append(Finding("diff", f"absolute local path: {m.group(0)}", *location))
        if any(t.search(text) for t in terms):
            findings.append(Finding("diff", "contains a private term", *location))
    return findings


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    base = argv[1]
    terms = private_terms(os.environ.get("STRATA_PRIVATE_TERMS"))
    if not terms:
        print("::notice::STRATA_PRIVATE_TERMS is not set; private terms are not checked")

    findings: list[Finding] = []
    for sha in _git("rev-list", f"{base}..HEAD").split():
        message = _git("log", "-1", "--format=%B", sha)
        findings += check_message(f"commit {sha[:10]}", message, terms)
    for name in ("PR_TITLE", "PR_BODY"):
        if text := os.environ.get(name):
            findings += check_message(name.lower().replace("_", " "), text, terms)
    findings += check_diff(_git("diff", "--unified=0", "--no-color", f"{base}...HEAD"), terms)

    for finding in findings:
        print(finding.annotation())
    if findings:
        print(f"{len(findings)} finding(s); see the repository rules in CONTRIBUTING.md")
        return 1
    print("Repository rules: no findings")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
