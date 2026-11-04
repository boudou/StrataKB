"""Ingest policy: which files are read and ingested.

A file whose name looks like a secret (environment files, keys, certificates,
credentials) is skipped on every ingest route, before it is opened
(is_secret_name). The full verdict (evaluate: secrets, over-long paths, build
files, source code, unsupported formats, size) is for `stratakb add` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from stratakb.knowledge.parsers.registry import SUPPORTED_PARSERS

if TYPE_CHECKING:
    from stratakb.knowledge.selection import FileEntry

SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "*.kdbx",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "*service-account*.json",
    "kubeconfig*",
    "*secret*",
    "*credential*",
)


def is_secret_name(relative_path: str) -> bool:
    """Return True when the file's basename looks like a secret (case-insensitive)."""
    name = relative_path.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    # The template for .env files is meant to be shared.
    if name == ".env.example":
        return False
    matched = {p for p in SECRET_PATTERNS if fnmatchcase(name, p)}
    if not matched:
        return False
    # A document only *about* secrets or credentials (credentials.md) is allowed.
    is_document = name.endswith((".md", ".rst", ".adoc", ".txt"))
    return not (matched <= {"*secret*", "*credential*"} and is_document)


# The VARCHAR(500) of strata_vectors.chunks.source_file (db/01_init.sql)
MAX_RELATIVE_PATH = 500

# Lock files and tool configs: matched on the basename, case-insensitively
BUILD_FILES: tuple[str, ...] = (
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    "tsconfig*.json",
    "requirements*.txt",
    "Pipfile",
    "Pipfile.lock",
    "*.min.js",
)

CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".java", ".kt", ".scala", ".cs", ".fs", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".m", ".swift", ".dart",
    ".rb", ".php", ".lua", ".sh", ".bash", ".ps1", ".bat",
    ".sql", ".proto", ".graphql", ".tf", ".hcl",
    ".vue", ".svelte", ".css", ".scss",
})


@dataclass(frozen=True)
class Verdict:
    """Whether a selected file is ingested and, when it is not, why."""

    ingest: bool
    reason: str | None


def _is_build_file(path: str, name: str) -> bool:
    if any(fnmatchcase(name, p.casefold()) for p in BUILD_FILES):
        return True
    # CI workflow files directly inside .github/workflows/
    folded = path.casefold()
    parent = folded.rsplit("/", 1)[0] if "/" in folded else ""
    return parent == ".github/workflows" and name.endswith((".yml", ".yaml"))


def evaluate(entry: FileEntry, max_file_size_bytes: int) -> Verdict:
    """Return the verdict for one selected file; the first matching check wins.

    Order: secret, path_too_long, build_file, code, unsupported, too_large.
    """
    path = entry.relative_path
    name = path.rsplit("/", 1)[-1].casefold()
    if is_secret_name(path):
        return Verdict(False, "secret")
    if len(path) > MAX_RELATIVE_PATH:
        return Verdict(False, "path_too_long")
    if _is_build_file(path, name):
        return Verdict(False, "build_file")
    ext = PurePosixPath(name).suffix
    if ext in CODE_EXTENSIONS:
        return Verdict(False, "code")
    if ext not in SUPPORTED_PARSERS:
        return Verdict(False, "unsupported")
    if entry.size > max_file_size_bytes:
        return Verdict(False, "too_large")
    return Verdict(True, None)
