"""Ingest policy: file names that are never read or ingested.

A file whose name looks like a secret (environment files, keys, certificates,
credentials) is skipped on every ingest route, before it is opened.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

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
