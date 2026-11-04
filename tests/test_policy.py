"""Tests for the ingest policy: secret-looking file names and verdicts. No database needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from stratakb.knowledge.policy import (
    BUILD_FILES,
    CODE_EXTENSIONS,
    SECRET_PATTERNS,
    Verdict,
    evaluate,
    is_secret_name,
)
from stratakb.knowledge.selection import FileEntry


@pytest.mark.parametrize("name", [
    ".env.example",
    ".ENV.EXAMPLE",
    "credentials.md",
    "Credentials.MD",
    "docs/credentials.md",
    "my-secret-notes.txt",
    "secret-rotation.rst",
    "credential-flow.adoc",
    "readme.md",
    "notes.txt",
    "openapi.yaml",
])
def test_allowed(name):
    assert is_secret_name(name) is False


@pytest.mark.parametrize("name", [
    ".env",
    ".env.local",
    ".env.json",
    "client_secret.json",
    "prod-credentials.yaml",
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "server.PEM",
    "tls.key",
    "store.p12",
    "store.pfx",
    "vault.kdbx",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "my-service-account.json",
    "kubeconfig",
    "kubeconfig.yaml",
    "secrets.yaml",
    ".env.md",
    "secret.key",
    "kubeconfig-secret.txt",
    "sub/dir/.env",
    "sub\\dir\\id_rsa",
    "Client_Secret.JSON",
])
def test_refused(name):
    assert is_secret_name(name) is True


def test_patterns_are_the_documented_set():
    assert len(SECRET_PATTERNS) == 16
    assert SECRET_PATTERNS[0] == ".env"
    assert SECRET_PATTERNS[-1] == "*credential*"


MB = 1024 * 1024


def _entry(relative_path: str, size: int = 100) -> FileEntry:
    return FileEntry(relative_path=relative_path, size=size, path=Path(relative_path))


@pytest.mark.parametrize("path, size, reason", [
    ("config/.env", 10, "secret"),
    ("a/" * 250 + "x.md", 10, "path_too_long"),
    ("package-lock.json", 10, "build_file"),
    ("sub/tsconfig.base.json", 10, "build_file"),
    ("requirements-dev.txt", 10, "build_file"),
    ("static/app.min.js", 10, "build_file"),
    (".github/workflows/ci.yml", 10, "build_file"),
    ("src/main.py", 10, "code"),
    ("styles/site.SCSS", 10, "code"),
    ("image.png", 10, "unsupported"),
    ("Makefile", 10, "unsupported"),
    ("big.md", 51 * MB, "too_large"),
])
def test_evaluate_reasons(path, size, reason):
    assert evaluate(_entry(path, size), 50 * MB) == Verdict(False, reason)


def test_evaluate_ingests_a_small_document():
    assert evaluate(_entry("docs/guide.md"), 50 * MB) == Verdict(True, None)
    # At the limit is still allowed
    assert evaluate(_entry("guide.md", 50 * MB), 50 * MB) == Verdict(True, None)


def test_evaluate_order_secret_first():
    assert evaluate(_entry("secrets.yaml", 60 * MB), 50 * MB).reason == "secret"


def test_evaluate_workflow_rule_is_a_path_rule():
    # Only YAML files directly inside .github/workflows/
    assert evaluate(_entry("docs/workflows/ci.yml"), 50 * MB).ingest is True
    assert evaluate(_entry(".github/workflows/sub/ci.yml"), 50 * MB).ingest is True
    assert evaluate(_entry(".GitHub/Workflows/CI.YAML"), 50 * MB).reason == "build_file"


@pytest.mark.parametrize("path", [".env.example", "credentials.md", "my-secret-notes.txt"])
def test_evaluate_allowed_not_secret(path):
    assert evaluate(_entry(path), 50 * MB).reason != "secret"


@pytest.mark.parametrize("path", ["client_secret.json", "prod-credentials.yaml"])
def test_evaluate_refused_secret(path):
    assert evaluate(_entry(path), 50 * MB) == Verdict(False, "secret")


@pytest.mark.parametrize("path", ["UV.LOCK", "cargo.lock", "PIPFILE", "Pipfile.lock"])
def test_build_file_casefold(path):
    assert evaluate(_entry(path), 50 * MB).reason == "build_file"


def test_lists_are_the_documented_set():
    assert len(BUILD_FILES) == 12
    assert len(CODE_EXTENSIONS) == 39
