"""Tests for the ingest policy: secret-looking file names. No database needed."""

from __future__ import annotations

import pytest

from stratakb.knowledge.policy import SECRET_PATTERNS, is_secret_name


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
