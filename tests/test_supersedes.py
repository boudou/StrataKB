"""Tests for the supersedes declaration stored with each document version.

The rules for reading and resolving the value are pure; the others ingest
through the real pipeline against the database with embeddings mocked.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from stratakb.knowledge.document_dates import frontmatter_date
from stratakb.knowledge.parsers.registry import FORMATS
from stratakb.knowledge.structure import frontmatter_source
from stratakb.knowledge.supersedes import (
    declared_supersedes,
    frontmatter_supersedes,
    resolve_supersedes,
)
from tests.helpers import file_outcomes, ingest, ingest_with_texts, write_file


@pytest.fixture
async def layer(db_url):
    """A fresh layer, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_supersedes_{uuid4().hex[:10]}"
    await create_layer(db_url, name=name, display_name="Test Supersedes")
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


async def _declared(db_url: str, layer: str, relative_path: str) -> dict[int, str | None]:
    """A file's stored versions: {version: supersedes}."""
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        """
        SELECT d.version, d.supersedes FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1 AND d.relative_path = $2
        """,
        layer,
        relative_path,
    )
    return {r["version"]: r["supersedes"] for r in rows}


def _with_key(value: str, body: str = "# Doc\n\nSome text about the topic.\n") -> str:
    return f"---\nsupersedes: {value}\n---\n{body}"


async def test_two_runbooks_store_the_declaration(db_url, layer, tmp_path):
    write_file(tmp_path / "failover-2024.md", "# Failover\n\nSwitch by hand.\n")
    write_file(
        tmp_path / "failover-2026.md",
        _with_key("failover-2024.md", "# Failover\n\nSwitch with the script.\n"),
    )

    await ingest(db_url, layer, tmp_path)

    assert await _declared(db_url, layer, "failover-2026.md") == {1: "failover-2024.md"}
    assert await _declared(db_url, layer, "failover-2024.md") == {1: None}


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("supersedes: old.md", "old.md"),
        ("SuperSedes: old.md", "old.md"),
        ("supersedes : old.md", "old.md"),
        ("supersedes: old file.md", "old file.md"),
        ('supersedes: "old file.md"', "old file.md"),
        ("supersedes: 'old.md'", "old.md"),
        ("supersedes: 'old.md' # the last one", "old.md"),
        ('supersedes: "[draft].md"', "[draft].md"),
        ("supersedes: old.md # the last one", "old.md"),
        ("supersedes: old.md\t# the last one", "old.md"),
        ("supersedes: old.md#part", "old.md#part"),
        ("title: x\nsupersedes: old.md\nauthor: y", "old.md"),
        ("supersedes: first.md\nsupersedes: second.md", "first.md"),
        ("supersedes: [a.md]\nsupersedes: b.md", None),
        ('supersedes: "old.md', None),
        ("supersedes: 'old.md", None),
        ('supersedes: ""', None),
        ("supersedes: '  '", None),
        ("supersedes:", None),
        ("supersedes:old.md", None),
        ("supersedes: # nothing", None),
        ("supersedes: [a.md, b.md]", None),
        ("supersedes: {file: a.md}", None),
        ("supersedes: |\n  a.md", None),
        ("supersedes: >\n  a.md", None),
        ("supersedes: &anchor a.md", None),
        ("supersedes: *anchor", None),
        ("supersedes: !tag a.md", None),
        ("  supersedes: old.md\ntitle: x", None),
        ('"supersedes": old.md', None),
        ("supersedes_old: old.md", None),
        ("old_supersedes: old.md", None),
        ("\u017fupersedes: old.md", None),
    ],
)
def test_frontmatter_supersedes_rules(block, expected):
    assert frontmatter_supersedes(f"---\n{block}\n---\n# Title\n") == expected


@pytest.mark.parametrize(
    "text",
    [
        "# Title\n\nsupersedes: old.md\n",
        "---\ntitle: x\n---\n# Title\n\nsupersedes: old.md\n",
        # A '---' block that is not front matter: a thematic break, then text
        "---\nsupersedes: old.md\nNot a key line\n---\n",
        "---\n\nsupersedes: old.md\n---\n",
        "",
    ],
)
def test_frontmatter_supersedes_looks_only_in_front_matter(text):
    assert frontmatter_supersedes(text) is None


@pytest.mark.parametrize(
    ("value", "relative_path", "expected"),
    [
        ("old.md", "new.md", "old.md"),
        ("../old.md", "docs/a/new.md", "docs/old.md"),
        ("./old.md", "docs/a/new.md", "docs/a/old.md"),
        ("b/old.md", "docs/a/new.md", "docs/a/b/old.md"),
        ("dir\\old.md", "docs/new.md", "docs/dir/old.md"),
        ("../../old.md", "docs/a/new.md", "old.md"),
        ("../old.md", "new.md", None),
        ("../../old.md", "docs/new.md", None),
        ("..", "new.md", None),
        (".", "new.md", None),
        ("/old.md", "new.md", None),
        ("\\old.md", "new.md", None),
        ("C:/old.md", "new.md", None),
        ("C:\\old.md", "new.md", None),
        ("https://example.com/old.md", "new.md", None),
        ("new.md", "new.md", None),
        ("../a/new.md", "a/new.md", None),
        ("old\x01.md", "new.md", None),
        ("old\x00.md", "new.md", None),
        ("x" * 501, "new.md", None),
        ("x" * 500, "new.md", "x" * 500),
    ],
)
def test_resolve_supersedes_rules(value, relative_path, expected):
    assert resolve_supersedes(value, relative_path) == expected


async def test_resolved_paths_through_an_ingest(db_url, layer, tmp_path):
    write_file(tmp_path / "docs" / "a" / "up.md", _with_key("../old.md"))
    write_file(tmp_path / "docs" / "a" / "here.md", _with_key("./old.md"))

    await ingest(db_url, layer, tmp_path)

    assert await _declared(db_url, layer, "docs/a/up.md") == {1: "docs/old.md"}
    assert await _declared(db_url, layer, "docs/a/here.md") == {1: "docs/a/old.md"}


async def test_unusable_values_do_not_fail_the_ingest(db_url, layer, tmp_path):
    values = {
        "outside.md": "../outside.md",
        "absolute.md": "/etc/old.md",
        "list.md": "[a.md, b.md]",
        "empty.md": '""',
        "own.md": "own.md",
        "nul.md": "old\x00.md",
        "long.md": "x" * 501,
    }
    for name, value in values.items():
        write_file(tmp_path / name, _with_key(value))

    result = await ingest(db_url, layer, tmp_path)

    assert result.status == "completed" and result.errors == []
    assert file_outcomes(result) == {name: ("added", None) for name in values}
    for name in values:
        assert await _declared(db_url, layer, name) == {1: None}


async def test_missing_target_is_stored_and_does_not_fail(db_url, layer, tmp_path):
    write_file(tmp_path / "new.md", _with_key("gone.md"))

    result = await ingest(db_url, layer, tmp_path)

    assert result.status == "completed" and result.errors == []
    assert file_outcomes(result) == {"new.md": ("added", None)}
    assert await _declared(db_url, layer, "new.md") == {1: "gone.md"}


async def test_outside_path_is_never_read(db_url, layer, tmp_path):
    write_file(tmp_path / "secret.md", "# Secret\n\nNot part of the layer.\n")
    root = tmp_path / "docs"
    write_file(root / "new.md", _with_key("../secret.md"))
    read: list[str] = []
    real_read_bytes = Path.read_bytes

    def spy(self):
        read.append(self.name)
        return real_read_bytes(self)

    with patch.object(Path, "read_bytes", spy):
        await ingest(db_url, layer, root)

    assert read == ["new.md"]
    assert await _declared(db_url, layer, "new.md") == {1: None}


async def test_front_matter_edit_reuses_version_and_updates(db_url, layer, tmp_path):
    path = write_file(tmp_path / "new.md", _with_key("a.md"))
    await ingest(db_url, layer, tmp_path)

    write_file(path, _with_key("b.md"))
    _, embedded = await ingest_with_texts(db_url, layer, tmp_path)

    assert embedded  # The bytes changed, so the file went through the store path
    assert await _declared(db_url, layer, "new.md") == {1: "b.md"}


async def test_removing_the_key_clears_it(db_url, layer, tmp_path):
    body = "# Doc\n\nSome text about the topic.\n"
    path = write_file(tmp_path / "new.md", _with_key("a.md", body))
    await ingest(db_url, layer, tmp_path)
    assert await _declared(db_url, layer, "new.md") == {1: "a.md"}

    write_file(path, f"---\ntitle: x\n---\n{body}")
    await ingest(db_url, layer, tmp_path)

    assert await _declared(db_url, layer, "new.md") == {1: None}


async def test_changed_text_gets_new_version_with_its_own_value(db_url, layer, tmp_path):
    path = write_file(tmp_path / "new.md", _with_key("a.md", "# Doc\n\nFirst text.\n"))
    await ingest(db_url, layer, tmp_path)
    # No prune: the old version stays, as it does while a receipt cites it
    with patch("stratakb.knowledge.ingestion.prune_documents"):
        write_file(path, _with_key("b.md", "# Doc\n\nSecond text.\n"))
        await ingest(db_url, layer, tmp_path)

    assert await _declared(db_url, layer, "new.md") == {1: "a.md", 2: "b.md"}


@pytest.mark.parametrize(
    "data",
    [
        b"---\nsupersedes: old.md\n---\n# Title\n",
        b"---\r\nsupersedes: old.md\r\n---\r\n# Title\r\n",
        b"---\rsupersedes: old.md\r---\r# Title\r",
        b"\xef\xbb\xbf---\nsupersedes: old.md\n---\n# Title\n",
        b"\n\n---\nsupersedes: old.md\n---\n# Title\n",
        b"---\nsupersedes: old.md\n---\n# Caf\xe9, not UTF-8\n",
    ],
)
def test_crlf_and_byte_order_mark_front_matter(data):
    assert declared_supersedes(FORMATS[".md"], data, "new.md") == "old.md"


async def test_crlf_file_through_an_ingest(db_url, layer, tmp_path):
    (tmp_path / "new.md").write_bytes(
        b"\xef\xbb\xbf---\r\nsupersedes: old.md\r\n---\r\n# Doc\r\n\r\nSome text.\r\n"
    )

    await ingest(db_url, layer, tmp_path)

    assert await _declared(db_url, layer, "new.md") == {1: "old.md"}


class _Undecodable(bytes):
    """Bytes that fail the test when decoded."""

    def decode(self, *args, **kwargs):
        raise AssertionError("decoded")


@pytest.mark.parametrize("suffix", [".docx", ".pdf"])
def test_binary_format_is_not_decoded(suffix):
    data = _Undecodable(b"---\nsupersedes: x.md\n---\n")
    assert declared_supersedes(FORMATS[suffix], data, f"new{suffix}") is None


def test_unknown_format_gives_nothing():
    assert declared_supersedes(None, b"---\nsupersedes: x.md\n---\n", "new.xyz") is None


@pytest.mark.parametrize("lead", [" ", " ", "\x1c"])
def test_block_after_unicode_whitespace_counts_as_it_does_for_the_date(lead):
    # The parser strips this block as front matter, so the date and the
    # declaration must both read it
    data = f"{lead}---\nsupersedes: old.md\nupdated: 2026-01-02\n---\nBody.\n".encode()
    assert frontmatter_date(frontmatter_source(data)) is not None
    assert declared_supersedes(FORMATS[".md"], data, "new.md") == "old.md"


async def test_single_file_ingest_resolves_against_its_own_folder(db_url, layer, tmp_path):
    from stratakb.core.database import get_pool

    path = write_file(tmp_path / "docs" / "new.md", _with_key("../old.md"))

    # With the folder as root the value stays inside it; alone, the file is the
    # only thing below its root, and the value leaves it
    await ingest(db_url, layer, tmp_path)
    await ingest(db_url, layer, path)

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        """
        SELECT d.relative_path, d.supersedes FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1
        """,
        layer,
    )
    assert {r["relative_path"]: r["supersedes"] for r in rows} == {
        "docs/new.md": "old.md",
        "new.md": None,
    }
