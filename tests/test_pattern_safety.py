"""Unsafe layer patterns never run (stratakb.knowledge.pattern_safety).

Every write path refuses a pattern that does not compile, repeats a
repeated part, is too long, holds a NUL or lone surrogate, or makes Python
warn; these tests cover a pattern that reached the layer row another way:
skipped at question time, at ingest and by routing, and dropped by
import-layer.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib.util
import json
import re
import subprocess
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import asyncpg
import pytest
from structlog.testing import capture_logs

from stratakb.knowledge.layers import LayerValidationError
from stratakb.knowledge.pattern_safety import (
    NESTED_REPEAT_REASON,
    pattern_problem,
    usable_patterns,
)
from stratakb.rag.clause_lookup import find_references
from stratakb.rag.layer_detect import compile_patterns
from tests.helpers import ingest

_spec = importlib.util.spec_from_file_location(
    "backup_restore", Path(__file__).parent.parent / "scripts" / "backup_restore.py"
)
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)


# --- the check ----------------------------------------------------------------


@pytest.mark.parametrize("pattern", [r"^(?P<label>AT-\d{2})", r"(\d{3})+"])
def test_safe_patterns_have_no_problem(pattern):
    assert pattern_problem(pattern) is None
    assert pattern_problem(pattern, line_starts=True) is None


@pytest.mark.parametrize("pattern", ["(a+)+$", "(x+y)*z"])
def test_nested_repeat_is_a_problem(pattern):
    assert pattern_problem(pattern) == NESTED_REPEAT_REASON


def test_other_problems():
    assert pattern_problem("(") == "missing ), unterminated subpattern at position 0"
    assert pattern_problem("x*") is None
    assert pattern_problem("x*", line_starts=True) == "it matches an empty line."
    assert pattern_problem(1) == "it is not a string."


def test_warning_pattern_is_refused_before_compiling():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert usable_patterns([f"[[:alpha:]]{uuid4().hex}"], kind="routing") == []
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]


def test_usable_patterns_keeps_safe_ones_in_order():
    with capture_logs() as logs:
        usable = usable_patterns(
            [r"\bb\b", "(a+)+$", 3, r"\ba\b", "("], kind="routing", layer="x", flags=re.I
        )
    assert [p.pattern for p in usable] == [r"\bb\b", r"\ba\b"]
    assert all(p.flags & re.I for p in usable)
    skipped = [e for e in logs if e["event"] == "unsafe_routing_pattern"]
    assert [e["pattern"] for e in skipped] == ["(a+)+$", "3", "("]
    assert all(e["layer"] == "x" and e["reason"] for e in skipped)


async def test_refusal_says_the_check_is_strict(db_url):
    from stratakb.knowledge.layers import create_layer

    with pytest.raises(LayerValidationError) as exc:
        await create_layer(
            db_url, name=f"test_safety_{uuid4().hex[:10]}", display_name="Safety",
            routing_patterns=["(x+y)*z"],
        )
    message = str(exc.value)
    assert "'(x+y)*z'" in message
    assert "repeats a repeated part" in message
    assert "The check is strict" in message


# --- question time ------------------------------------------------------------


@pytest.fixture
def settings():
    from stratakb.config import Settings, get_settings

    return Settings(
        _env_file=None,
        database_url=get_settings().database_url,
        learning_enabled=False,
        reranking_enabled=False,
        strata_user="test-user",
        strata_role="viewer",
    )


@pytest.fixture
def server(settings):
    """A StrataKB server on the test database, its settings patched in everywhere."""
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
        patch("stratakb.learning.tools.write_audit_log", AsyncMock()),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(side_effect=lambda question, **_: [question]),
        ),
    ):
        yield build_server(settings)


async def test_ask_skips_long_and_warning_patterns(server, db_url):
    from stratakb.knowledge import layers
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_safety_{uuid4().hex[:10]}"
    tag = uuid4().hex
    routing = ["a" * 600 + tag, f"[[:alpha:]]{tag}"]
    structure = [rf"^(?P<label>ZZ{tag}-\d+)" + "q?" * 300, rf"^(?P<label>[[A-Z]{tag}-\d+)"]
    await create_layer(db_url, name=name, display_name="Safety")
    try:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(
                "UPDATE strata_core.knowledge_layers SET routing_patterns = $1::jsonb, "
                "structure_patterns = $2::jsonb WHERE name = $3",
                json.dumps(routing), json.dumps(structure), name,
            )
        finally:
            await conn.close()
        layers._routing_cache.clear()
        with (
            patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[])),
            capture_logs() as logs,
        ):
            await server.call_tool("strata_ask", {"question": f"what is {uuid4().hex}?"})
        mine = [e for e in logs if e.get("layer") == name]
        routing_skips = [e for e in mine if e["event"] == "unsafe_routing_pattern"]
        structure_skips = [e for e in mine if e["event"] == "unsafe_structure_pattern"]
        assert len(routing_skips) == 2
        assert len(structure_skips) == 2
        reasons = " ".join(e["reason"] for e in routing_skips + structure_skips)
        assert "longer than 500 characters" in reasons
        assert "later Python" in reasons
    finally:
        layers._routing_cache.clear()
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)



def test_unsafe_stored_pattern_finds_no_reference():
    # The unsafe part is inside the label group, which is all a question reads
    assert find_references("show AT-44", [r"^(?P<label>(AT-\d+)+)"]) == []


@pytest.mark.parametrize("pattern", [r"[(?P<label>(a+)+)]", r"[(?P<label>(a*)*)]"])
def test_label_group_text_in_a_class_is_checked_as_cut_out(pattern):
    # Safe as stored (a character class), unsafe as the question form
    started = time.monotonic()
    with capture_logs() as logs:
        assert find_references("clause " + "a" * 28 + "b", [pattern]) == []
    assert time.monotonic() - started < 1
    assert any(e["event"] == "unsafe_structure_pattern" for e in logs)


def test_invalid_stored_pattern_is_skipped():
    refs = find_references("show AT-44", ["(unclosed", r"^(?P<label>AT-\d+)"])
    assert [r.labels for r in refs] == [("at-44",)]


def test_backtracking_pattern_does_not_stall_a_question():
    started = time.monotonic()
    assert find_references("clause " + "a" * 28 + "!", ["(a+)+$"]) == []
    assert time.monotonic() - started < 1


def test_non_pattern_entry_is_skipped():
    refs = find_references("show AT-44", [1, re.compile(r"AT-\d+")])
    assert [r.labels for r in refs] == [("at-44",)]


# --- routing ------------------------------------------------------------------


def test_compile_patterns_is_cached_and_warns_once():
    raw = {f"layer_{uuid4().hex[:8]}": ["(a+)+$", r"\bsepa\b", 7]}
    with capture_logs() as logs:
        first = compile_patterns(raw)
        second = compile_patterns({k: list(v) for k, v in raw.items()})
    assert first is second
    assert [p.pattern for p in next(iter(first.values()))] == [r"\bsepa\b"]
    events = [e["event"] for e in logs]
    assert events.count("unsafe_routing_pattern") == 2  # (a+)+$ and 7, each once


# --- ingest -------------------------------------------------------------------


async def test_ingest_skips_unsafe_structure_pattern(db_url, tmp_path):
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_safety_{uuid4().hex[:10]}"
    await create_layer(
        db_url, name=name, display_name="Safety",
        chunk_config={"chunk_size": 200, "chunk_overlap": 40},
    )
    try:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(
                "UPDATE strata_core.knowledge_layers SET structure_patterns = $1::jsonb "
                "WHERE name = $2",
                json.dumps([r"^(?P<label>(AT-\d+)+)", r"^(?P<label>REQ-\d+)"]),
                name,
            )
        finally:
            await conn.close()
        source = tmp_path / "docs"
        source.mkdir()
        (source / "rules.txt").write_text(
            "REQ-1 The payer must be named.\n\nAT-44 The payee's account.\n", encoding="utf-8"
        )
        with capture_logs() as logs:
            result = await ingest(db_url, name, source)
        assert result.docs_failed == 0
        assert any(e["event"] == "unsafe_structure_pattern" for e in logs)

        conn = await asyncpg.connect(db_url)
        try:
            labels = {r["label"] for r in await conn.fetch(
                "SELECT s.label FROM strata_vectors.document_sections s "
                "JOIN strata_vectors.documents d ON d.id = s.document_id "
                "JOIN strata_core.knowledge_layers l ON l.id = d.layer_id WHERE l.name = $1",
                name,
            )}
        finally:
            await conn.close()
        assert "req-1" in labels
        assert "at-44" not in labels
    finally:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


async def test_ingest_skips_long_and_warning_structure_patterns(db_url, tmp_path):
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_safety_{uuid4().hex[:10]}"
    await create_layer(
        db_url, name=name, display_name="Safety",
        chunk_config={"chunk_size": 200, "chunk_overlap": 40},
    )
    try:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(
                "UPDATE strata_core.knowledge_layers SET structure_patterns = $1::jsonb "
                "WHERE name = $2",
                json.dumps([
                    r"^(?P<label>AT-\d+)(?:" + "q?" * 300 + ")",
                    r"^(?P<label>[[A-Z]+-\d+)",
                    r"^(?P<label>REQ-\d+)",
                ]),
                name,
            )
        finally:
            await conn.close()
        source = tmp_path / "docs"
        source.mkdir()
        (source / "rules.txt").write_text(
            "REQ-1 The payer must be named.\n\nAT-44 The payee's account.\n\nBT-7 The amount.\n",
            encoding="utf-8",
        )
        with capture_logs() as logs:
            result = await ingest(db_url, name, source)
        assert result.docs_failed == 0
        assert [e["event"] for e in logs].count("unsafe_structure_pattern") == 2

        conn = await asyncpg.connect(db_url)
        try:
            labels = {r["label"] for r in await conn.fetch(
                "SELECT s.label FROM strata_vectors.document_sections s "
                "JOIN strata_vectors.documents d ON d.id = s.document_id "
                "JOIN strata_core.knowledge_layers l ON l.id = d.layer_id WHERE l.name = $1",
                name,
            )}
        finally:
            await conn.close()
        assert "req-1" in labels
        assert "at-44" not in labels
        assert "bt-7" not in labels
    finally:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


# --- import-layer -------------------------------------------------------------


def _layer_part(routing: list[str], structure: list[str]) -> bytes:
    """One knowledge_layers row in COPY text, holding these patterns."""
    values = {c: "\\N" for c in br.LAYERS.columns}
    values.update(
        name="x",
        routing_patterns=br._copy_text(json.dumps(routing)).decode(),
        structure_patterns=br._copy_text(json.dumps(structure)).decode(),
    )
    return "\t".join(values[c] for c in br.LAYERS.columns).encode() + b"\n"


def _export_file(layer_part: bytes) -> bytes:
    return br.build_layer_export("x", "0", [layer_part] + [b""] * (len(br.EXPORT_TABLES) - 1))


def _import(monkeypatch, tmp_path, data: bytes) -> bytes:
    """Run cmd_import_layer on data; return what psql was fed."""
    calls = []

    def fake(sql="", args=None, stdin=None):
        calls.append(stdin)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(br, "_psql", fake)
    src = tmp_path / "layer.sql.gz"
    with gzip.open(src, "wb") as f:
        f.write(data)
    br.cmd_import_layer(SimpleNamespace(input=str(src), yes=True, root=None))
    assert len(calls) == 1
    return calls[0]


def _stored_patterns(data: bytes) -> tuple[list, list]:
    columns, start, end = br._layer_row_span(data)
    row = dict(zip(columns, map(br._copy_field, data[start:end].split(b"\t")), strict=True))
    return json.loads(row["routing_patterns"]), json.loads(row["structure_patterns"])


def test_import_drops_unsafe_patterns(monkeypatch, tmp_path, capsys):
    data = _export_file(_layer_part(["(a+)+$", r"\bsepa\b"], [r"^(?P<label>(AT-\d+)+)"]))
    fed = _import(monkeypatch, tmp_path, data)
    assert _stored_patterns(fed) == ([r"\bsepa\b"], [])
    out = capsys.readouterr().out
    assert out.count("Dropped the ") == 2
    assert "Dropped the routing pattern '(a+)+$' of layer 'x'" in out
    assert "Dropped the structure pattern '^(?P<label>(AT-\\d+)+)' of layer 'x'" in out
    assert out.index("Dropped the ") < out.index("Importing")


def test_import_drops_long_warning_and_surrogate_patterns(monkeypatch, tmp_path, capsys):
    long = "a" * 600
    data = _export_file(_layer_part([long, r"\bsepa\b", "\ud800x"], [r"^[[:alpha:]]"]))
    fed = _import(monkeypatch, tmp_path, data)
    assert _stored_patterns(fed) == ([r"\bsepa\b"], [])
    out = capsys.readouterr().out
    out.encode("utf-8")
    assert out.count("Dropped the ") == 3
    assert f"Dropped the routing pattern {long[:40]!r}... of layer 'x'" in out
    assert long[:41] not in out
    assert "Dropped the routing pattern '\\ud800x' of layer 'x'" in out
    assert "Dropped the structure pattern '^[[:alpha:]]' of layer 'x'" in out
    assert out.rindex("Dropped the ") < out.index("Importing")


def test_import_keeps_a_safe_file_byte_identical(monkeypatch, tmp_path, capsys):
    data = _export_file(_layer_part([r"\bsepa\b"], [r"^(?P<label>AT-\d{2})"]))
    assert _import(monkeypatch, tmp_path, data) == data
    assert "Dropped" not in capsys.readouterr().out


# The knowledge_layers columns of an export from before format lines
_LEGACY_LAYER_COLUMNS = ("id", "name", "display_name", "description", "metadata_schema",
                         "chunk_config", "routing_patterns", "status", "doc_count",
                         "chunk_count", "last_ingested", "created_at", "updated_at")


def _legacy_row(routing: list[str]) -> bytes:
    """One legacy knowledge_layers row in COPY text, holding these routing patterns."""
    values = {c: "\\N" for c in _LEGACY_LAYER_COLUMNS}
    values.update(name="x", routing_patterns=br._copy_text(json.dumps(routing)).decode())
    return "\t".join(values[c] for c in _LEGACY_LAYER_COLUMNS).encode() + b"\n"


def _legacy_file(row: bytes) -> bytes:
    """An export from before format lines, with this knowledge_layers row."""
    data = (b"-- StrataKB Layer Export\n-- Layer: x\n-- Date: 2026-01-01T00:00:00\n-- Chunks: 0\n\n"
            b"BEGIN;\n\nDELETE FROM strata_vectors.chunks WHERE layer_name = 'x';\n\n"
            b"COPY strata_core.knowledge_layers (" + ", ".join(_LEGACY_LAYER_COLUMNS).encode()
            + b") FROM stdin;\n" + row + b"\\.\n\nCOMMIT;\n")
    assert br.export_format(data) is None
    return data


def _legacy_routing(data: bytes) -> list:
    columns, start, end = br._layer_row_span(data)
    row = dict(zip(columns, map(br._copy_field, data[start:end].split(b"\t")), strict=True))
    return json.loads(row["routing_patterns"])


def test_legacy_import_drops_unsafe_patterns(monkeypatch, tmp_path, capsys):
    fed = _import(monkeypatch, tmp_path, _legacy_file(_legacy_row(["(a+)+$", r"\bsepa\b"])))
    assert _legacy_routing(fed) == [r"\bsepa\b"]
    out = capsys.readouterr().out
    assert out.count("Dropped the routing pattern '(a+)+$' of layer 'x'") == 1
    assert out.count("Dropped the ") == 1
    assert out.index("Dropped the ") < out.index("Importing")
    assert "made by an older version" in out


def test_legacy_import_names_before_the_prompt(monkeypatch, tmp_path, capsys):
    psql = []
    monkeypatch.setattr(br, "_psql", lambda *a, **k: psql.append(k))
    seen = []

    def answer(prompt):
        seen.append(capsys.readouterr().out)
        return "n"

    monkeypatch.setattr("builtins.input", answer)
    src = tmp_path / "layer.sql.gz"
    with gzip.open(src, "wb") as f:
        f.write(_legacy_file(_legacy_row(["(a+)+$", r"\bsepa\b"])))
    br.cmd_import_layer(SimpleNamespace(input=str(src), yes=False, root=None))
    assert len(seen) == 1
    assert "Dropped the routing pattern '(a+)+$' of layer 'x'" in seen[0]
    assert psql == []


def test_legacy_import_keeps_a_safe_file_byte_identical(monkeypatch, tmp_path, capsys):
    data = _legacy_file(_legacy_row([r"\bsepa\b"]))
    assert _import(monkeypatch, tmp_path, data) == data
    assert "Dropped" not in capsys.readouterr().out


@pytest.mark.parametrize("row", [
    b"\\N\tx\t\\N\n",
    "\t".join("\\N" if c != "routing_patterns" else "not json" for c in _LEGACY_LAYER_COLUMNS)
    .encode() + b"\n",
], ids=["too-few-fields", "patterns-not-json"])
def test_legacy_import_refuses_an_unreadable_layer_row(monkeypatch, tmp_path, capsys, row):
    psql = []
    monkeypatch.setattr(br, "_psql", lambda *a, **k: psql.append(k))
    src = tmp_path / "layer.sql.gz"
    with gzip.open(src, "wb") as f:
        f.write(_legacy_file(row))
    with pytest.raises(SystemExit) as exc:
        br.cmd_import_layer(SimpleNamespace(input=str(src), yes=True, root=None))
    assert exc.value.code == 1
    assert psql == []
    err = capsys.readouterr().err
    assert "layer.sql.gz holds a layer row this version cannot check" in err
    assert "Import failed: nothing was changed." in err
