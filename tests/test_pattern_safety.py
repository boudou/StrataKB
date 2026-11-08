"""Unsafe layer patterns never run (stratakb.knowledge.pattern_safety).

Every write path refuses a pattern that does not compile or repeats a
repeated part; these tests cover a pattern that reached the layer row
another way: skipped at question time, at ingest and by routing, and
dropped by import-layer.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
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


def test_import_keeps_a_safe_file_byte_identical(monkeypatch, tmp_path, capsys):
    data = _export_file(_layer_part([r"\bsepa\b"], [r"^(?P<label>AT-\d{2})"]))
    assert _import(monkeypatch, tmp_path, data) == data
    assert "Dropped" not in capsys.readouterr().out
