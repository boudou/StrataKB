"""Tests for `stratakb receipt <id>`: the text and JSON forms, refusals, exit
codes and the audit entry.

The command runs its own event loop, so these tests are synchronous and set
up their data with asyncio.run against the test database, embeddings mocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import structlog

from stratakb import cli
from tests.helpers import call_tool
from tests.test_receipts import QUOTE, _as, _audit_rows, _cited, _ingest

UNKNOWN_ID = "rcpt_" + "A" * 22
NOT_READABLE = "Receipt not found, or you may not open it."


def _run(coro):
    """Run coro on a fresh loop and close the pools it opened."""
    from stratakb.core.database import close_pool

    async def run():
        try:
            return await coro
        finally:
            await close_pool()

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def cli_env(monkeypatch, tmp_path):
    """No STRATAKB_CONFIG, no ./.env; the suite's logging put back after."""
    monkeypatch.delenv("STRATAKB_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    saved_logging = structlog.get_config()
    yield
    structlog.configure(**saved_logging)


@pytest.fixture
def db_url():
    from stratakb.config import get_settings

    return get_settings().database_url


@pytest.fixture
def layer(db_url):
    """A fresh layer, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_clircpt_{uuid4().hex[:10]}"
    _run(create_layer(db_url, name=name, display_name="Test CLI Receipts"))
    yield name
    with contextlib.suppress(LayerNotFoundError):
        _run(delete_layer(db_url, name=name))


@pytest.fixture
def cited(db_url, layer, tmp_path):
    """A receipt issued by alice for QUOTE in policy.md: (receipt_id, document)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    return _run(_cited(db_url, layer, docs))


def _main(argv: list[str], user: str = "alice", role: str = "viewer") -> int:
    with patch("stratakb.core.auth.get_current_user", return_value=(user, role)):
        return cli.main(argv)


def test_prints_quotes(cited, layer, capsys):
    receipt_id, doc = cited
    assert _main(["receipt", receipt_id]) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert captured.err == ""
    out = captured.out
    assert out.startswith(f"Receipt {receipt_id} (extractive, extractive mode)\n")
    assert "by alice" in out
    assert f"layers: {layer}" in out
    assert "Question: How long are records kept?" in out
    assert "Draft: Records are kept for 30 days." in out
    assert (
        f"[1] policy.md (layer {layer}, v{doc['version']}, clause Retention, "
        in out
    )
    assert "): current" in out
    # The quote marked in its context, the text around it on both sides
    assert f">>> {QUOTE} <<<" in out
    assert out.index("Intro paragraph") < out.index(">>>") < out.index("Closing remarks")


def test_changed_source_flagged(cited, db_url, layer, tmp_path, capsys):
    from tests.test_receipts import V1_TEXT, _ingest

    receipt_id, _ = cited
    docs = tmp_path / "docs"
    (docs / "policy.md").write_text(V1_TEXT.replace("30 days", "13 days"), encoding="utf-8")
    _run(_ingest(db_url, layer, docs))

    assert _main(["receipt", receipt_id]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "): changed" in out
    assert "source changed since this answer (v1 → v2)" in out
    # The cited text, not the new one
    assert f">>> {QUOTE} <<<" in out


OSC_52 = "\x1b]52;c;cm0gLXJmIH4=\x07"


async def _cited_hostile(db_url: str, layer: str, docs, quoted: str = "\x1b[8m hidden") -> str:
    """A receipt quoting a document and a draft that carry terminal escapes.

    The quote is QUOTE followed by `quoted`.
    """
    from stratakb.rag.receipts import issue_receipt
    from tests.test_receipts import _latest_document

    text = f"Intro{OSC_52} text.\n\n{QUOTE}{quoted}\x1b[0m\n\nClosing remarks.\n"
    # Bytes, so the newlines stay as written on every platform
    (docs / "policy.md").write_bytes(text.encode("utf-8"))
    await _ingest(db_url, layer, docs)
    doc = await _latest_document(db_url, layer, "policy.md")
    assert "\x1b" in doc["text"]
    start = doc["text"].index(QUOTE)
    with _as("alice", "viewer"):
        return await issue_receipt(
            db_url,
            kind="extractive",
            mode="extractive",
            question=f"How long?\rFake line{OSC_52}",
            quotes=[
                {
                    "document_id": doc["id"],
                    "char_start": start,
                    "char_end": start + len(QUOTE) + len(quoted),
                    "clause_path": None,
                }
            ],
            draft="Kept\x08\x08\x08\x08Gone\x1b[2K\x9b31m‮",
        )


def test_control_characters_shown_escaped(db_url, layer, tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    receipt_id = _run(_cited_hostile(db_url, layer, docs))
    capsys.readouterr()  # the setup's log lines

    assert _main(["receipt", receipt_id]) == cli.EXIT_OK
    out = capsys.readouterr().out
    for raw in ("\x1b", "\r", "\x07", "\x08", "\x9b", "‮"):
        assert raw not in out
    # Shown, not swallowed: what the reader sees is what was stored
    assert "\\x1b]52;c;cm0gLXJmIH4=\\x07" in out
    assert "Question: How long?\\x0dFake line" in out
    assert "Draft: Kept\\x08\\x08\\x08\\x08Gone\\x1b[2K\\x9b31m\\u202e" in out
    assert f">>> {QUOTE}\\x1b[8m hidden <<<" in out


def test_json_keeps_control_characters_escaped(db_url, layer, tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    receipt_id = _run(_cited_hostile(db_url, layer, docs))
    capsys.readouterr()  # the setup's log lines

    assert _main(["receipt", receipt_id, "--json"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\r" not in out
    assert json.loads(out)["receipt"]["draft"].startswith("Kept\x08")


# A one-byte CSI, DEL and a right-to-left override: json.dumps leaves them raw
C1_QUOTED = " \x9b2J\x7f‮ hidden"


def _assert_json_escapes(out: str) -> None:
    for raw in ("\x9b", "\x7f", "‮"):
        assert raw not in out
    for escape in ("\\u009b", "\\u007f", "\\u202e"):
        assert escape in out


def test_json_escapes_c1_del_and_bidi(db_url, layer, tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    receipt_id = _run(_cited_hostile(db_url, layer, docs, quoted=C1_QUOTED))
    capsys.readouterr()  # the setup's log lines

    assert _main(["receipt", receipt_id, "--json"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    _assert_json_escapes(out)
    # Still valid JSON, decoding to the stored text
    receipt = json.loads(out)["receipt"]
    assert receipt["quotes"][0]["quote"] == QUOTE + C1_QUOTED
    assert receipt["draft"] == "Kept\x08\x08\x08\x08Gone\x1b[2K\x9b31m‮"


def test_text_escapes_c1_del_and_bidi(db_url, layer, tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    receipt_id = _run(_cited_hostile(db_url, layer, docs, quoted=C1_QUOTED))
    capsys.readouterr()  # the setup's log lines

    assert _main(["receipt", receipt_id]) == cli.EXIT_OK
    out = capsys.readouterr().out
    for raw in ("\x9b", "\x7f", "‮"):
        assert raw not in out
    assert f">>> {QUOTE} \\x9b2J\\x7f\\u202e hidden <<<" in out


def test_render_receipt_escapes_every_field():
    """The whole text is escaped once, so no field can slip through unescaped."""
    hostile = "x\x9b‮"
    quote = {
        "file": hostile,
        "layer": hostile,
        "version": hostile,
        "clause_path": hostile,
        "char_start": hostile,
        "char_end": hostile,
        "status": hostile,
        "note": hostile,
        "context_before": f"before{hostile}\r\nline\x0c",
        "quote": hostile,
        "context_after": f"after{hostile}\n",
    }
    receipt = {
        "receipt_id": hostile,
        "kind": hostile,
        "mode": hostile,
        "layers": [hostile],
        "issued_at": hostile,
        "issued_by": hostile,
        "question": hostile,
        "draft": hostile,
        "quotes": [quote],
    }
    text = cli.render_receipt(receipt)
    for raw in ("\x9b", "‮", "\r", "\x0c"):
        assert raw not in text
    assert text.count("x\\x9b\\u202e") == 19
    # Lines break on newlines only; a carriage return or form feed is shown
    assert "    beforex\\x9b\\u202e\\x0d\n    line\\x0c>>> x\\x9b\\u202e <<<afterx" in text
    assert text.endswith("<<<afterx\\x9b\\u202e")


def test_json_error_form_escapes_c1_del_and_bidi(capsys):
    import argparse

    message = "Can't reach the database at db\x9b2J\x7f‮."
    code = cli._receipt_failed(message, argparse.Namespace(json=True), cli.EXIT_DATABASE)
    assert code == cli.EXIT_DATABASE
    captured = capsys.readouterr()
    _assert_json_escapes(captured.out)
    assert json.loads(captured.out) == {"status": "error", "error": message}
    # The stderr line too
    assert captured.err == "Can't reach the database at db\\x9b2J\\x7f\\u202e.\n"


def test_json(cited, server, capsys):
    receipt_id, _ = cited
    assert _main(["receipt", receipt_id, "--json"]) == cli.EXIT_OK
    printed = json.loads(capsys.readouterr().out)

    with _as("alice", "viewer"):
        tool = _run(call_tool(server, "strata_get_receipt", receipt_id=receipt_id))
    assert printed == json.loads(json.dumps(tool))
    assert printed["status"] == "ok"
    assert printed["receipt"]["quotes"][0]["quote"] == QUOTE


@pytest.mark.parametrize("receipt_id", [UNKNOWN_ID, "nope"])
def test_unknown_id_exits_1(receipt_id, capsys):
    assert _main(["receipt", receipt_id]) == cli.EXIT_FAILED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == NOT_READABLE + "\n"


def test_unknown_id_json(capsys):
    assert _main(["receipt", UNKNOWN_ID, "--json"]) == cli.EXIT_FAILED
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "error", "error": NOT_READABLE}
    assert captured.err == NOT_READABLE + "\n"


def test_refused_exits_1(cited, capsys):
    receipt_id, _ = cited
    with patch("stratakb.rag.receipts.readable_layers", AsyncMock(return_value=set())):
        code = _main(["receipt", receipt_id], user="bob")
    assert code == cli.EXIT_FAILED
    captured = capsys.readouterr()
    assert captured.out == ""
    # The same line as an unknown id
    assert captured.err == NOT_READABLE + "\n"
    assert QUOTE not in captured.err


def test_below_viewer_exits_1(cited, capsys):
    receipt_id, _ = cited
    assert _main(["receipt", receipt_id], user="guest", role="none") == cli.EXIT_FAILED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert captured.err.startswith("Permission denied: user 'guest' has role 'none'")


def test_audited(cited, db_url):
    receipt_id, _ = cited
    assert _main(["receipt", receipt_id]) == cli.EXIT_OK
    with patch("stratakb.rag.receipts.readable_layers", AsyncMock(return_value=set())):
        assert _main(["receipt", receipt_id], user="bob") == cli.EXIT_FAILED

    rows = _run(_audit_rows(db_url, receipt_id))
    opens = [
        (r["action"], r["tool_name"], r["user_id"], r["result_summary"])
        for r in rows
        if r["action"] == "receipt_open"
    ]
    assert opens == [
        ("receipt_open", "cli:receipt", "alice", "opened"),
        ("receipt_open", "cli:receipt", "bob", "refused"),
    ]


def test_unreachable_database_exits_3(capsys):
    with patch("stratakb.rag.receipts.ensure_schema", AsyncMock(side_effect=OSError("refused"))):
        assert _main(["receipt", UNKNOWN_ID]) == cli.EXIT_DATABASE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Can't reach the database at" in captured.err
    assert "(OSError)" in captured.err


def test_missing_id_is_a_usage_error(capsys):
    assert cli.main(["receipt"]) == cli.EXIT_REFUSED
    assert "receipt" in capsys.readouterr().err
