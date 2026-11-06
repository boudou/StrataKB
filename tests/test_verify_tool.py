"""Tests for the strata_verify tool: refs resolved from the store, bounds,
access, receipts and the audit entry.

Documents are ingested through the real pipeline against the database with
embeddings mocked.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.helpers import call_tool

FAKE_EMBEDDING = [0.01] * 768
IBAN = "GR1601101250000000012300695"

POLICY = (
    "# Retention\n\nIntro paragraph about the policy.\n\n"
    "Records are kept for thirty (30) days after closure.\n\n"
    f"Refunds go to account {IBAN} within 5 business days.\n\n"
    + "Closing remarks that pad the document. " * 12
)
QUOTE = "Records are kept for thirty (30) days after closure."
REFUND = f"Refunds go to account {IBAN} within 5 business days."


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def layer(db_url):
    """A fresh layer with small chunks (a passage may span 200 characters), deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_verify_{uuid4().hex[:10]}"
    await create_layer(
        db_url,
        name=name,
        display_name="Test Verify",
        chunk_config={"chunk_size": 100, "chunk_overlap": 10},
    )
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


@pytest.fixture
def user():
    return f"verifier_{uuid4().hex[:8]}"


@contextlib.contextmanager
def _as(user: str, role: str = "viewer"):
    with (
        patch("stratakb.core.auth.get_current_user", return_value=(user, role)),
        patch("stratakb.rag.receipts.get_current_user", return_value=(user, role)),
        patch("stratakb.rag.verify_tools.get_current_user", return_value=(user, role)),
        patch("stratakb.rag.receipt_tools.get_current_user", return_value=(user, role)),
    ):
        yield


async def _document(db_url: str, layer: str, tmp_path: Path):
    """Ingest policy.md; return its stored document row."""
    from stratakb.core.database import get_pool
    from stratakb.knowledge.ingestion import ingest_documents

    (tmp_path / "policy.md").write_text(POLICY, encoding="utf-8")
    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda texts, **_: [FAKE_EMBEDDING for _ in texts]
        await ingest_documents(db_url, "fake-key", layer_name=layer, source_path=str(tmp_path))
    pool = await get_pool(db_url)
    return await pool.fetchrow(
        """
        SELECT d.id, d.text FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1 AND d.relative_path = 'policy.md'
        """,
        layer,
    )


def _ref(doc, sentence: str) -> str:
    start = doc["text"].index(sentence)
    return f"d{doc['id']}:{start}-{start + len(sentence)}"


async def _verify(server, draft: str, passages: list[str]) -> dict:
    return await call_tool(server, "strata_verify", draft=draft, passages=passages)


async def _audit_rows(db_url: str, user: str) -> list:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetch(
        "SELECT * FROM strata_core.audit_log WHERE user_id = $1 AND action = 'verify' "
        "ORDER BY created_at, id",
        user,
    )


class TestVerifyTool:
    async def test_verified_has_receipt(self, server, db_url, layer, tmp_path, user):
        from stratakb.rag.receipts import open_receipt

        doc = await _document(db_url, layer, tmp_path)
        passages = [_ref(doc, QUOTE), _ref(doc, REFUND)]
        draft = (
            f'Records are kept for 30 days [S1]: "{QUOTE}" [S1]. '
            f"Refunds reach {IBAN} within five business days [S2]."
        )
        with _as(user):
            result = await _verify(server, draft, passages)
        assert result["status"] == "ok"
        assert result["result"] == "verified"
        assert result["receipt_id"].startswith("rcpt_")
        assert result["quotes_checked"] == 1
        assert "flags" not in result

        receipt = await open_receipt(db_url, result["receipt_id"], user, "viewer")
        assert receipt["kind"] == "verify"
        assert receipt["mode"] == "grounded"
        assert receipt["question"] is None
        assert [q["quote"] for q in receipt["quotes"]] == [QUOTE, REFUND]
        assert receipt["verification"]["result"] == "verified"
        assert [p["id"] for p in receipt["verification"]["passages"]] == ["S1", "S2"]
        # The draft is stored scrubbed; the quote keeps the document's own text
        assert IBAN not in receipt["draft"]
        assert "[IBAN_REDACTED]" in receipt["draft"]
        assert receipt["draft"].startswith("Records are kept for 30 days [S1]")

    async def test_flagged_has_no_receipt(self, server, db_url, layer, tmp_path, user):
        doc = await _document(db_url, layer, tmp_path)
        with _as(user):
            result = await _verify(server, "Records are kept for 13 days [S1].", [_ref(doc, QUOTE)])
        assert result["result"] == "flagged"
        assert "receipt_id" not in result
        assert [(f["kind"], f["text"], f["message"]) for f in result["flags"]] == [
            ("quantity", "13 days", '"13 days" not found in S1')
        ]

    async def test_flag_list_is_bounded(self, server, db_url, layer, tmp_path, user):
        from stratakb.rag.verify import MAX_FLAGS

        doc = await _document(db_url, layer, tmp_path)
        draft = " ".join(f"Q{i}Z" for i in range(1, 2001))
        with _as(user):
            result = await _verify(server, draft, [_ref(doc, QUOTE)])
        assert result["result"] == "flagged"
        assert len(result["flags"]) == MAX_FLAGS
        assert result["more_flags"] == 2000 - MAX_FLAGS
        rows = await _audit_rows(db_url, user)
        assert json.loads(rows[-1]["parameters"])["flags"] == 2000

    async def test_text_is_read_from_the_store(self, server, db_url, layer, tmp_path, user):
        # The ref names the quote's span; the draft cannot supply its own text
        doc = await _document(db_url, layer, tmp_path)
        with _as(user):
            result = await _verify(server, '"kept for 60 days" [S1]', [_ref(doc, QUOTE)])
        assert [f["kind"] for f in result["flags"]] == ["quote"]

    async def test_wide_ref_flagged(self, server, db_url, layer, tmp_path, user):
        doc = await _document(db_url, layer, tmp_path)
        whole = f"d{doc['id']}:0-{len(doc['text'])}"
        beyond = f"d{doc['id']}:0-{len(doc['text']) + 5}"
        with _as(user):
            result = await _verify(
                server, "Kept for 30 days [S2].", [whole, _ref(doc, QUOTE), beyond]
            )
        assert result["result"] == "flagged"
        flags = [(f["kind"], f["text"]) for f in result["flags"]]
        assert flags == [("passage", whole), ("passage", beyond)]
        assert "more than a passage can (200)" in result["flags"][0]["message"]
        assert "lies outside its document" in result["flags"][1]["message"]

    async def test_unreadable_layer_refused(self, server, db_url, layer, tmp_path, user):
        doc = await _document(db_url, layer, tmp_path)
        ref = _ref(doc, QUOTE)
        with (
            _as(user),
            patch("stratakb.rag.verify_tools.readable_layers", AsyncMock(return_value=set())),
        ):
            refused = await _verify(server, "Kept for 30 days [S1].", [ref])
        with _as(user):
            unknown = await _verify(server, "Kept for 30 days [S1].", ["d999999999999:0-10"])
            malformed = await _verify(server, "Kept for 30 days [S1].", ["whole document please"])
        assert refused == {
            "status": "error",
            "error": f"Passage S1 ({ref}) was not found, or you may not read it.",
        }
        # An unknown passage reads the same as an unreadable one
        assert unknown["error"] == (
            "Passage S1 (d999999999999:0-10) was not found, or you may not read it."
        )
        assert malformed["status"] == "error"
        rows = await _audit_rows(db_url, user)
        assert [r["result_summary"] for r in rows] == ["refused", "refused", "refused"]

    async def test_legacy_chunk_ref(self, server, db_url, layer, user):
        # A chunk stored before documents: its ref is c<chunk id>, its content
        # may start with the old [layer > file] prefix
        from stratakb.core.database import get_pool

        pool = await get_pool(db_url)
        chunk_id = await pool.fetchval(
            """
            INSERT INTO strata_vectors.chunks (layer_name, content, source_file)
            VALUES ($1, $2, 'old.md') RETURNING id::text
            """,
            layer,
            f"[{layer} > old]\nOld records are kept for 45 days.",
        )
        with _as(user):
            ok = await _verify(server, '"Old records are kept for 45 days." [S1]', [f"c{chunk_id}"])
            bad = await _verify(server, f"[{layer} > old] 46 days [S1]", [f"c{chunk_id}"])
        assert ok["result"] == "verified"
        assert ok["receipt_id"].startswith("rcpt_")
        assert {f["text"] for f in bad["flags"]} == {"46 days"}

    async def test_audit_row_without_draft(self, server, db_url, layer, tmp_path, user):
        doc = await _document(db_url, layer, tmp_path)
        with _as(user):
            verified = await _verify(server, f'"{QUOTE}" [S1]', [_ref(doc, QUOTE)])
            await _verify(server, f"Refunds reach {IBAN} in 9 days [S1].", [_ref(doc, QUOTE)])

        rows = await _audit_rows(db_url, user)
        assert [(r["tool_name"], r["result_summary"]) for r in rows] == [
            ("strata_verify", "verified"),
            ("strata_verify", "flagged"),
        ]
        first, second = (json.loads(r["parameters"]) for r in rows)
        assert first == {
            "passages": 1, "quotes": 1, "facts": 0, "flags": 0,
            "receipt_id": verified["receipt_id"],
        }
        assert second == {"passages": 1, "quotes": 0, "facts": 2, "flags": 2}
        for r in rows:
            assert QUOTE not in r["parameters"]
            assert IBAN not in r["parameters"]

    async def test_input_limits(self, server, user):
        with _as(user):
            assert (await _verify(server, "x", []))["status"] == "error"
            assert (await _verify(server, "x", ["c1"] * 51))["status"] == "error"

    async def test_tool_requires_viewer(self, server, user):
        from stratakb.core.auth import PermissionDeniedError

        with _as(user, role="none"), pytest.raises(PermissionDeniedError):
            await _verify(server, "x", ["d1:0-1"])
