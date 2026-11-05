"""Tests for receipts: issuing, opening through the tool and the resource,
changed and deleted sources, access, audit, PII scrubbing and pruning.

All tests ingest through the real pipeline against the database with
embeddings mocked.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

FAKE_EMBEDDING = [0.01] * 768

V1_TEXT = (
    "# Retention\n\nIntro paragraph about the policy.\n\n"
    "Records are kept for 30 days after closure.\n\nClosing remarks.\n"
)
QUOTE = "Records are kept for 30 days after closure."


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def layer(db_url):
    """A fresh layer, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_rcpt_{uuid4().hex[:10]}"
    await create_layer(db_url, name=name, display_name="Test Receipts")
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


@contextlib.contextmanager
def _as(user: str, role: str):
    """Run as user with role, for the issuer, the tool and the resource."""
    with (
        patch("stratakb.core.auth.get_current_user", return_value=(user, role)),
        patch("stratakb.rag.receipts.get_current_user", return_value=(user, role)),
        patch("stratakb.rag.receipt_tools.get_current_user", return_value=(user, role)),
    ):
        yield


async def _ingest(db_url: str, layer: str, source: Path):
    from stratakb.knowledge.ingestion import ingest_documents

    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda texts, **_: [FAKE_EMBEDDING for _ in texts]
        return await ingest_documents(
            db_url, "fake-key", layer_name=layer, source_path=str(source)
        )


async def _latest_document(db_url: str, layer: str, relative_path: str):
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetchrow(
        """
        SELECT d.id, d.version, d.text, d.sha256 FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1 AND d.relative_path = $2
        ORDER BY d.version DESC LIMIT 1
        """,
        layer,
        relative_path,
    )


async def _cited(db_url, layer, tmp_path, question="How long are records kept?", user="alice"):
    """Ingest policy.md and issue a receipt quoting QUOTE; return (receipt_id, document)."""
    from stratakb.rag.receipts import issue_receipt

    (tmp_path / "policy.md").write_text(V1_TEXT, encoding="utf-8")
    await _ingest(db_url, layer, tmp_path)
    doc = await _latest_document(db_url, layer, "policy.md")
    start = doc["text"].index(QUOTE)
    with _as(user, "viewer"):
        receipt_id = await issue_receipt(
            db_url,
            kind="extractive",
            mode="extractive",
            question=question,
            quotes=[
                {
                    "document_id": doc["id"],
                    "char_start": start,
                    "char_end": start + len(QUOTE),
                    "clause_path": "Retention",
                }
            ],
            draft="Records are kept for 30 days.",
        )
    return receipt_id, doc


async def _call_tool(receipt_id: str) -> dict:
    from stratakb.mcp_server import mcp

    tool = mcp._tool_manager.get_tool("strata_get_receipt")
    return await tool.fn(receipt_id=receipt_id)


async def _read_resource(receipt_id: str) -> dict:
    from stratakb.mcp_server import mcp

    (content,) = list(await mcp.read_resource(f"strata://receipts/{receipt_id}"))
    assert content.mime_type == "application/json"
    return json.loads(content.content)


async def _audit_rows(db_url: str, receipt_id: str) -> list:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetch(
        "SELECT * FROM strata_core.audit_log WHERE parameters->>'receipt_id' = $1 "
        "ORDER BY created_at, id",
        receipt_id,
    )


async def _stored_question(db_url: str, receipt_id: str) -> str:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetchval(
        "SELECT question FROM strata_core.receipts WHERE id = $1", receipt_id
    )


def _assert_quote_in_context(receipt: dict, doc) -> None:
    (quote,) = receipt["quotes"]
    start, end = quote["char_start"], quote["char_end"]
    assert quote["quote"] == QUOTE == doc["text"][start:end]
    assert quote["context_before"] == doc["text"][max(0, start - 300):start]
    assert quote["context_after"] == doc["text"][end:end + 300]
    assert "Intro paragraph" in quote["context_before"]
    assert "Closing remarks" in quote["context_after"]
    assert quote["version"] == doc["version"]
    assert quote["sha256"] == doc["sha256"]
    assert quote["document_id"] == doc["id"]
    assert quote["clause_path"] == "Retention"
    assert quote["text_origin"] == "text"
    assert quote["file"] == "policy.md"


class TestReceipts:
    async def test_issue_and_open_via_tool(self, db_url, layer, tmp_path):
        receipt_id, doc = await _cited(db_url, layer, tmp_path)
        assert receipt_id.startswith("rcpt_")

        with _as("alice", "viewer"):
            result = await _call_tool(receipt_id)
        assert result["status"] == "ok"
        receipt = result["receipt"]
        assert receipt["receipt_id"] == receipt_id
        assert receipt["kind"] == "extractive"
        assert receipt["mode"] == "extractive"
        assert receipt["layers"] == [layer]
        assert receipt["draft"] == "Records are kept for 30 days."
        _assert_quote_in_context(receipt, doc)
        assert receipt["quotes"][0]["changed"] is None
        assert receipt["quotes"][0]["note"] is None

    async def test_open_via_resource(self, db_url, layer, tmp_path):
        receipt_id, doc = await _cited(db_url, layer, tmp_path)
        with _as("alice", "viewer"):
            receipt = await _read_resource(receipt_id)
        assert receipt["receipt_id"] == receipt_id
        _assert_quote_in_context(receipt, doc)

    async def test_resource_template_listed(self):
        from stratakb.mcp_server import mcp

        templates = await mcp.list_resource_templates()
        assert "strata://receipts/{receipt_id}" in [t.uriTemplate for t in templates]

    async def test_changed_source_flagged(self, db_url, layer, tmp_path):
        receipt_id, v1 = await _cited(db_url, layer, tmp_path)
        (tmp_path / "policy.md").write_text(
            V1_TEXT.replace("30 days", "13 days"), encoding="utf-8"
        )
        await _ingest(db_url, layer, tmp_path)
        assert (await _latest_document(db_url, layer, "policy.md"))["version"] == 2

        with _as("alice", "viewer"):
            receipt = (await _call_tool(receipt_id))["receipt"]
        (quote,) = receipt["quotes"]
        assert quote["changed"] == "v1 → v2"
        assert quote["note"] == "source changed since this answer (v1 → v2)"
        _assert_quote_in_context(receipt, v1)

    async def test_changed_after_preset_reapply(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import clear_layer

        receipt_id, v1 = await _cited(db_url, layer, tmp_path)
        # A preset apply clears the layer (every ingested_files row goes,
        # which nulls documents.file_id), then ingests again
        await clear_layer(db_url, layer)
        (tmp_path / "policy.md").write_text(
            V1_TEXT.replace("30 days", "13 days"), encoding="utf-8"
        )
        await _ingest(db_url, layer, tmp_path)

        with _as("alice", "viewer"):
            receipt = (await _call_tool(receipt_id))["receipt"]
        assert receipt["quotes"][0]["changed"] == "v1 → v2"
        _assert_quote_in_context(receipt, v1)

    async def test_delete_cited_layer(self, db_url, layer, tmp_path):
        from stratakb.knowledge.layers import delete_layer

        receipt_id, _ = await _cited(db_url, layer, tmp_path)
        await delete_layer(db_url, name=layer)

        with _as("alice", "viewer"):
            result = await _call_tool(receipt_id)
        assert result["status"] == "ok"
        (quote,) = result["receipt"]["quotes"]
        assert quote["quote"] == QUOTE
        assert quote["note"] == "source no longer in the knowledge base"
        assert quote["context_before"] == quote["context_after"] == ""

    async def test_refused_without_layer_access(self, db_url, layer, tmp_path):
        receipt_id, _ = await _cited(db_url, layer, tmp_path)

        # Another viewer who may read every layer opens it
        with _as("bob", "viewer"):
            assert (await _call_tool(receipt_id))["status"] == "ok"

        # Without access to the cited layer, the tool and the resource refuse
        with (
            _as("bob", "viewer"),
            patch("stratakb.rag.receipts.readable_layers", AsyncMock(return_value=set())),
        ):
            refused = await _call_tool(receipt_id)
            with pytest.raises(Exception, match="may not open it"):
                await _read_resource(receipt_id)
        assert refused["status"] == "error"
        assert "receipt" not in refused

        # The issuer still opens it
        with (
            _as("alice", "viewer"),
            patch("stratakb.rag.receipts.readable_layers", AsyncMock(return_value=set())),
        ):
            assert (await _call_tool(receipt_id))["status"] == "ok"

    async def test_unknown_id_and_refusal_read_the_same(self, db_url, layer, tmp_path):
        receipt_id, _ = await _cited(db_url, layer, tmp_path)
        with (
            _as("bob", "viewer"),
            patch("stratakb.rag.receipts.readable_layers", AsyncMock(return_value=set())),
        ):
            refused = await _call_tool(receipt_id)
            unknown = await _call_tool("rcpt_does_not_exist")
        assert refused == unknown

    async def test_resource_requires_viewer(self, db_url, layer, tmp_path):
        receipt_id, _ = await _cited(db_url, layer, tmp_path)
        with _as("guest", "none"), pytest.raises(Exception, match="Permission denied"):
            await _read_resource(receipt_id)

    async def test_audit_rows(self, db_url, layer, tmp_path):
        receipt_id, _ = await _cited(db_url, layer, tmp_path)
        with _as("alice", "viewer"):
            await _call_tool(receipt_id)
        with (
            _as("bob", "viewer"),
            patch("stratakb.rag.receipts.readable_layers", AsyncMock(return_value=set())),
        ):
            await _call_tool(receipt_id)

        rows = await _audit_rows(db_url, receipt_id)
        assert [(r["action"], r["user_id"], r["result_summary"]) for r in rows] == [
            ("receipt_issue", "alice", "Issued receipt with 1 quotes"),
            ("receipt_open", "alice", "opened"),
            ("receipt_open", "bob", "refused"),
        ]
        assert rows[1]["tool_name"] == "strata_get_receipt"
        # Ids and counts only, never quote text
        for r in rows:
            assert QUOTE not in (r["parameters"] or "")
            assert set(json.loads(r["parameters"])) <= {"receipt_id", "quotes", "layers"}

    async def test_question_iban_scrubbed(self, db_url, layer, tmp_path):
        receipt_id, _ = await _cited(
            db_url, layer, tmp_path,
            question="How long is GB82WEST12345698765432 kept?",
        )
        stored = await _stored_question(db_url, receipt_id)
        assert "GB82WEST12345698765432" not in stored
        assert "[IBAN_REDACTED]" in stored
        with _as("alice", "viewer"):
            receipt = (await _call_tool(receipt_id))["receipt"]
        assert receipt["question"] == stored

    async def test_referenced_version_kept(self, db_url, layer, tmp_path):
        from stratakb.core.database import get_pool
        from stratakb.knowledge.ingestion import clear_layer

        receipt_id, v1 = await _cited(db_url, layer, tmp_path)
        (tmp_path / "policy.md").write_text("# Retention\n\nRewritten.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        await clear_layer(db_url, layer)

        pool = await get_pool(db_url)
        assert await pool.fetchval(
            "SELECT text FROM strata_vectors.documents WHERE id = $1", v1["id"]
        ) == v1["text"]
        # Version 2 has no receipt and no chunks left, so it is pruned
        assert [
            r["version"]
            for r in await pool.fetch(
                """
                SELECT d.version FROM strata_vectors.documents d
                JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
                WHERE kl.name = $1
                """,
                layer,
            )
        ] == [1]

    async def test_issue_refuses_bad_input(self, db_url, layer, tmp_path):
        from stratakb.rag.receipts import ReceiptError, issue_receipt

        _, doc = await _cited(db_url, layer, tmp_path)
        with _as("alice", "viewer"):
            with pytest.raises(ReceiptError, match="kind"):
                await issue_receipt(db_url, kind="other", mode="x", question=None, quotes=[])
            with pytest.raises(ReceiptError, match="outside"):
                await issue_receipt(
                    db_url, kind="verify", mode="grounded", question=None,
                    quotes=[{"document_id": doc["id"], "char_start": 0,
                             "char_end": len(doc["text"]) + 1}],
                )
            with pytest.raises(ReceiptError, match="Unknown document"):
                await issue_receipt(
                    db_url, kind="verify", mode="grounded", question=None,
                    quotes=[{"document_id": -1, "char_start": 0, "char_end": 1}],
                )
