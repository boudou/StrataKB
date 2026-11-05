"""Tests for receipts: issuing, opening through the tool and the resource,
changed and deleted sources, access, audit, PII scrubbing and pruning.

All tests ingest through the real pipeline against the database with
embeddings mocked.
"""

from __future__ import annotations

import asyncio
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


async def _wait_until_blocked_by(pool, blocker_pid: int) -> None:
    """Wait until some other backend is waiting on a lock blocker_pid holds."""
    for _ in range(200):
        if await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
            "WHERE $1 = ANY(pg_blocking_pids(pid)))",
            blocker_pid,
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("nothing waited on the lock")


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
        assert receipt["quotes"][0]["status"] == "current"
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
        assert quote["status"] == "changed"
        assert "changed" not in quote
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
        assert receipt["quotes"][0]["status"] == "changed"
        assert receipt["quotes"][0]["note"] == "source changed since this answer (v1 → v2)"
        _assert_quote_in_context(receipt, v1)

    async def test_removed_file_reads_gone(self, db_url, layer, tmp_path):
        receipt_id, v1 = await _cited(db_url, layer, tmp_path)
        (tmp_path / "policy.md").unlink()
        (tmp_path / "other.md").write_text("# Other\n\nSomething else.\n", encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)

        with _as("alice", "viewer"):
            receipt = (await _call_tool(receipt_id))["receipt"]
        (quote,) = receipt["quotes"]
        assert quote["status"] == "gone"
        assert quote["note"] == "source no longer in the knowledge base"
        # The cited version is still stored, so its context still shows
        _assert_quote_in_context(receipt, v1)

        # The same text added back is the same version again
        (tmp_path / "policy.md").write_text(V1_TEXT, encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        with _as("alice", "viewer"):
            (quote,) = (await _call_tool(receipt_id))["receipt"]["quotes"]
        assert quote["status"] == "current"
        assert quote["note"] is None

    async def test_cleared_layer_reads_gone(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import clear_layer

        receipt_id, v1 = await _cited(db_url, layer, tmp_path)
        await clear_layer(db_url, layer)

        with _as("alice", "viewer"):
            receipt = (await _call_tool(receipt_id))["receipt"]
        assert receipt["quotes"][0]["status"] == "gone"
        _assert_quote_in_context(receipt, v1)

    async def test_reverted_text_reads_current(self, db_url, layer, tmp_path):
        from stratakb.rag.receipts import issue_receipt

        first, v1 = await _cited(db_url, layer, tmp_path)
        (tmp_path / "policy.md").write_text(
            V1_TEXT.replace("30 days", "13 days"), encoding="utf-8"
        )
        await _ingest(db_url, layer, tmp_path)
        v2 = await _latest_document(db_url, layer, "policy.md")
        with _as("alice", "viewer"):
            second = await issue_receipt(
                db_url, kind="extractive", mode="extractive", question=None,
                quotes=[{"document_id": v2["id"], "char_start": 0, "char_end": 5}],
            )

        # Back to the first text: compared only with the latest version, it is
        # stored as version 3
        (tmp_path / "policy.md").write_text(V1_TEXT, encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        assert (await _latest_document(db_url, layer, "policy.md"))["version"] == 3

        with _as("alice", "viewer"):
            (quote,) = (await _call_tool(first))["receipt"]["quotes"]
            (later,) = (await _call_tool(second))["receipt"]["quotes"]
        assert (quote["status"], quote["note"]) == ("current", None)
        assert later["status"] == "changed"
        assert later["note"] == "source changed since this answer (v2 → v3)"

    async def test_slices_count_characters(self, db_url, layer, tmp_path):
        """The SQL slices match Python's, past multi-byte text and on both sides."""
        from stratakb.rag.receipts import issue_receipt

        text = "# Ελληνικά\n\n" + "€é😀 " * 200 + "\n\n" + QUOTE + "\n\n" + "€é😀 " * 200 + "\n"
        (tmp_path / "wide.md").write_text(text, encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        doc = await _latest_document(db_url, layer, "wide.md")
        start = doc["text"].index(QUOTE)
        end = start + len(QUOTE)

        with _as("alice", "viewer"):
            receipt_id = await issue_receipt(
                db_url, kind="extractive", mode="extractive", question=None,
                quotes=[
                    {"document_id": doc["id"], "char_start": start, "char_end": end},
                    {"document_id": doc["id"], "char_start": 3, "char_end": 7},
                    {
                        "document_id": doc["id"],
                        "char_start": len(doc["text"]) - 2,
                        "char_end": len(doc["text"]),
                    },
                ],
            )
            quotes = (await _call_tool(receipt_id))["receipt"]["quotes"]

        assert [q["quote"] for q in quotes] == [
            QUOTE, doc["text"][3:7], doc["text"][-2:],
        ]
        for q in quotes:
            s, e = q["char_start"], q["char_end"]
            assert q["context_before"] == doc["text"][max(0, s - 300):s]
            assert q["context_after"] == doc["text"][e:e + 300]

    async def test_delete_cited_layer(self, db_url, layer, tmp_path):
        from stratakb.knowledge.layers import delete_layer

        receipt_id, _ = await _cited(db_url, layer, tmp_path)
        await delete_layer(db_url, name=layer)

        with _as("alice", "viewer"):
            result = await _call_tool(receipt_id)
        assert result["status"] == "ok"
        (quote,) = result["receipt"]["quotes"]
        assert quote["quote"] == QUOTE
        assert quote["status"] == "gone"
        assert quote["note"] == "source no longer in the knowledge base"
        assert quote["context_before"] == quote["context_after"] == ""

        # An admin who did not issue it opens it, through the tool and the resource
        with _as("carol", "admin"):
            result = await _call_tool(receipt_id)
            resource = await _read_resource(receipt_id)
        assert result["status"] == "ok"
        for quote in (result["receipt"]["quotes"][0], resource["quotes"][0]):
            assert quote["quote"] == QUOTE
            assert quote["status"] == "gone"
            assert quote["note"] == "source no longer in the knowledge base"
            assert quote["context_before"] == quote["context_after"] == ""

        # A viewer who did not issue it is refused like an unknown id
        with _as("bob", "viewer"):
            refused = await _call_tool(receipt_id)
            unknown = await _call_tool("rcpt_" + "A" * 22)
        assert refused["status"] == "error"
        assert refused == unknown

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
            unknown = await _call_tool("rcpt_" + "A" * 22)
        assert refused == unknown

    @pytest.mark.parametrize("bad_id", ["nope", "rcpt_short", "rcpt_" + "A" * 5000])
    async def test_malformed_id_refused_before_query(self, db_url, bad_id):
        from stratakb.core.database import get_pool

        user = f"probe_{uuid4().hex[:10]}"
        with _as("bob", "viewer"):
            expected = await _call_tool("rcpt_" + "A" * 22)
        assert expected["status"] == "error"

        untouched = AsyncMock(side_effect=AssertionError("queried for a malformed id"))
        with (
            _as(user, "viewer"),
            patch("stratakb.rag.receipts.get_pool", untouched),
            patch("stratakb.rag.receipts.ensure_schema", untouched),
        ):
            assert await _call_tool(bad_id) == expected
        untouched.assert_not_called()

        pool = await get_pool(db_url)
        (row,) = await pool.fetch(
            "SELECT * FROM strata_core.audit_log WHERE user_id = $1", user
        )
        assert row["action"] == "receipt_open"
        assert row["result_summary"] == "malformed"
        parameters = json.loads(row["parameters"])
        assert parameters["receipt_id"] == bad_id[:32]
        assert len(parameters["receipt_id"]) <= 32

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
            for start, end in ((-1, 2), (5, 2)):
                with pytest.raises(ReceiptError, match="outside"):
                    await issue_receipt(
                        db_url, kind="verify", mode="grounded", question=None,
                        quotes=[{"document_id": doc["id"], "char_start": start,
                                 "char_end": end}],
                    )
            with pytest.raises(ReceiptError, match="Unknown document"):
                await issue_receipt(
                    db_url, kind="verify", mode="grounded", question=None,
                    quotes=[{"document_id": -1, "char_start": 0, "char_end": 1}],
                )

    async def test_prune_waits_for_receipt_in_flight(self, db_url, layer, tmp_path):
        """A prune meeting an uncommitted receipt link keeps the document and the link."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.documents import prune_documents
        from stratakb.rag.receipts import new_receipt_id

        (tmp_path / "policy.md").write_text(V1_TEXT, encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        doc = await _latest_document(db_url, layer, "policy.md")
        pool = await get_pool(db_url)
        await pool.execute("DELETE FROM strata_vectors.chunks WHERE document_id = $1", doc["id"])
        layer_id = await pool.fetchval(
            "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", layer
        )
        receipt_id = new_receipt_id()

        async def prune() -> int:
            async with pool.acquire() as conn, conn.transaction():
                return await prune_documents(conn, layer_id)

        async with pool.acquire() as issuing:
            tx = issuing.transaction()
            await tx.start()
            try:
                await issuing.execute(
                    "SELECT id FROM strata_vectors.documents WHERE id = $1 FOR KEY SHARE",
                    doc["id"],
                )
                await issuing.execute(
                    """
                    INSERT INTO strata_core.receipts
                        (id, issued_by, kind, mode, layers, body)
                    VALUES ($1, 'alice', 'verify', 'grounded', '[]', '{}')
                    """,
                    receipt_id,
                )
                await issuing.execute(
                    "INSERT INTO strata_core.receipt_documents (receipt_id, document_id) "
                    "VALUES ($1, $2)",
                    receipt_id,
                    doc["id"],
                )
                pruning = asyncio.create_task(prune())
                await _wait_until_blocked_by(pool, issuing.get_server_pid())
            except BaseException:
                await tx.rollback()
                raise
            await tx.commit()
        assert await asyncio.wait_for(pruning, 10) == 0

        assert await pool.fetchval(
            "SELECT count(*) FROM strata_vectors.documents WHERE id = $1", doc["id"]
        ) == 1
        assert await pool.fetchval(
            "SELECT count(*) FROM strata_core.receipt_documents "
            "WHERE receipt_id = $1 AND document_id = $2",
            receipt_id,
            doc["id"],
        ) == 1

    async def test_issue_during_prune_names_the_error(self, db_url, layer, tmp_path):
        """A document deleted while a receipt is issued gives a named error, not asyncpg's."""
        from stratakb.core.database import get_pool
        from stratakb.rag.receipts import ReceiptError, issue_receipt

        user = f"racer_{uuid4().hex[:10]}"
        (tmp_path / "policy.md").write_text(V1_TEXT, encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        doc = await _latest_document(db_url, layer, "policy.md")
        pool = await get_pool(db_url)

        with _as(user, "viewer"):
            async with pool.acquire() as deleting:
                tx = deleting.transaction()
                await tx.start()
                try:
                    await deleting.execute(
                        "DELETE FROM strata_vectors.chunks WHERE document_id = $1", doc["id"]
                    )
                    await deleting.execute(
                        "DELETE FROM strata_vectors.documents WHERE id = $1", doc["id"]
                    )
                    issuing = asyncio.create_task(
                        issue_receipt(
                            db_url, kind="verify", mode="grounded", question=None,
                            quotes=[{"document_id": doc["id"], "char_start": 0, "char_end": 5}],
                        )
                    )
                    await _wait_until_blocked_by(pool, deleting.get_server_pid())
                except BaseException:
                    await tx.rollback()
                    raise
                await tx.commit()
            with pytest.raises(ReceiptError, match="Unknown document"):
                await asyncio.wait_for(issuing, 10)

        assert await pool.fetchval(
            "SELECT count(*) FROM strata_core.receipts WHERE issued_by = $1", user
        ) == 0
