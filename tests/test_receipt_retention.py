"""Tests for receipt retention: expired receipts are deleted, the document
versions only they kept are pruned, 0 keeps everything, and a prune is
audited with its counts.

All tests ingest through the real pipeline against the database with
embeddings mocked; a receipt is aged by moving its issued_at back.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

FAKE_EMBEDDING = [0.01] * 768

V1_TEXT = "# Retention\n\nRecords are kept for 30 days after closure.\n"
V2_TEXT = "# Retention\n\nRecords are kept for 60 days after closure.\n"
QUOTE = "Records are kept"


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def layer(db_url):
    """A fresh layer, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_rret_{uuid4().hex[:10]}"
    await create_layer(db_url, name=name, display_name="Test Receipt Retention")
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


async def _ingest(db_url: str, layer: str, source: Path):
    from stratakb.knowledge.ingestion import ingest_documents

    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda texts, **_: [FAKE_EMBEDDING for _ in texts]
        return await ingest_documents(
            db_url, "fake-key", layer_name=layer, source_path=str(source)
        )


async def _versions(db_url: str, layer: str) -> list[int]:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        """
        SELECT d.version FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1 ORDER BY d.version
        """,
        layer,
    )
    return [r["version"] for r in rows]


async def _cite_latest(db_url: str, layer: str) -> str:
    """Issue a receipt quoting the latest version of policy.md; return its id."""
    from stratakb.core.database import get_pool
    from stratakb.rag.receipts import issue_receipt

    pool = await get_pool(db_url)
    doc = await pool.fetchrow(
        """
        SELECT d.id, d.text FROM strata_vectors.documents d
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        WHERE kl.name = $1 AND d.relative_path = 'policy.md'
        ORDER BY d.version DESC LIMIT 1
        """,
        layer,
    )
    start = doc["text"].index(QUOTE)
    with patch("stratakb.rag.receipts.get_current_user", return_value=("alice", "viewer")):
        return await issue_receipt(
            db_url,
            kind="extractive",
            mode="extractive",
            question="How long are records kept?",
            quotes=[
                {"document_id": doc["id"], "char_start": start, "char_end": start + len(QUOTE)}
            ],
        )


async def _age(db_url: str, receipt_id: str, days: int) -> None:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    await pool.execute(
        "UPDATE strata_core.receipts SET issued_at = NOW() - make_interval(days => $2) "
        "WHERE id = $1",
        receipt_id,
        days,
    )


async def _exists(db_url: str, receipt_id: str) -> bool:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetchval(
        "SELECT EXISTS (SELECT 1 FROM strata_core.receipts WHERE id = $1)", receipt_id
    )


async def _old_version_cited(db_url: str, layer: str, tmp_path: Path) -> str:
    """policy.md v1 cited by a receipt, then rewritten to v2; return the receipt id."""
    (tmp_path / "policy.md").write_text(V1_TEXT, encoding="utf-8")
    await _ingest(db_url, layer, tmp_path)
    receipt_id = await _cite_latest(db_url, layer)
    (tmp_path / "policy.md").write_text(V2_TEXT, encoding="utf-8")
    await _ingest(db_url, layer, tmp_path)
    return receipt_id


class TestReceiptRetention:
    async def test_expired_deleted_fresh_kept(self, db_url, layer, tmp_path):
        from stratakb.rag.receipts import prune_receipts

        (tmp_path / "policy.md").write_text(V1_TEXT, encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        old = await _cite_latest(db_url, layer)
        fresh = await _cite_latest(db_url, layer)
        await _age(db_url, old, 31)
        await _age(db_url, fresh, 29)

        counts = await prune_receipts(db_url, 30)

        assert counts["receipts"] >= 1
        assert not await _exists(db_url, old)
        assert await _exists(db_url, fresh)
        # The current version keeps its chunks
        assert await _versions(db_url, layer) == [1]

    async def test_zero_keeps_all(self, db_url, layer, tmp_path):
        from stratakb.rag.receipts import prune_receipts

        receipt_id = await _old_version_cited(db_url, layer, tmp_path)
        await _age(db_url, receipt_id, 10_000)

        assert await prune_receipts(db_url, 0) == {"receipts": 0, "documents": 0}
        assert await _exists(db_url, receipt_id)
        assert await _versions(db_url, layer) == [1, 2]

    async def test_unreferenced_versions_pruned(self, db_url, layer, tmp_path):
        from stratakb.rag.receipts import prune_receipts

        receipt_id = await _old_version_cited(db_url, layer, tmp_path)
        assert await _versions(db_url, layer) == [1, 2]
        await _age(db_url, receipt_id, 91)

        counts = await prune_receipts(db_url, 90)

        assert counts["documents"] >= 1
        assert not await _exists(db_url, receipt_id)
        # v1 was kept only by the receipt; v2 is current and stays
        assert await _versions(db_url, layer) == [2]

    async def test_version_kept_by_fresh_receipt(self, db_url, layer, tmp_path):
        from stratakb.rag.receipts import prune_receipts

        (tmp_path / "policy.md").write_text(V1_TEXT, encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        old = await _cite_latest(db_url, layer)
        fresh = await _cite_latest(db_url, layer)
        (tmp_path / "policy.md").write_text(V2_TEXT, encoding="utf-8")
        await _ingest(db_url, layer, tmp_path)
        await _age(db_url, old, 91)

        await prune_receipts(db_url, 90)

        assert await _exists(db_url, fresh)
        assert await _versions(db_url, layer) == [1, 2]

    async def test_prune_audited(self, db_url, layer, tmp_path):
        from stratakb.core.database import get_pool
        from stratakb.rag.receipts import prune_receipts

        receipt_id = await _old_version_cited(db_url, layer, tmp_path)
        await _age(db_url, receipt_id, 91)
        pool = await get_pool(db_url)
        before = await pool.fetchval("SELECT COALESCE(max(id), 0) FROM strata_core.audit_log")

        counts = await prune_receipts(db_url, 90)

        rows = await pool.fetch(
            "SELECT * FROM strata_core.audit_log WHERE id > $1 AND action = 'receipt_prune'",
            before,
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["user_id"] == "system"
        assert row["tool_name"] == "receipt_retention"
        params = json.loads(row["parameters"])
        assert params == {"retention_days": 90, **counts}
        assert receipt_id not in str(dict(row))

    async def test_nothing_expired_not_audited(self, db_url, layer, tmp_path):
        from stratakb.core.database import get_pool
        from stratakb.rag.receipts import prune_receipts

        receipt_id = await _old_version_cited(db_url, layer, tmp_path)
        pool = await get_pool(db_url)
        # Age every other receipt out first, so this run finds nothing
        await prune_receipts(db_url, 1)
        before = await pool.fetchval("SELECT COALESCE(max(id), 0) FROM strata_core.audit_log")

        assert await prune_receipts(db_url, 1) == {"receipts": 0, "documents": 0}

        assert await _exists(db_url, receipt_id)
        assert not await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM strata_core.audit_log "
            "WHERE id > $1 AND action = 'receipt_prune')",
            before,
        )
