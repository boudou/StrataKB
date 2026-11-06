"""Tests for receipt retention: expired receipts are deleted, the document
versions only they kept are pruned, 0 keeps everything, and a prune is
audited with its counts.

All tests ingest through the real pipeline against the database with
embeddings mocked; a receipt is aged by moving its issued_at back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Sequence
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


@pytest.fixture
async def second_layer(db_url):
    """Another fresh layer, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_rret_{uuid4().hex[:10]}"
    await create_layer(db_url, name=name, display_name="Test Receipt Retention 2")
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


async def _blocked_by(pool, pids: Sequence[int]) -> int:
    """Wait until some backend waits on a lock one of pids holds; return that pid."""
    for _ in range(200):
        pid = await pool.fetchval(
            "SELECT b FROM pg_stat_activity, unnest(pg_blocking_pids(pid)) AS b "
            "WHERE b = ANY($1::int[]) LIMIT 1",
            list(pids),
        )
        if pid is not None:
            return pid
        await asyncio.sleep(0.05)
    raise AssertionError("nothing waited on the lock")


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

    async def test_no_deadlock_with_receipt_across_layers(
        self, db_url, layer, second_layer, tmp_path
    ):
        """A receipt citing two layers, issued mid-retention, does not deadlock it.

        Document ids interleave across the layers (a1 < b1 < a2 < b2 < gate A
        < gate B). Two blockers hold the gates, so retention stops in the
        middle with some versions locked. The receipt then cites a lower
        version of the other layer and one retention holds: if retention
        locked layer by layer, it would next need the version the issuer
        holds, and PostgreSQL would abort one side with a deadlock.
        """
        import asyncpg

        from stratakb.core.database import get_pool
        from stratakb.rag.receipts import ReceiptError, issue_receipt, prune_receipts

        pool = await get_pool(db_url)
        docs: dict[str, int] = {}
        for name, layer_name in [
            ("a1", layer), ("b1", second_layer), ("a2", layer),
            ("b2", second_layer), ("gate_a", layer), ("gate_b", second_layer),
        ]:
            folder = tmp_path / name
            folder.mkdir()
            (folder / f"{name}.md").write_text(f"# {name}\n\nText of {name}.\n", encoding="utf-8")
            await _ingest(db_url, layer_name, folder)
            docs[name] = await pool.fetchval(
                """
                SELECT d.id FROM strata_vectors.documents d
                JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
                WHERE kl.name = $1 AND d.relative_path = $2
                """,
                layer_name,
                f"{name}.md",
            )
        assert sorted(docs.values()) == list(docs.values())
        # Unreferenced versions, so retention sweeps all six
        await pool.execute(
            "DELETE FROM strata_vectors.chunks WHERE document_id = ANY($1::bigint[])",
            list(docs.values()),
        )

        def quotes(*names: str) -> list[dict]:
            return [{"document_id": docs[n], "char_start": 0, "char_end": 1} for n in names]

        with patch("stratakb.rag.receipts.get_current_user", return_value=("alice", "viewer")):
            expired = await issue_receipt(
                db_url, kind="verify", mode="grounded", question=None,
                quotes=quotes("a1", "b1"),
            )
        await _age(db_url, expired, 91)

        async with pool.acquire() as gate_a, pool.acquire() as gate_b:
            gates = {}
            for conn, name in [(gate_a, "gate_a"), (gate_b, "gate_b")]:
                gates[conn.get_server_pid()] = conn.transaction()
                await gates[conn.get_server_pid()].start()
                await conn.execute(
                    "SELECT id FROM strata_vectors.documents WHERE id = $1 FOR KEY SHARE",
                    docs[name],
                )
            try:
                retention = asyncio.create_task(prune_receipts(db_url, 90))
                blocker = await _blocked_by(pool, list(gates))
                # Retention holds the lower versions of the layer it stopped in
                cited = ("b1", "a2") if blocker == gate_a.get_server_pid() else ("a1", "b1")
                with patch(
                    "stratakb.rag.receipts.get_current_user", return_value=("bob", "viewer")
                ):
                    issuing = asyncio.create_task(
                        issue_receipt(
                            db_url, kind="verify", mode="grounded", question=None,
                            quotes=quotes(*cited),
                        )
                    )
                    # The issuer waits on retention as well
                    for _ in range(200):
                        if await pool.fetchval(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname = current_database() "
                            "AND cardinality(pg_blocking_pids(pid)) > 0"
                        ) >= 2:
                            break
                        await asyncio.sleep(0.05)
                    else:
                        raise AssertionError("the issuer never waited")
            finally:
                for tx in gates.values():
                    await tx.rollback()
            outcomes = await asyncio.wait_for(
                asyncio.gather(retention, issuing, return_exceptions=True), 15
            )

        assert not any(isinstance(o, asyncpg.DeadlockDetectedError) for o in outcomes), outcomes
        counts, issued = outcomes
        assert counts["receipts"] >= 1
        assert not await _exists(db_url, expired)
        # Retention locked the cited versions first and deleted them
        assert isinstance(issued, ReceiptError), issued
        assert "Unknown document" in str(issued)
