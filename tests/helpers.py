"""Helpers shared by test modules: import them with `from tests.helpers import ...`.

Fixtures live in `tests/conftest.py`; plain functions and constants live here.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from stratakb.config import Settings

FAKE_EMBEDDING = [0.01] * 768

# The tables strata_readonly may read, written out here so the tests catch a
# change to the list in core/migrations.py
READONLY_READABLE = (
    "strata_core.knowledge_layers",
    "strata_core.ingestion_runs",
    "strata_core.ingested_files",
    "strata_vectors.chunks",
    "strata_vectors.documents",
)


async def call_tool(server: FastMCP, name: str, /, **arguments) -> dict:
    """Call tool `name` on a built server, through its role and logging decorators."""
    return await server._tool_manager.get_tool(name).fn(**arguments)


async def call_health(settings: Settings) -> dict:
    """Call strata_admin_health on a server built from `settings`, database reported up."""
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch(
            "stratakb.mcp_server.check_connection",
            AsyncMock(return_value={"status": "ok"}),
        ),
    ):
        return await call_tool(build_server(settings), "strata_admin_health")


def write_file(path: Path, text: str) -> Path:
    """Write text to path, creating its parent folders; return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def fake_embed(texts, **_) -> list[list[float]]:
    """A side effect for a mocked embed_texts: FAKE_EMBEDDING for every text."""
    return [FAKE_EMBEDDING for _ in texts]


async def ingest_with_texts(db_url: str, layer: str, source: Path, **kwargs):
    """Run ingest_documents with mocked embeddings; return (result, texts embedded)."""
    from stratakb.knowledge.ingestion import ingest_documents

    embedded: list[str] = []

    def recording_embed(texts, **_):
        embedded.extend(texts)
        return fake_embed(texts)

    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = recording_embed
        result = await ingest_documents(
            db_url, "fake-key", layer_name=layer, source_path=str(source), **kwargs
        )
    return result, embedded


async def ingest(db_url: str, layer: str, source: Path, **kwargs):
    """Run ingest_documents with mocked embeddings; return the result."""
    result, _ = await ingest_with_texts(db_url, layer, source, **kwargs)
    return result


def file_outcomes(result) -> dict[str, tuple[str, str | None]]:
    """Map each file's relative path to its (outcome, reason) in an ingest result."""
    return {f.relative_path: (f.outcome, f.reason) for f in result.files}


def openai_settings(**overrides) -> Settings:
    """Settings for the openai embedding backend, read from no .env file."""
    from stratakb.config import Settings

    values = {
        "embedding_backend": "openai",
        "embedding_dimensions": 768,
        "strata_user": "test",
        "strata_role": "admin",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def fake_openai_client(n: int = 768) -> MagicMock:
    """An OpenAI-compatible client whose embeddings.create returns one n-length vector."""
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1] * n)])
    )
    return client


async def wait_until_blocked_by(
    pool, blocker_pid: int, waiter: asyncio.Task | None = None, *, timeout: float = 10
) -> None:
    """Wait until some other backend is waiting on a lock blocker_pid holds.

    If that does not happen within timeout seconds, or the wait fails, cancel
    and await waiter (the task expected to block), so no task is left pending.
    """
    try:
        async with asyncio.timeout(timeout):
            while not await pool.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                "WHERE $1 = ANY(pg_blocking_pids(pid)))",
                blocker_pid,
            ):
                await asyncio.sleep(0.05)
    except BaseException as exc:
        if waiter is not None:
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await waiter
        if isinstance(exc, TimeoutError):
            raise AssertionError("nothing waited on the lock") from None
        raise


# Receipts: a policy document, a quote from it, and a receipt citing that quote

V1_TEXT = (
    "# Retention\n\nIntro paragraph about the policy.\n\n"
    "Records are kept for 30 days after closure.\n\nClosing remarks.\n"
)
QUOTE = "Records are kept for 30 days after closure."


@contextlib.contextmanager
def as_user(user: str, role: str):
    """Run as user with role, for the receipt issuer, tool and resource."""
    with (
        patch("stratakb.core.auth.get_current_user", return_value=(user, role)),
        patch("stratakb.rag.receipts.get_current_user", return_value=(user, role)),
        patch("stratakb.rag.receipt_tools.get_current_user", return_value=(user, role)),
    ):
        yield


async def latest_document(db_url: str, layer: str, relative_path: str):
    """The latest stored document row (id, version, text, sha256) of a file in layer."""
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


async def cite_policy(
    db_url: str, layer: str, folder: Path,
    question: str = "How long are records kept?", user: str = "alice",
):
    """Ingest policy.md from folder and issue a receipt quoting QUOTE as user.

    Return (receipt_id, document).
    """
    from stratakb.rag.receipts import issue_receipt

    (folder / "policy.md").write_text(V1_TEXT, encoding="utf-8")
    await ingest(db_url, layer, folder)
    doc = await latest_document(db_url, layer, "policy.md")
    start = doc["text"].index(QUOTE)
    with as_user(user, "viewer"):
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


async def receipt_audit_rows(db_url: str, receipt_id: str) -> list:
    """The audit rows naming receipt_id, oldest first."""
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetch(
        "SELECT * FROM strata_core.audit_log WHERE parameters->>'receipt_id' = $1 "
        "ORDER BY created_at, id",
        receipt_id,
    )


async def owner_execute(db_url: str, sql: str) -> None:
    """Run sql on a connection of its own, as the user in db_url."""
    import asyncpg

    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def recorded_versions(db_url: str) -> set[int]:
    """The migration versions strata_core.schema_migrations records."""
    import asyncpg

    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch("SELECT version FROM strata_core.schema_migrations")
    finally:
        await conn.close()
    return {row["version"] for row in rows}


async def readonly_privileges(db_url: str) -> dict:
    """What strata_readonly may read in the database at db_url."""
    import asyncpg

    conn = await asyncpg.connect(db_url)
    try:
        tables = {
            row["t"]: row["readable"]
            for row in await conn.fetch(
                """
                SELECT format('%I.%I', n.nspname, c.relname) AS t,
                       has_table_privilege('strata_readonly', c.oid, 'SELECT') AS readable
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname IN ('strata_core', 'strata_vectors', 'strata_learning')
                  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                """
            )
        }
        learning_usage = await conn.fetchval(
            "SELECT has_schema_privilege('strata_readonly', 'strata_learning', 'USAGE')"
        )
        default_acls = await conn.fetchval(
            """
            SELECT count(*) FROM pg_default_acl d
            CROSS JOIN LATERAL aclexplode(d.defaclacl) a
            WHERE a.grantee = 'strata_readonly'::regrole
            """
        )
    finally:
        await conn.close()
    return {"tables": tables, "learning_usage": learning_usage, "default_acls": default_acls}
