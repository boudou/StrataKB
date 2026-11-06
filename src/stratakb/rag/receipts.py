"""Receipts: a stored, checkable record of what the server returned or verified.

A receipt holds the question and any draft answer (both always PII-scrubbed),
the answer mode and each quote with the document version it came from (id,
version, SHA-256, offsets, clause path, text origin) and the quote's own text,
so it still opens after its source changed or was deleted. Opening one shows every quote in its
surrounding text with a status: current, changed or gone. Issuing and opening
are audited; the audit entry holds the receipt id and counts, never quote
text. Receipts older than RECEIPT_RETENTION_DAYS are pruned (prune_receipts).
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Mapping, Sequence

import structlog

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import check_role, get_current_user
from stratakb.core.database import get_pool
from stratakb.core.migrations import ensure_schema
from stratakb.knowledge.documents import prune_documents
from stratakb.learning.pii import scrub_query

logger = structlog.get_logger()

RECEIPT_PREFIX = "rcpt_"
RECEIPT_KINDS = ("extractive", "verify")

# The shape new_receipt_id makes: token_urlsafe(16) is 22 characters
_RECEIPT_ID = re.compile(rf"{RECEIPT_PREFIX}[A-Za-z0-9_-]{{22}}")

# Characters of a caller-supplied receipt id kept in the audit log
_AUDITED_ID_CHARS = 32

# Characters of the cited document's text shown on each side of a quote
CONTEXT_CHARS = 300

# The note on a quote whose file or layer is no longer in the knowledge base
SOURCE_GONE = "source no longer in the knowledge base"

# The same message for an unknown receipt and a refused one: an id alone
# tells the caller nothing
_NOT_READABLE = "Receipt not found, or you may not open it."


class ReceiptError(Exception):
    """A receipt could not be issued."""


class ReceiptAccessError(Exception):
    """The receipt does not exist or the caller may not open it."""


def new_receipt_id() -> str:
    """A fresh receipt id: the prefix and 16 random bytes, URL-safe."""
    return RECEIPT_PREFIX + secrets.token_urlsafe(16)


async def readable_layers(database_url: str, user_id: str, role: str) -> set[str]:
    """The layer names user_id may read.

    Every layer for viewer and above, none below. Per-layer access (#19)
    replaces only this function.
    """
    if not check_role(role, "viewer"):
        return set()
    pool = await get_pool(database_url)
    rows = await pool.fetch("SELECT name FROM strata_core.knowledge_layers")
    return {row["name"] for row in rows}


async def issue_receipt(
    database_url: str,
    *,
    kind: str,
    mode: str,
    question: str | None,
    quotes: Sequence[dict],
    draft: str | None = None,
    verification: dict | None = None,
    tool_name: str = "issue_receipt",
) -> str:
    """Store a receipt for the quoted spans and return its id.

    Each quote names a document version and a span of its text:
    {"document_id", "char_start", "char_end"} and optionally "clause_path".
    Everything else (version, SHA-256, text origin, layer, file and the quote
    text) is read from the stored document, so a receipt records what the
    knowledge base holds, not what a caller claims. The question and the
    draft are scrubbed with scrub_query whatever LEARNING_STORE_RAW_QUERIES
    says.

    Raises:
        ReceiptError: For an unknown kind, an unknown document or a span
            outside its text.
    """
    await ensure_schema(database_url)
    if kind not in RECEIPT_KINDS:
        raise ReceiptError(f"Unknown receipt kind '{kind}'; use one of {', '.join(RECEIPT_KINDS)}.")
    user_id, _ = get_current_user()
    receipt_id = new_receipt_id()
    scrubbed = scrub_query(question) if question else question

    pool = await get_pool(database_url)
    async with pool.acquire() as conn, conn.transaction():
        ids = sorted({int(q["document_id"]) for q in quotes})
        # The lock keeps a concurrent prune from deleting a row before it is
        # linked below; a row it already deleted is simply not found. In id
        # order, as prune_documents locks.
        rows = await conn.fetch(
            """
            SELECT d.id, d.version, d.sha256, d.text_origin, d.relative_path,
                   char_length(d.text) AS text_length, kl.name AS layer
            FROM strata_vectors.documents d
            JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
            WHERE d.id = ANY($1::bigint[])
            ORDER BY d.id
            FOR KEY SHARE OF d
            """,
            ids,
        )
        docs = {row["id"]: row for row in rows}

        stored: list[dict] = []
        for q in quotes:
            document_id = int(q["document_id"])
            doc = docs.get(document_id)
            if doc is None:
                raise ReceiptError(
                    f"Unknown document {document_id}: it is not stored, "
                    "or it was replaced or removed since it was found."
                )
            start, end = int(q["char_start"]), int(q["char_end"])
            if not 0 <= start <= end <= doc["text_length"]:
                raise ReceiptError(
                    f"Span {start}-{end} is outside document {document_id}'s text."
                )
            stored.append(
                {
                    "document_id": document_id,
                    "version": doc["version"],
                    "sha256": doc["sha256"],
                    "char_start": start,
                    "char_end": end,
                    "clause_path": q.get("clause_path"),
                    "text_origin": doc["text_origin"],
                    "layer": doc["layer"],
                    "file": doc["relative_path"],
                }
            )

        # Only the quoted spans are read; substr counts characters, as str does
        slices = await conn.fetch(
            """
            SELECT x.ord, substr(d.text, x.char_start + 1, x.char_end - x.char_start) AS quote
            FROM unnest($1::bigint[], $2::int[], $3::int[]) WITH ORDINALITY
                AS x(id, char_start, char_end, ord)
            JOIN strata_vectors.documents d ON d.id = x.id
            """,
            [q["document_id"] for q in stored],
            [q["char_start"] for q in stored],
            [q["char_end"] for q in stored],
        )
        for row in slices:
            stored[row["ord"] - 1]["quote"] = row["quote"]

        layers = sorted({q["layer"] for q in stored})
        body: dict = {"quotes": stored}
        if draft is not None:
            body["draft"] = scrub_query(draft)
        if verification is not None:
            body["verification"] = verification

        await conn.execute(
            """
            INSERT INTO strata_core.receipts
                (id, issued_by, kind, mode, question, layers, body)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            receipt_id,
            user_id,
            kind,
            mode,
            scrubbed,
            json.dumps(layers),
            json.dumps(body),
        )
        await conn.executemany(
            "INSERT INTO strata_core.receipt_documents (receipt_id, document_id) VALUES ($1, $2)",
            [(receipt_id, document_id) for document_id in ids],
        )

    await write_audit_log(
        database_url,
        user_id=user_id,
        action="receipt_issue",
        tool_name=tool_name,
        parameters={"receipt_id": receipt_id, "quotes": len(stored), "layers": len(layers)},
        result_summary=f"Issued receipt with {len(stored)} quotes",
    )
    logger.info("receipt_issued", quotes=len(stored), layers=len(layers))
    return receipt_id


def _source_status(quote: dict, source: Mapping | None) -> tuple[str, str | None]:
    """A quote's (status, note), from the row open_receipt read for its document.

    gone: the cited version is deleted (with its layer), or its file is no
    longer tracked (removed from its folder, or the layer cleared and not
    ingested again). Tracked by identity, not by the version's file_id: older
    versions keep the file_id of the live row. changed: the file's latest text
    differs from the cited text, by hash, since a file put back to an earlier
    text is stored as a new version. Otherwise current.
    """
    if source is None or not source["tracked"]:
        return "gone", SOURCE_GONE
    if source["latest_sha256"] != quote["sha256"]:
        change = f"v{quote['version']} → v{source['latest_version']}"
        return "changed", f"source changed since this answer ({change})"
    return "current", None


async def open_receipt(
    database_url: str,
    receipt_id: str,
    user_id: str,
    role: str,
    *,
    tool_name: str = "open_receipt",
) -> dict:
    """The receipt with each quote shown in its surrounding text.

    Allowed for the issuer, for admins and for anyone who may read every cited
    layer (a deleted layer counts as unreadable). Each quote gets
    context_before and context_after from the document version it cites, a
    status and a note saying why it is not current (see _source_status). A
    quote whose document is gone keeps its stored text, with empty context.
    An id of the wrong shape is refused before any receipt query. Refusals are
    audited too.

    Raises:
        ReceiptAccessError: If the receipt does not exist or the caller may
            not open it (one message for both).
    """

    async def audit(outcome: str, quotes: int = 0) -> None:
        await write_audit_log(
            database_url,
            user_id=user_id,
            action="receipt_open",
            tool_name=tool_name,
            parameters={"receipt_id": receipt_id[:_AUDITED_ID_CHARS], "quotes": quotes},
            result_summary=outcome,
        )

    if not _RECEIPT_ID.fullmatch(receipt_id):
        await audit("malformed")
        raise ReceiptAccessError(_NOT_READABLE)

    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        """
        SELECT id, issued_at, issued_by, kind, mode, question, layers, body
        FROM strata_core.receipts WHERE id = $1
        """,
        receipt_id,
    )

    if row is None:
        await audit("not_found")
        raise ReceiptAccessError(_NOT_READABLE)
    layers = json.loads(row["layers"])
    if (
        user_id != row["issued_by"]
        and not check_role(role, "admin")
        and not set(layers) <= await readable_layers(database_url, user_id, role)
    ):
        await audit("refused")
        logger.warning("receipt_refused", user=user_id)
        raise ReceiptAccessError(_NOT_READABLE)

    body = json.loads(row["body"])
    quotes = body.get("quotes", [])
    # One row per quote: only the context shown is read, in characters as
    # str counts them. A cited document that is gone has no row.
    sources = {
        r["ord"]: r
        for r in await pool.fetch(
            """
            SELECT x.ord,
                   substr(d.text, greatest(x.char_start - $4::int, 0) + 1,
                          x.char_start - greatest(x.char_start - $4::int, 0)) AS context_before,
                   substr(d.text, x.char_end + 1, $4::int) AS context_after,
                   EXISTS (
                       SELECT 1 FROM strata_core.ingested_files f
                       WHERE f.layer_id = d.layer_id AND f.source_root = d.source_root
                         AND f.relative_path = d.relative_path
                   ) AS tracked,
                   latest.version AS latest_version,
                   latest.sha256 AS latest_sha256
            FROM unnest($1::bigint[], $2::int[], $3::int[]) WITH ORDINALITY
                AS x(id, char_start, char_end, ord)
            JOIN strata_vectors.documents d ON d.id = x.id
            CROSS JOIN LATERAL (
                SELECT n.version, n.sha256 FROM strata_vectors.documents n
                WHERE n.layer_id = d.layer_id AND n.source_root = d.source_root
                  AND n.relative_path = d.relative_path
                ORDER BY n.version DESC LIMIT 1
            ) latest
            """,
            [q["document_id"] for q in quotes],
            [q["char_start"] for q in quotes],
            [q["char_end"] for q in quotes],
            CONTEXT_CHARS,
        )
    }

    shown: list[dict] = []
    for position, q in enumerate(quotes, start=1):
        source = sources.get(position)
        status, note = _source_status(q, source)
        shown.append(
            {
                **q,
                "context_before": source["context_before"] if source else "",
                "context_after": source["context_after"] if source else "",
                "status": status,
                "note": note,
            }
        )

    await audit("opened", len(shown))
    result = {
        "receipt_id": row["id"],
        "issued_at": row["issued_at"].isoformat(),
        "issued_by": row["issued_by"],
        "kind": row["kind"],
        "mode": row["mode"],
        "question": row["question"],
        "layers": layers,
        "quotes": shown,
    }
    for key in ("draft", "verification"):
        if key in body:
            result[key] = body[key]
    return result


async def prune_receipts(database_url: str, retention_days: int) -> dict:
    """Delete receipts older than retention_days and the versions only they kept.

    Then sweeps every layer those receipts cited with prune_documents, which
    removes the versions no chunk and no other receipt references; current
    versions keep their chunks and stay. 0 deletes nothing. A run that deleted
    something is audited as user "system" with the counts. Safe to run from
    several servers at once: a receipt already deleted is simply not counted.

    Returns:
        {"receipts": deleted receipts, "documents": pruned document versions}
    """
    if retention_days <= 0:
        return {"receipts": 0, "documents": 0}
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    async with pool.acquire() as conn, conn.transaction():
        # NOW() is the transaction's start, so both statements share one cutoff
        cutoff = "NOW() - make_interval(days => $1)"
        layer_ids = await conn.fetch(
            f"""
            SELECT DISTINCT d.layer_id
            FROM strata_core.receipts r
            JOIN strata_core.receipt_documents rd ON rd.receipt_id = r.id
            JOIN strata_vectors.documents d ON d.id = rd.document_id
            WHERE r.issued_at < {cutoff}
            """,
            retention_days,
        )
        deleted = await conn.execute(
            f"DELETE FROM strata_core.receipts WHERE issued_at < {cutoff}", retention_days
        )
        receipts = int(deleted.split()[-1])
        documents = 0
        for row in layer_ids:
            documents += await prune_documents(conn, row["layer_id"])

    counts = {"receipts": receipts, "documents": documents}
    if receipts:
        await write_audit_log(
            database_url,
            user_id="system",
            action="receipt_prune",
            tool_name="receipt_retention",
            parameters={"retention_days": retention_days, **counts},
            result_summary=(
                f"Deleted {receipts} receipts and {documents} document versions"
            ),
        )
        logger.info("receipts_pruned", **counts)
    return counts
