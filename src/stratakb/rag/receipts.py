"""Receipts: a stored, checkable record of what the server returned or verified.

A receipt holds the question (always PII-scrubbed), the answer mode and each
quote with the document version it came from (id, version, SHA-256, offsets,
clause path, text origin) and the quote's own text, so it still opens after
its source changed or was deleted. Opening one shows every quote in its
surrounding text and flags a source that changed since. Issuing and opening
are audited; the audit entry holds the receipt id and counts, never quote
text.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Sequence

import structlog

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import check_role, get_current_user
from stratakb.core.database import get_pool
from stratakb.core.migrations import ensure_schema
from stratakb.learning.pii import scrub_query

logger = structlog.get_logger()

RECEIPT_PREFIX = "rcpt_"
RECEIPT_KINDS = ("extractive", "verify")

# Characters of the cited document's text shown on each side of a quote
CONTEXT_CHARS = 300

# Shown in place of the surrounding text when the cited document is gone
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
    knowledge base holds, not what a caller claims. The question is scrubbed
    with scrub_query whatever LEARNING_STORE_RAW_QUERIES says.

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
        rows = await conn.fetch(
            """
            SELECT d.id, d.version, d.sha256, d.text_origin, d.relative_path,
                   d.text, kl.name AS layer
            FROM strata_vectors.documents d
            JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
            WHERE d.id = ANY($1::bigint[])
            """,
            ids,
        )
        docs = {row["id"]: row for row in rows}

        stored: list[dict] = []
        for q in quotes:
            document_id = int(q["document_id"])
            doc = docs.get(document_id)
            if doc is None:
                raise ReceiptError(f"Unknown document {document_id}.")
            start, end = int(q["char_start"]), int(q["char_end"])
            if not 0 <= start <= end <= len(doc["text"]):
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
                    "quote": doc["text"][start:end],
                }
            )

        layers = sorted({q["layer"] for q in stored})
        body: dict = {"quotes": stored}
        if draft is not None:
            body["draft"] = draft
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


async def open_receipt(
    database_url: str,
    receipt_id: str,
    user_id: str,
    role: str,
    *,
    tool_name: str = "open_receipt",
) -> dict:
    """The receipt with each quote shown in its surrounding text.

    Allowed for the issuer and for anyone who may read every cited layer
    (a deleted layer counts as unreadable). Each quote gets context_before
    and context_after from the document version it cites, and changed
    ("v1 → v2") when a newer version of the same file is stored, with a
    note saying so. A quote whose document is gone keeps its stored text,
    with empty context and the note SOURCE_GONE. Refusals are audited too.

    Raises:
        ReceiptAccessError: If the receipt does not exist or the caller may
            not open it (one message for both).
    """
    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        """
        SELECT id, issued_at, issued_by, kind, mode, question, layers, body
        FROM strata_core.receipts WHERE id = $1
        """,
        receipt_id,
    )

    async def audit(outcome: str, quotes: int = 0) -> None:
        await write_audit_log(
            database_url,
            user_id=user_id,
            action="receipt_open",
            tool_name=tool_name,
            parameters={"receipt_id": receipt_id, "quotes": quotes},
            result_summary=outcome,
        )

    if row is None:
        await audit("not_found")
        raise ReceiptAccessError(_NOT_READABLE)
    layers = json.loads(row["layers"])
    if user_id != row["issued_by"] and not set(layers) <= await readable_layers(
        database_url, user_id, role
    ):
        await audit("refused")
        logger.warning("receipt_refused", user=user_id)
        raise ReceiptAccessError(_NOT_READABLE)

    body = json.loads(row["body"])
    quotes = body.get("quotes", [])
    ids = sorted({q["document_id"] for q in quotes})
    docs = {
        r["id"]: r
        for r in await pool.fetch(
            """
            SELECT d.id, d.text,
                   (SELECT max(n.version) FROM strata_vectors.documents n
                    WHERE n.layer_id = d.layer_id AND n.source_root = d.source_root
                      AND n.relative_path = d.relative_path) AS latest
            FROM strata_vectors.documents d
            WHERE d.id = ANY($1::bigint[])
            """,
            ids,
        )
    }

    shown: list[dict] = []
    for q in quotes:
        entry = dict(q)
        doc = docs.get(q["document_id"])
        if doc is None:
            entry.update(context_before="", context_after="", changed=None, note=SOURCE_GONE)
        else:
            text, start, end = doc["text"], q["char_start"], q["char_end"]
            changed = (
                f"v{q['version']} → v{doc['latest']}" if doc["latest"] > q["version"] else None
            )
            entry.update(
                context_before=text[max(0, start - CONTEXT_CHARS):start],
                context_after=text[end:end + CONTEXT_CHARS],
                changed=changed,
                note=f"source changed since this answer ({changed})" if changed else None,
            )
        shown.append(entry)

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
