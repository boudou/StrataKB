"""The strata_verify tool: check a drafted answer against the passages it cites.

The passages are named by the refs strata_ask returned, in S-id order, and
their text is always read from the store, never taken from the caller. A
ref is bounded: its span must lie inside its document and be at most twice
the layer's chunk_size long, so a citation cannot be widened to a whole
document. A verified draft gets a receipt, and a draft with nothing to check
gets none; every call is audited with counts only, never the draft.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass

import structlog
from mcp.server.fastmcp import FastMCP

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import get_current_user, require_role
from stratakb.core.database import get_pool
from stratakb.core.logging import log_tool_call
from stratakb.core.migrations import ensure_schema
from stratakb.knowledge.layers import DEFAULT_CHUNK_SIZE
from stratakb.rag.passages import ChunkRef, DocumentRef, parse_ref, passage_id
from stratakb.rag.receipts import ReceiptError, issue_receipt, readable_layers
from stratakb.rag.verify import MAX_FLAGS, Flag, check

logger = structlog.get_logger()

# Limits on one call, so a check stays quick
MAX_PASSAGES = 50
MAX_DRAFT_CHARS = 50_000

# A passage may be at most this many times its layer's chunk_size long
MAX_REF_WIDTH_FACTOR = 2

# Chunks stored before the context prefix left content still start with it
_LEGACY_PREFIX = re.compile(r"\A\[[^\]\n]*\]\n")

# The same message for an unknown passage and an unreadable one
_NOT_READABLE = "Passage {pid} ({ref}) was not found, or you may not read it."

# Why a draft with citations but no quote or fact to check has no receipt
_NOTHING_CHECKED = (
    "The draft has no quote and no fact to check, so it is not verified and has no receipt. "
    'Quote the passage that supports it as "..." [S1] and verify again, or say the answer is '
    "unverified."
)


class _RefusedError(Exception):
    """A passage that does not exist, is malformed or may not be read."""


@dataclass
class _Passage:
    """One cited passage: its S-id, ref, layer and stored text.

    text is None exactly when problem says why (the span is out of bounds);
    span is the document span for a d-ref, None for a c-ref.
    """

    id: str
    ref: str
    layer: str
    text: str | None
    problem: str | None
    span: DocumentRef | None


async def _resolve(database_url: str, refs: list[str]) -> list[_Passage]:
    """Each ref's layer and stored text, with its span checked, in order.

    Raises:
        _RefusedError: For a malformed ref or one that names nothing stored.
    """
    parsed: list[tuple[str, str, DocumentRef | ChunkRef]] = []
    for n, ref in enumerate(refs, start=1):
        pid = passage_id(n)
        target = parse_ref(ref)
        if target is None:
            raise _RefusedError(_NOT_READABLE.format(pid=pid, ref=ref[:40]))
        parsed.append((pid, ref, target))

    documents = [t for _, _, t in parsed if isinstance(t, DocumentRef)]
    chunk_ids = sorted({t.chunk_id for _, _, t in parsed if isinstance(t, ChunkRef)})
    pool = await get_pool(database_url)
    # Only the cited span is read, cut to the widest span allowed
    doc_rows = await pool.fetch(
        """
        SELECT x.n, kl.name AS layer, char_length(d.text) AS length,
               coalesce((kl.chunk_config->>'chunk_size')::int, $4) AS chunk_size,
               substr(d.text, x.s + 1,
                      greatest(least(x.e - x.s,
                                     $5 * coalesce((kl.chunk_config->>'chunk_size')::int, $4)),
                               0)) AS text
        FROM unnest($1::bigint[], $2::int[], $3::int[]) WITH ORDINALITY AS x(id, s, e, n)
        JOIN strata_vectors.documents d ON d.id = x.id
        JOIN strata_core.knowledge_layers kl ON kl.id = d.layer_id
        """,
        [d.document_id for d in documents],
        [d.char_start for d in documents],
        [d.char_end for d in documents],
        DEFAULT_CHUNK_SIZE,
        MAX_REF_WIDTH_FACTOR,
    ) if documents else []
    by_position = {row["n"]: row for row in doc_rows}
    chunk_rows = {
        row["id"]: row
        for row in (
            await pool.fetch(
                """
                SELECT id::text AS id, layer_name AS layer, content
                FROM strata_vectors.chunks WHERE id = ANY($1::uuid[])
                """,
                chunk_ids,
            )
            if chunk_ids
            else []
        )
    }

    passages: list[_Passage] = []
    position = 0
    for pid, ref, target in parsed:
        if isinstance(target, ChunkRef):
            row = chunk_rows.get(target.chunk_id)
            if row is None:
                raise _RefusedError(_NOT_READABLE.format(pid=pid, ref=ref))
            text = _LEGACY_PREFIX.sub("", row["content"], count=1)
            passages.append(_Passage(pid, ref, row["layer"], text, None, None))
            continue
        position += 1
        row = by_position.get(position)
        if row is None:
            raise _RefusedError(_NOT_READABLE.format(pid=pid, ref=ref))
        start, end = target.char_start, target.char_end
        limit = MAX_REF_WIDTH_FACTOR * row["chunk_size"]
        text, problem = row["text"], None
        if not start <= end <= row["length"]:
            text, problem = None, "lies outside its document"
        elif end - start > limit:
            text = None
            problem = f"spans {end - start} characters, more than a passage can ({limit})"
        passages.append(_Passage(pid, ref, row["layer"], text, problem, target))
    return passages


async def verify_draft(
    database_url: str,
    draft: str,
    refs: list[str],
    user_id: str,
    role: str,
    *,
    tool_name: str = "verify_draft",
) -> dict:
    """Check the draft against the passages named by refs; issue a receipt when verified."""

    async def audit(outcome: str, **counts) -> None:
        await write_audit_log(
            database_url,
            user_id=user_id,
            action="verify",
            tool_name=tool_name,
            parameters={"passages": len(refs), **counts},
            result_summary=outcome,
        )

    if not refs:
        return {"status": "error", "error": "Pass the refs of the passages the draft cites."}
    if len(refs) > MAX_PASSAGES or len(draft) > MAX_DRAFT_CHARS:
        return {
            "status": "error",
            "error": (
                f"Verify takes at most {MAX_PASSAGES} passages and a draft of "
                f"{MAX_DRAFT_CHARS} characters."
            ),
        }

    await ensure_schema(database_url)
    try:
        passages = await _resolve(database_url, refs)
        readable = await readable_layers(database_url, user_id, role)
        for p in passages:
            if p.layer not in readable:
                raise _RefusedError(_NOT_READABLE.format(pid=p.id, ref=p.ref))
    except _RefusedError as e:
        await audit("refused")
        logger.warning("verify_refused", user=user_id)
        return {"status": "error", "error": str(e)}

    given = [p.id for p in passages]
    texts = {p.id: p.text for p in passages if p.text is not None}
    # Pure-Python and CPU-bound: kept off the event loop
    result = await asyncio.to_thread(check, draft, texts, given)
    flags = [
        Flag("passage", p.ref, f"{p.id} ({p.ref}) {p.problem}", [p.id])
        for p in passages
        if p.problem
    ] + result.flags
    total = len(flags) + result.flags_unlisted
    counts = {
        "quotes": result.quotes_checked,
        "facts": result.facts_checked,
        "flags": total,
    }
    response: dict = {
        "status": "ok",
        "result": "flagged" if total else "verified",
        "quotes_checked": result.quotes_checked,
        "facts_checked": result.facts_checked,
    }
    if total:
        response["flags"] = [f.to_dict() for f in flags[:MAX_FLAGS]]
        if total > MAX_FLAGS:
            response["more_flags"] = total - MAX_FLAGS
        await audit("flagged", **counts)
        return response
    if result.nothing_checked:
        await audit("nothing_checked", **counts)
        return {**response, "result": "nothing_checked", "note": _NOTHING_CHECKED}

    try:
        receipt_id = await issue_receipt(
            database_url,
            kind="verify",
            mode="grounded",
            question=None,
            quotes=[asdict(p.span) for p in passages if p.span is not None],
            draft=draft,
            verification={
                "result": "verified",
                "passages": [{"id": p.id, "ref": p.ref} for p in passages],
                **counts,
            },
            tool_name=tool_name,
        )
    except ReceiptError as e:
        # A cited document replaced or removed since it was read
        await audit("error", **counts)
        return {"status": "error", "error": str(e)}
    response["receipt_id"] = receipt_id
    await audit("verified", receipt_id=receipt_id, **counts)
    return response


def register_verify_tools(mcp: FastMCP, database_url: str) -> None:
    """Register strata_verify on the MCP server."""

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_verify(draft: str, passages: list[str]) -> dict:
        """Verify a drafted answer: its quotes and facts against the passages it cites.

        Call it after writing an answer from strata_ask's passages and before
        giving it. It checks, without a model and against the stored text:
        every quoted span ("..." [S2]) appears in the passage it cites; every
        number, date, amount, percentage, code (AC01, pacs.008, IBANs) and
        clause reference appears in a passage cited in the same sentence, or in
        any passage when the sentence cites none ("30 days" matches "thirty
        (30) days", "1,000.00" matches "1000", "2026-11-04" matches "4 November
        2026"); every [Sn] names a passage given. A draft with no citations at
        all is flagged. A draft with citations but no quote or fact to check
        returns nothing_checked, with no receipt. A verified draft
        gets a receipt (open it with strata_get_receipt). A paraphrase that
        changes the meaning without changing a fact is not caught.

        Args:
            draft: The answer as written, citing passages as [S1], [S2], ...
            passages: The passages' refs from strata_ask, in S-id order
                      (passages[0] is S1), for example ["d42:1830-1905"].
        """
        user_id, role = get_current_user()
        return await verify_draft(
            database_url, draft, passages, user_id, role, tool_name="strata_verify"
        )
