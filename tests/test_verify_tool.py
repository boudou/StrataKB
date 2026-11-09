"""Tests for the strata_verify tool: refs resolved from the store, bounds,
access, receipts and the audit entry.

Documents are ingested through the real pipeline against the database with
embeddings mocked.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from tests.helpers import call_tool, ingest

IBAN = "GR1601101250000000012300695"

POLICY = (
    "# Retention\n\nIntro paragraph about the policy.\n\n"
    "Records are kept for thirty (30) days after closure.\n\n"
    f"Refunds go to account {IBAN} within 5 business days.\n\n"
    + "Closing remarks that pad the document. " * 12
)
QUOTE = "Records are kept for thirty (30) days after closure."
REFUND = f"Refunds go to account {IBAN} within 5 business days."
_NOTHING_CHECKED = (
    "The draft has no quote and no fact to check, so it is not verified and has no receipt. "
    'Quote the passage that supports it as "..." [S1] and verify again, or say the answer is '
    "unverified."
)


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

    (tmp_path / "policy.md").write_text(POLICY, encoding="utf-8")
    await ingest(db_url, layer, tmp_path)
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


async def _receipts_of(db_url: str, user: str) -> int:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetchval(
        "SELECT count(*) FROM strata_core.receipts WHERE issued_by = $1", user
    )


async def _old_receipt(db_url: str, layer: str, tmp_path: Path, user: str, *, quotes, facts) -> str:
    """A verify receipt stored as it was before a draft could return nothing_checked."""
    from stratakb.rag.receipts import issue_receipt

    doc = await _document(db_url, layer, tmp_path)
    start = doc["text"].index(QUOTE)
    span = {"document_id": doc["id"], "char_start": start, "char_end": start + len(QUOTE)}
    with _as(user):
        return await issue_receipt(
            db_url,
            kind="verify",
            mode="grounded",
            question=None,
            quotes=[span],
            draft="No facts here [S1].",
            verification={
                "result": "verified", "passages": [], "quotes": quotes, "facts": facts, "flags": 0,
            },
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
        assert "unchecked_sentences" not in result

        receipt = await open_receipt(db_url, result["receipt_id"], user, "viewer")
        assert receipt["kind"] == "verify"
        assert receipt["mode"] == "grounded"
        assert receipt["question"] is None
        assert [q["quote"] for q in receipt["quotes"]] == [QUOTE, REFUND]
        assert receipt["verification"]["result"] == "verified"
        assert [p["id"] for p in receipt["verification"]["passages"]] == ["S1", "S2"]
        assert "unchecked_sentences" not in receipt["verification"]
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

    async def test_fact_checked_against_cited_passage(
        self, server, db_url, layer, tmp_path, user
    ):
        # S1 says 30 days; S2, which the sentence cites, does not
        doc = await _document(db_url, layer, tmp_path)
        passages = [_ref(doc, QUOTE), _ref(doc, REFUND)]
        with _as(user):
            result = await _verify(
                server, "Refunds reach the account within 30 days [S2].", passages
            )
        assert result["result"] == "flagged"
        assert "receipt_id" not in result
        assert [f["message"] for f in result["flags"]] == ['"30 days" not found in S2']
        assert [f["passages"] for f in result["flags"]] == [["S2"]]

        with _as(user):
            result = await _verify(server, "Records are kept for 30 days [S1].", passages)
        assert result["result"] == "verified"
        assert result["receipt_id"].startswith("rcpt_")

    async def test_flag_list_is_bounded(self, server, db_url, layer, tmp_path, user):
        from stratakb.rag.verify import MAX_FLAGS

        doc = await _document(db_url, layer, tmp_path)
        draft = " ".join(f"Q{i}Z" for i in range(1, 2001)) + " [S1]"
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
        # No section starts at 1, so the span keeps the chunk_size bound
        whole = f"d{doc['id']}:1-{len(doc['text'])}"
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

    async def test_section_ref_verifies(self, server, db_url, layer, tmp_path, user):
        # policy.md is one section (# Retention), wider than the 200 limit
        doc = await _document(db_url, layer, tmp_path)
        assert len(doc["text"]) > 200
        section = f"d{doc['id']}:0-{len(doc['text'])}"
        with _as(user):
            result = await _verify(server, f'"{QUOTE}" [S1]', [section])
        assert result["result"] == "verified", result
        assert result["receipt_id"]

    async def test_section_ref_over_the_budget_flagged(self, db_url, layer, tmp_path, user):
        from stratakb.rag.verify_tools import verify_draft

        doc = await _document(db_url, layer, tmp_path)
        section = f"d{doc['id']}:0-{len(doc['text'])}"
        with _as(user):
            result = await verify_draft(
                db_url, "No facts here [S1].", [section], user, "viewer", answer_token_budget=100
            )
        assert result["result"] == "flagged"
        assert [f["kind"] for f in result["flags"]] == ["passage"]
        assert "more than a passage can (400)" in result["flags"][0]["message"]

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

    async def test_uncited_draft_flagged_without_receipt(
        self, server, db_url, layer, tmp_path, user
    ):
        doc = await _document(db_url, layer, tmp_path)
        with _as(user):
            result = await _verify(server, "The moon is made of cheese.", [_ref(doc, QUOTE)])
        assert result["result"] == "flagged"
        assert "receipt_id" not in result
        assert [(f["kind"], f["text"]) for f in result["flags"]] == [("citation", "no citation")]
        # Flagged even when its quote and fact match the passage
        with _as(user):
            matching = await _verify(server, f'"{QUOTE}" That is 30 days.', [_ref(doc, QUOTE)])
        assert matching["result"] == "flagged"
        assert (matching["quotes_checked"], matching["facts_checked"]) == (1, 1)
        assert [f["text"] for f in matching["flags"]] == ["no citation"]
        assert await _receipts_of(db_url, user) == 0

    async def test_uncited_quote_in_a_cited_draft_verified(
        self, server, db_url, layer, tmp_path, user
    ):
        # A quote with no citation of its own, in a draft that cites elsewhere
        doc = await _document(db_url, layer, tmp_path)
        draft = f'"{QUOTE}" Either way it ends there [S1].'
        with _as(user):
            result = await _verify(server, draft, [_ref(doc, QUOTE)])
        assert result["result"] == "verified"
        assert result["receipt_id"].startswith("rcpt_")

    async def test_nothing_to_check_has_no_receipt(self, server, db_url, layer, tmp_path, user):
        doc = await _document(db_url, layer, tmp_path)
        draft = "Termination also requires approval from the board of directors [S1]."
        with _as(user):
            result = await _verify(server, draft, [_ref(doc, QUOTE)])
        assert result == {
            "status": "ok",
            "result": "nothing_checked",
            "quotes_checked": 0,
            "facts_checked": 0,
            "note": _NOTHING_CHECKED,
        }
        assert await _receipts_of(db_url, user) == 0
        rows = await _audit_rows(db_url, user)
        assert [(r["tool_name"], r["result_summary"]) for r in rows] == [
            ("strata_verify", "nothing_checked")
        ]
        assert json.loads(rows[0]["parameters"]) == {
            "passages": 1, "quotes": 0, "facts": 0, "flags": 0,
        }
        assert "board" not in rows[0]["parameters"]

    async def test_passage_flag_beats_nothing_checked(self, server, db_url, layer, tmp_path, user):
        doc = await _document(db_url, layer, tmp_path)
        whole = f"d{doc['id']}:1-{len(doc['text'])}"
        with _as(user):
            result = await _verify(server, "No facts here [S1].", [whole])
        assert result["result"] == "flagged"
        assert [f["kind"] for f in result["flags"]] == ["passage"]
        assert "receipt_id" not in result

    async def test_old_receipt_with_nothing_checked_says_so(
        self, server, db_url, layer, tmp_path, user
    ):
        from stratakb.cli import render_receipt
        from stratakb.rag.receipts import NOTHING_CHECKED_NOTE

        receipt_id = await _old_receipt(db_url, layer, tmp_path, user, quotes=0, facts=0)
        with _as(user):
            opened = await call_tool(server, "strata_get_receipt", receipt_id=receipt_id)
        receipt = opened["receipt"]
        # The stored result stays as it was
        assert receipt["verification"]["result"] == "verified"
        assert receipt["verification_note"] == NOTHING_CHECKED_NOTE
        lines = render_receipt(receipt).split("\n")
        at = next(i for i, line in enumerate(lines) if line.startswith("Verification:"))
        assert lines[at + 1] == NOTHING_CHECKED_NOTE

    @pytest.mark.parametrize(("quotes", "facts"), [(1, 0), (0, 1)])
    async def test_receipt_with_checks_has_no_note(
        self, server, db_url, layer, tmp_path, user, quotes, facts
    ):
        from stratakb.cli import render_receipt
        from stratakb.rag.receipts import NOTHING_CHECKED_NOTE

        receipt_id = await _old_receipt(db_url, layer, tmp_path, user, quotes=quotes, facts=facts)
        with _as(user):
            opened = await call_tool(server, "strata_get_receipt", receipt_id=receipt_id)
        assert "verification_note" not in opened["receipt"]
        assert NOTHING_CHECKED_NOTE not in render_receipt(opened["receipt"])

    async def test_unchecked_sentence_listed_and_on_receipt(
        self, server, db_url, layer, tmp_path, user
    ):
        from stratakb.cli import render_receipt

        doc = await _document(db_url, layer, tmp_path)
        draft = "The moon is made of cheese. Records are kept for 30 days [S1]."
        with _as(user):
            result = await _verify(server, draft, [_ref(doc, QUOTE)])
            opened = await call_tool(server, "strata_get_receipt", receipt_id=result["receipt_id"])
        assert result["result"] == "verified"
        assert result["receipt_id"].startswith("rcpt_")
        assert result["unchecked_sentences"] == ["The moon is made of cheese."]
        assert "more_unchecked_sentences" not in result
        receipt = opened["receipt"]
        assert receipt["verification"]["unchecked_sentences"] == ["The moon is made of cheese."]
        lines = render_receipt(receipt).split("\n")
        at = lines.index("Not checked (no quote or fact to compare):")
        assert lines[at + 1] == "  The moon is made of cheese."
        rows = await _audit_rows(db_url, user)
        assert json.loads(rows[0]["parameters"]) == {
            "passages": 1, "quotes": 0, "facts": 1, "flags": 0,
            "receipt_id": result["receipt_id"],
        }

    async def test_unchecked_list_capped_end_to_end(self, server, db_url, layer, tmp_path, user):
        from stratakb.cli import render_receipt
        from stratakb.rag.verify import MAX_UNCHECKED_SENTENCES

        doc = await _document(db_url, layer, tmp_path)
        draft = "\n".join(
            ["No fact here."] * (MAX_UNCHECKED_SENTENCES + 10)
            + ["Records are kept for 30 days [S1]."]
        )
        with _as(user):
            result = await _verify(server, draft, [_ref(doc, QUOTE)])
            opened = await call_tool(server, "strata_get_receipt", receipt_id=result["receipt_id"])
        assert result["result"] == "verified"
        assert result["unchecked_sentences"] == ["No fact here."] * MAX_UNCHECKED_SENTENCES
        assert result["more_unchecked_sentences"] == 10
        receipt = opened["receipt"]
        verification = receipt["verification"]
        assert verification["unchecked_sentences"] == ["No fact here."] * MAX_UNCHECKED_SENTENCES
        assert verification["more_unchecked_sentences"] == 10
        lines = render_receipt(receipt).split("\n")
        at = lines.index("Not checked (no quote or fact to compare):")
        end = at + 1 + MAX_UNCHECKED_SENTENCES
        assert lines[at + 1 : end] == ["  No fact here."] * MAX_UNCHECKED_SENTENCES
        assert lines[end] == "  and 10 more"

    async def test_flagged_path_does_not_scrub(self, server, db_url, layer, tmp_path, user):
        from stratakb.learning.pii import scrub_query

        doc = await _document(db_url, layer, tmp_path)
        scrub = Mock(wraps=scrub_query)
        with _as(user), patch("stratakb.rag.verify_tools.scrub_query", scrub):
            flagged = await _verify(
                server,
                "The moon is made of cheese. Records are kept for 31 days [S1].",
                [_ref(doc, QUOTE)],
            )
            assert flagged["result"] == "flagged"
            assert flagged["unchecked_sentences"] == ["The moon is made of cheese."]
            scrub.assert_not_called()
            verified = await _verify(
                server,
                "Send it to DEUTDEFF first. Records are kept for 30 days [S1].",
                [_ref(doc, QUOTE)],
            )
        assert verified["result"] == "verified"
        scrub.assert_called_with("Send it to DEUTDEFF first.")

    async def test_unchecked_sentence_scrubbed_on_receipt(
        self, server, db_url, layer, tmp_path, user
    ):
        from stratakb.rag.receipts import open_receipt

        doc = await _document(db_url, layer, tmp_path)
        draft = "Send it to DEUTDEFF first. Records are kept for 30 days [S1]."
        with _as(user):
            result = await _verify(server, draft, [_ref(doc, QUOTE)])
        assert result["result"] == "verified"
        assert result["unchecked_sentences"] == ["Send it to DEUTDEFF first."]
        receipt = await open_receipt(db_url, result["receipt_id"], user, "viewer")
        assert receipt["verification"]["unchecked_sentences"] == [
            "Send it to [BIC_REDACTED] first."
        ]
        assert "DEUTDEFF" not in json.dumps(receipt["verification"])

    async def test_long_unchecked_sentence_cut(self, server, db_url, layer, tmp_path, user):
        from stratakb.rag.verify_tools import MAX_SENTENCE_CHARS

        doc = await _document(db_url, layer, tmp_path)
        long = "word " * 60 + "end."
        with _as(user):
            result = await _verify(
                server, f"{long} Records are kept for 30 days [S1].", [_ref(doc, QUOTE)]
            )
        assert result["result"] == "verified"
        [shown] = result["unchecked_sentences"]
        assert len(shown) == MAX_SENTENCE_CHARS
        assert shown.endswith("…")

    async def test_flagged_lists_unchecked(self, server, db_url, layer, tmp_path, user):
        doc = await _document(db_url, layer, tmp_path)
        draft = "The moon is made of cheese. Records are kept for 31 days [S1]."
        with _as(user):
            result = await _verify(server, draft, [_ref(doc, QUOTE)])
        assert result["result"] == "flagged"
        assert result["unchecked_sentences"] == ["The moon is made of cheese."]
        assert "receipt_id" not in result

    async def test_input_limits(self, server, user):
        with _as(user):
            assert (await _verify(server, "x", []))["status"] == "error"
            assert (await _verify(server, "x", ["c1"] * 51))["status"] == "error"

    async def test_tool_requires_viewer(self, server, user):
        from stratakb.core.auth import PermissionDeniedError

        with _as(user, role="none"), pytest.raises(PermissionDeniedError):
            await _verify(server, "x", ["d1:0-1"])
