"""strata_ask in extractive mode: which mode it picks and what it answers.

The tests call the tool in-process with the search and the cross-encoder
patched; layers, documents and receipts use the test database.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from stratakb.rag.clause_lookup import ClauseMatches
from stratakb.rag.confidence import RERANK_THRESHOLDS
from stratakb.rag.extractive import GROUNDED_REFUSED_NOTE, LEGACY_ONLY_NOTE, OCR_WARNING
from stratakb.rag.passages import ask_chunk
from stratakb.rag.retriever import SearchResult
from tests.helpers import (
    BLOCK_HIT,
    DOC_TEXT,
    LETTER,
    LONG_BLOCK,
    NINETY,
    as_admin,
    call_tool,
    ingest,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _payload(result) -> dict:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return result
    return json.loads(result[0].text)


@pytest.fixture
def settings():
    from stratakb.config import Settings, get_settings

    return Settings(
        _env_file=None,
        database_url=get_settings().database_url,
        learning_enabled=False,
        reranking_enabled=False,
        strata_user="test-user",
        strata_role="viewer",
    )


@pytest.fixture
def audit():
    return AsyncMock()


@pytest.fixture
def server(settings, audit):
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.rag.tools.write_audit_log", audit),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(side_effect=lambda question, **_: [question]),
        ),
    ):
        yield build_server(settings)


@pytest.fixture
async def make_layer(settings):
    """Make layers in a given mode, each holding DOC_TEXT; yields (layer, document_id)."""
    from stratakb.core.database import get_pool
    from stratakb.knowledge.documents import store_document_version
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    db_url = settings.database_url
    names = []

    async def make(
        answer_mode: str,
        *,
        text_origin: str = "text",
        text: str = DOC_TEXT,
        relative_path: str = "contracts/acme-msa.md",
    ) -> tuple[str, int]:
        name = f"test_extract_{uuid4().hex[:10]}"
        names.append(name)
        await create_layer(db_url, name=name, display_name="Extract", answer_mode=answer_mode)
        pool = await get_pool(db_url)
        layer_id = await pool.fetchval(
            "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", name
        )
        async with pool.acquire() as conn, conn.transaction():
            document_id, _ = await store_document_version(
                conn, layer_id=layer_id, file_id=None, source_root="/docs",
                relative_path=relative_path, text=text, page_map=None,
                text_origin=text_origin,
            )
        return name, document_id

    yield make
    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


def _hit(layer, document_id=None, span=None, *, similarity=0.9, name="hit") -> SearchResult:
    start, end = span if span else (0, len(DOC_TEXT)) if document_id else (None, None)
    return SearchResult(
        chunk_id=f"{name}-{uuid4().hex[:6]}",
        content="content",
        score=1.0,
        layer_name=layer,
        source_file=f"{name}.md",
        source_section="section",
        doc_type="markdown",
        metadata={"section_path": ["Master agreement", "7 Termination"]},
        similarity=similarity,
        document_id=document_id,
        char_start=start,
        char_end=end,
    )


def _relevant(*words: str, score: float = 0.9):
    """A cross-encoder that finds the sentences holding any of words relevant."""

    def score_pairs(_question, texts, _model):
        return [score if any(w in t for w in words) else 0.05 for t in texts]

    return score_pairs


def _looked_up(*chunks, **counts):
    """The clause lookup returns chunks, with any unread counts."""
    return patch(
        "stratakb.rag.tools._exact_clauses",
        AsyncMock(return_value=ClauseMatches(list(chunks), **counts)),
    )


async def _ask(server, results, score_pairs=None, **arguments) -> dict:
    scorer = score_pairs or _relevant("ninety")
    with (
        patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=results)),
        patch("stratakb.rag.reranker.score_pairs", side_effect=scorer),
    ):
        return _payload(
            await server.call_tool("strata_ask", {"question": "how to end it?", **arguments})
        )


class TestAskExtractive:
    async def test_answer_is_verbatim(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        response = await _ask(
            server, [_hit(layer, document_id)], _relevant("ninety", "registered"), layers=[layer]
        )

        assert response["status"] == "found"
        assert response["mode"] == "extractive"
        assert "passages" not in response
        assert [item["text"] for item in response["answer"]] == [NINETY, LETTER]
        for item in response["answer"]:
            document, span = item["ref"].split(":")
            start, end = map(int, span.split("-"))
            assert document == f"d{document_id}"
            assert item["text"] == DOC_TEXT[start:end]
            assert item["citation"]["file"] == "contracts/acme-msa.md"
            assert item["citation"]["layer"] == layer
            assert item["passage"] == "S1"
            assert item["text_origin"] == "text"
            assert "warning" not in item
            assert "score" not in item
        assert response["answer"][0]["citation"]["lines"] == "7-8"
        assert response["receipt_id"].startswith("rcpt_")

    async def test_receipt_opens_on_the_quoted_sentences(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        response = await _ask(server, [_hit(layer, document_id)], layers=[layer])
        opened = _payload(
            await server.call_tool("strata_get_receipt", {"receipt_id": response["receipt_id"]})
        )

        receipt = opened["receipt"]
        assert receipt["kind"] == "extractive"
        assert receipt["mode"] == "extractive"
        assert receipt["layers"] == [layer]
        assert [q["quote"] for q in receipt["quotes"]] == [NINETY]
        assert receipt["quotes"][0]["document_id"] == document_id
        assert receipt["quotes"][0]["clause_path"] == "Master agreement > 7 Termination"

    async def test_ocr_sentence_warns(self, server, make_layer):
        layer, document_id = await make_layer("extractive", text_origin="ocr")

        response = await _ask(server, [_hit(layer, document_id)], layers=[layer])

        assert response["answer"][0]["text_origin"] == "ocr"
        assert response["answer"][0]["warning"] == OCR_WARNING

    async def test_grounded_request_on_extractive_layer_stays_extractive(
        self, server, make_layer
    ):
        layer, document_id = await make_layer("extractive")

        response = await _ask(server, [_hit(layer, document_id)], layers=[layer], mode="grounded")

        assert response["mode"] == "extractive"
        assert response["note"] == GROUNDED_REFUSED_NOTE
        assert response["answer"][0]["text"] == NINETY

    async def test_extractive_request_on_grounded_layer(self, server, make_layer):
        layer, document_id = await make_layer("grounded")

        response = await _ask(
            server, [_hit(layer, document_id)], layers=[layer], mode="extractive"
        )

        assert response["mode"] == "extractive"
        assert "note" not in response
        assert response["answer"][0]["text"] == NINETY

    async def test_grounded_layer_unchanged(self, server, make_layer):
        layer, document_id = await make_layer("grounded")

        with patch("stratakb.rag.tools.issue_receipt", AsyncMock()) as issue:
            response = await _ask(server, [_hit(layer, document_id)], layers=[layer])

        issue.assert_not_called()
        assert response["mode"] == "grounded"
        assert set(response) == {"status", "mode", "passages"}

    async def test_mixed_layers_extractive(self, server, make_layer):
        grounded, _ = await make_layer("grounded")
        extractive, document_id = await make_layer("extractive")

        response = await _ask(
            server, [_hit(extractive, document_id)], layers=[grounded, extractive]
        )

        assert response["mode"] == "extractive"
        assert "note" not in response

    async def test_routing_fallback_to_all_layers_finds_the_extractive_one(
        self, server, make_layer
    ):
        grounded, _ = await make_layer("grounded")
        extractive, document_id = await make_layer("extractive")

        with patch(
            "stratakb.rag.tools._detect_layers", AsyncMock(return_value=[grounded])
        ):
            # A low-confidence result sends the question to every layer
            response = await _ask(
                server, [_hit(extractive, document_id, similarity=0.1)], top_k=1
            )

        assert response["mode"] == "extractive"

    async def test_layer_made_extractive_is_read_at_once(self, server, settings, make_layer):
        from stratakb.knowledge.layers import set_layer_answer_mode

        layer, document_id = await make_layer("grounded")
        hits = [_hit(layer, document_id)]
        assert (await _ask(server, hits, layers=[layer]))["mode"] == "grounded"

        await set_layer_answer_mode(settings.database_url, layer, "extractive")

        assert (await _ask(server, hits, layers=[layer]))["mode"] == "extractive"

    async def test_below_threshold_not_found(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        response = await _ask(
            server,
            [_hit(layer, document_id)],
            _relevant("ninety", score=RERANK_THRESHOLDS.medium_score - 0.01),
            layers=[layer],
        )

        assert response["status"] == "not_found"
        assert response["answer"] == []
        assert "receipt_id" not in response
        assert "passages" not in response
        assert response["next_step"]

    async def test_long_block_is_not_found_and_issues_no_receipt(self, server, make_layer):
        text = f"{LONG_BLOCK}\n\nA short line on something else.\n"
        layer, document_id = await make_layer("extractive", text=text)
        hit_start = text.index(BLOCK_HIT)
        span = (hit_start, hit_start + len(BLOCK_HIT) * 21)

        with patch("stratakb.rag.tools.issue_receipt", AsyncMock()) as issue:
            response = await _ask(server, [_hit(layer, document_id, span)], layers=[layer])

        assert response["status"] == "not_found"
        assert response["answer"] == []
        issue.assert_not_called()

    async def test_named_clause_in_a_long_block_is_not_found(self, server, make_layer):
        text = f"{LONG_BLOCK}\n\nA short line on something else.\n"
        layer, document_id = await make_layer("extractive", text=text)
        hit_start = text.index(BLOCK_HIT)
        chunk = ask_chunk(
            chunk_id="clause", content=text[hit_start:hit_start + 200], score=1.0,
            similarity=1.0, layer=layer, source_file="contracts/acme-msa.md", section=None,
            doc_type="markdown", metadata={}, document_id=document_id, char_start=hit_start,
            char_end=hit_start + 200, clause_path=None,
        )
        chunk["match"] = "exact"

        with _looked_up(chunk):
            response = await _ask(server, [], layers=[layer])

        assert response["status"] == "not_found"
        assert response["answer"] == []

    @staticmethod
    def _exact_chunk(layer, document_id, text):
        chunk = ask_chunk(
            chunk_id=None, content=text.rstrip(), score=1.0, similarity=None, layer=layer,
            source_file="contracts/acme-msa.md", section="7 Payment", doc_type="markdown",
            metadata={}, document_id=document_id, char_start=0,
            char_end=len(text.rstrip()), clause_path="7 Payment",
        )
        chunk["match"] = "exact"
        return chunk

    async def test_clause_missing_a_long_sentence_is_partly_found(self, server, make_layer):
        due, interest = "Payment is due within ten days.", "Interest accrues daily."
        exception = f"However, this does not apply where {'word ' * 500}in writing."
        text = f"{due} {exception} {interest}\n"
        layer, document_id = await make_layer("extractive", text=text)
        chunk = self._exact_chunk(layer, document_id, text)

        with _looked_up(chunk):
            response = await _ask(server, [], layers=[layer])

        assert [a["text"] for a in response["answer"]] == [due, interest]
        assert response["status"] == "partly_found"
        assert response["note"].startswith("7 Payment is only partly quoted")
        assert "contracts/acme-msa.md" in response["note"]
        assert response["receipt_id"].startswith("rcpt_")

    async def test_clause_over_the_budget_is_partly_found(self, server, make_layer):
        from stratakb.rag.clause_lookup import fit_to_budget

        text = " ".join(f"Rule {i} applies to {'word ' * 40}here." for i in range(60)) + "\n"
        layer, document_id = await make_layer("extractive", text=text)
        chunk = fit_to_budget(self._exact_chunk(layer, document_id, text), 500)

        with _looked_up(chunk):
            response = await _ask(server, [], layers=[layer], response_format="detailed")

        assert 0 < len(response["answer"]) < 60
        assert response["answer_status"] == "partly_found"
        assert "the rest does not fit in the answer" in response["note"]
        assert "cut" not in response["passages"][0]

    async def test_clause_left_out_by_the_budget_is_named(self, server, settings, make_layer):
        from stratakb.rag.passages import passage_cost

        due, interest = "Payment is due within ten days.\n", "Interest accrues daily.\n"
        layer, due_id = await make_layer("extractive", text=due)
        _, interest_id = await make_layer(
            "extractive", text=interest, relative_path="contracts/globex-msa.md"
        )
        a = self._exact_chunk(layer, due_id, due)
        b = self._exact_chunk(layer, interest_id, interest)
        b["clause_path"] = "8 Interest"
        settings.answer_token_budget = passage_cost(a)

        with _looked_up(a, b):
            response = await _ask(server, [], _relevant("Payment"), layers=[layer])

        assert [s["text"] for s in response["answer"]] == [due.strip()]
        assert response["status"] == "partly_found"
        assert response["note"].startswith("1 named clause left out")
        assert "8 Interest in contracts/globex-msa.md" in response["note"]
        assert "only partly quoted" not in response["note"]
        assert response["receipt_id"].startswith("rcpt_")

    async def test_matches_past_the_cap_make_found_partly_found(self, server, make_layer):
        due = "Payment is due within ten days.\n"
        layer, document_id = await make_layer("extractive", text=due)
        chunk = self._exact_chunk(layer, document_id, due)

        with _looked_up(chunk, unread=3, unread_is_minimum=True):
            response = await _ask(server, [], _relevant("Payment"), layers=[layer])

        assert response["status"] == "partly_found"
        assert "At least 3 more matching sections not read." in response["note"]

    async def test_matches_past_the_cap_stay_not_found_with_a_note(self, server, make_layer):
        text = f"{LONG_BLOCK}\n\nA short line on something else.\n"
        layer, document_id = await make_layer("extractive", text=text)
        hit_start = text.index(BLOCK_HIT)
        chunk = ask_chunk(
            chunk_id="clause", content=text[hit_start:hit_start + 200], score=1.0,
            similarity=1.0, layer=layer, source_file="contracts/acme-msa.md", section=None,
            doc_type="markdown", metadata={}, document_id=document_id, char_start=hit_start,
            char_end=hit_start + 200, clause_path=None,
        )
        chunk["match"] = "exact"

        with _looked_up(chunk, unread=2):
            response = await _ask(server, [], layers=[layer])

        assert response["status"] == "not_found"
        assert response["answer"] == []
        assert "2 more matching sections not read." in response["note"]

    async def test_answer_stays_within_the_budget(self, server, settings, make_layer):
        sentences = [f"Clause {i} says ninety {'word ' * 370}here." for i in range(5)]
        text = " ".join(sentences) + " Tail.\n"
        layer, document_id = await make_layer("extractive", text=text)
        scores = {s: 0.9 - i / 100 for i, s in enumerate(sentences)}

        def score_pairs(_question, texts, _model):
            return [scores.get(t, 0.05) for t in texts]

        response = await _ask(
            server,
            [_hit(layer, document_id, (0, len(text) - 7))],
            score_pairs,
            layers=[layer],
        )

        quoted = [a["text"] for a in response["answer"]]
        assert quoted == sentences[: len(quoted)]
        assert 0 < len(quoted) < len(sentences)
        assert len(json.dumps(response["answer"])) / 4 <= settings.answer_token_budget

    async def test_status_follows_the_sentence_score(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        response = await _ask(
            server, [_hit(layer, document_id)], _relevant("ninety", score=0.6), layers=[layer]
        )

        assert response["status"] == "partly_found"
        assert "quotes" in response["next_step"]

    async def test_no_results_not_found(self, server, make_layer):
        layer, _ = await make_layer("extractive")

        response = await _ask(server, [], layers=[layer])

        assert response["status"] == "not_found"
        assert response["answer"] == []
        assert "note" not in response

    async def test_legacy_chunks_ask_for_a_reingest(self, server, make_layer):
        layer, _ = await make_layer("extractive")

        response = await _ask(server, [_hit(layer)], layers=[layer])

        assert response["status"] == "not_found"
        assert response["note"] == LEGACY_ONLY_NOTE

    async def test_refused_grounded_and_legacy_notes_join(self, server, make_layer):
        layer, _ = await make_layer("extractive")

        response = await _ask(server, [_hit(layer)], layers=[layer], mode="grounded")

        assert response["note"] == f"{GROUNDED_REFUSED_NOTE} {LEGACY_ONLY_NOTE}"

    async def test_audit_records_mode(self, server, make_layer, audit):
        extractive, document_id = await make_layer("extractive")
        grounded, _ = await make_layer("grounded")

        await _ask(server, [_hit(extractive, document_id)], layers=[extractive])
        assert audit.await_args.kwargs["parameters"]["mode"] == "extractive"

        await _ask(server, [_hit(grounded, document_id)], layers=[grounded])
        assert audit.await_args.kwargs["parameters"]["mode"] == "grounded"

    async def test_missing_model_is_an_error_not_an_empty_answer(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        def no_model(*_):
            raise OSError("model not found")

        response = await _ask(server, [_hit(layer, document_id)], no_model, layers=[layer])

        assert response["status"] == "error"
        assert "Extractive mode needs the reranker model" in response["error"]

    async def test_invalid_mode_refused(self, server, make_layer):
        layer, document_id = await make_layer("extractive")

        with pytest.raises(ToolError):
            await _ask(server, [_hit(layer, document_id)], layers=[layer], mode="free")

    async def test_named_clause_quoted_without_the_model(self, server, settings, tmp_path):
        from stratakb.knowledge.layers import create_layer, delete_layer

        db_url = settings.database_url
        layer = f"test_extract_{uuid4().hex[:10]}"
        await create_layer(
            db_url, name=layer, display_name="Extract", answer_mode="extractive",
            chunk_config={"chunk_size": 200, "chunk_overlap": 40},
        )
        try:
            shutil.copy(FIXTURES / "contract.md", tmp_path / "contract.md")
            assert (await ingest(db_url, layer, tmp_path)).docs_failed == 0

            with (
                patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[])),
                patch("stratakb.rag.reranker.score_pairs") as score,
            ):
                response = _payload(
                    await server.call_tool(
                        "strata_ask", {"question": "what does clause 2.2 say", "layers": [layer]}
                    )
                )

            score.assert_not_called()
            assert response["status"] == "found"
            assert response["receipt_id"].startswith("rcpt_")
            assert response["answer"]
            for item in response["answer"]:
                assert item["citation"]["clause_path"].endswith("2.2 Termination for cause")
        finally:
            await delete_layer(db_url, name=layer)


def _unrouted():
    """A question no layer's routing picks up, so every layer is searched."""
    return patch("stratakb.rag.tools._detect_layers", AsyncMock(return_value=[]))


class TestModeFromPassages:
    async def test_unrouted_grounded_passages_stay_grounded(self, server, audit, make_layer):
        grounded, document_id = await make_layer("grounded")
        await make_layer("extractive")

        with _unrouted():
            response = await _ask(server, [_hit(grounded, document_id)])

        assert response["mode"] == "grounded"
        assert response["passages"]
        for absent in ("answer", "note", "receipt_id"):
            assert absent not in response
        assert audit.await_args.kwargs["parameters"]["mode"] == "grounded"
        assert audit.await_args.kwargs["parameters"]["layers"] is None

    async def test_named_layers_grounded_passages_stay_grounded(self, server, make_layer):
        grounded, document_id = await make_layer("grounded")
        extractive, _ = await make_layer("extractive")

        response = await _ask(
            server, [_hit(grounded, document_id)], layers=[grounded, extractive]
        )

        assert response["mode"] == "grounded"
        assert response["passages"]

    async def test_routing_fallback_grounded_passages_stay_grounded(self, server, make_layer):
        grounded, document_id = await make_layer("grounded")
        await make_layer("extractive")

        with patch(
            "stratakb.rag.tools._detect_layers", AsyncMock(return_value=[grounded])
        ):
            # A low-confidence result sends the question to every layer
            response = await _ask(
                server,
                [_hit(grounded, document_id, similarity=0.1)],
                top_k=1,
                response_format="detailed",
            )

        assert response["routing_fallback"]["reason"] == "low_confidence"
        assert response["mode"] == "grounded"

    async def test_mixed_passages_are_extractive(self, server, make_layer):
        grounded, grounded_id = await make_layer("grounded")
        extractive, extractive_id = await make_layer("extractive")

        with _unrouted():
            response = await _ask(
                server, [_hit(grounded, grounded_id), _hit(extractive, extractive_id)]
            )

        assert response["mode"] == "extractive"
        assert "passages" not in response
        assert "note" not in response

    async def test_mixed_passages_not_loosened_by_grounded_request(self, server, make_layer):
        grounded, grounded_id = await make_layer("grounded")
        extractive, extractive_id = await make_layer("extractive")

        with _unrouted():
            response = await _ask(
                server,
                [_hit(grounded, grounded_id), _hit(extractive, extractive_id)],
                mode="grounded",
            )

        assert response["mode"] == "extractive"
        assert response["note"] == GROUNDED_REFUSED_NOTE

    async def test_exact_clause_from_extractive_layer_makes_it_extractive(
        self, server, make_layer
    ):
        grounded, grounded_id = await make_layer("grounded")
        extractive, extractive_id = await make_layer("extractive")
        start = DOC_TEXT.index(NINETY)
        chunk = ask_chunk(
            chunk_id="clause", content=NINETY, score=1.0, similarity=1.0, layer=extractive,
            source_file="contracts/acme-msa.md", section=None, doc_type="markdown", metadata={},
            document_id=extractive_id, char_start=start, char_end=start + len(NINETY),
            clause_path=None,
        )
        chunk["match"] = "exact"

        with (
            _unrouted(),
            _looked_up(chunk),
        ):
            response = await _ask(server, [_hit(grounded, grounded_id)])

        assert response["mode"] == "extractive"
        assert response["answer"][0]["citation"]["layer"] == extractive

    async def test_passage_trimmed_by_rerank_does_not_change_the_mode(
        self, server, settings, make_layer
    ):
        from stratakb.mcp_server import build_server

        grounded, grounded_id = await make_layer("grounded")
        extractive, extractive_id = await make_layer("extractive")
        reranking = build_server(settings.model_copy(update={"reranking_enabled": True}))
        hits = [
            replace(_hit(extractive, extractive_id), content="an unrelated passage"),
            replace(_hit(grounded, grounded_id), content="the passage on ending it"),
        ]

        with _unrouted():
            response = await _ask(reranking, hits, _relevant("ending"), top_k=1)

        assert response["mode"] == "grounded"
        assert [p["citation"]["layer"] for p in response["passages"]] == [grounded]

    async def test_passage_trimmed_by_budget_does_not_change_the_mode(self, server, make_layer):
        grounded, grounded_id = await make_layer("grounded")
        extractive, extractive_id = await make_layer("extractive")
        hits = [
            replace(_hit(grounded, grounded_id), content="word " * 2000),
            _hit(extractive, extractive_id),
        ]

        with _unrouted():
            response = await _ask(server, hits)

        assert response["mode"] == "grounded"
        assert [p["citation"]["layer"] for p in response["passages"]] == [grounded]

    @pytest.mark.parametrize("with_hits", [True, False], ids=["hits", "no_hits"])
    async def test_failed_mode_read_fails_the_call(self, server, make_layer, with_hits):
        grounded, document_id = await make_layer("grounded")
        hits = [_hit(grounded, document_id)] if with_hits else []

        with (
            _unrouted(),
            patch(
                "stratakb.knowledge.layers.get_answer_modes",
                AsyncMock(side_effect=RuntimeError("down")),
            ),
        ):
            response = await _ask(server, hits)

        assert response["status"] == "error"

    async def test_no_passages_grounded_request_reports_grounded(self, server, make_layer):
        extractive, _ = await make_layer("extractive")

        response = await _ask(server, [], layers=[extractive], mode="grounded")

        assert response["status"] == "not_found"
        assert response["mode"] == "grounded"
        assert "note" not in response

    async def test_no_passages_unrouted_reports_the_strictest_searched(self, server, make_layer):
        await make_layer("extractive")

        with _unrouted():
            response = await _ask(server, [])

        assert response["status"] == "not_found"
        assert response["mode"] == "extractive"

    async def test_no_passages_extractive_request_reports_extractive(self, server, make_layer):
        grounded, _ = await make_layer("grounded")

        response = await _ask(server, [], layers=[grounded], mode="extractive")

        assert response["status"] == "not_found"
        assert response["mode"] == "extractive"


class TestSettingTheMode:
    async def test_preset_layer_reports_extractive(self, server, settings, tmp_path):
        from stratakb.knowledge.layers import delete_layer
        from stratakb.knowledge.presets import create_preset_layer, load_preset

        db_url = settings.database_url
        name = f"test_extract_{uuid4().hex[:10]}"
        path = tmp_path / "preset.yaml"
        path.write_text(
            f"name: {name}\ndisplay_name: Extract\nanswer_mode: extractive\n"
            "sources:\n  - name: docs\n    path: docs\n",
            encoding="utf-8",
        )
        try:
            created = await create_preset_layer(db_url, load_preset(path))
            assert created["layer"]["answer_mode"] == "extractive"

            listed = await call_tool(server, "strata_list_knowledge_layers")

            layer = next(row for row in listed["layers"] if row["name"] == name)
            assert layer["answer_mode"] == "extractive"
        finally:
            await delete_layer(db_url, name=name)

    async def test_create_tool_accepts_the_mode(self, server, settings):
        from stratakb.knowledge.layers import delete_layer, list_layers

        db_url = settings.database_url
        name = f"test_extract_{uuid4().hex[:10]}"
        try:
            with as_admin():
                created = await call_tool(
                    server, "strata_create_knowledge_layer",
                    name=name, display_name="Extract", answer_mode="extractive",
                )
                refused = await call_tool(
                    server, "strata_create_knowledge_layer",
                    name=f"{name}_bad", display_name="Bad", answer_mode="free",
                )

            assert created["status"] == "created"
            assert refused["status"] == "error"
            assert "answer_mode" in refused["error"]
            rows = await list_layers(db_url)
            assert next(r for r in rows if r["name"] == name)["answer_mode"] == "extractive"
        finally:
            await delete_layer(db_url, name=name)


async def _one_migration_behind(db_url: str) -> None:
    """Undo the last migration, as a database the startup update did not reach."""
    from stratakb.core import migrations
    from tests.helpers import owner_execute

    last = migrations.MIGRATIONS[-1]
    assert last.file == "16_archived_patterns.sql"
    await owner_execute(
        db_url,
        f"""
        ALTER TABLE strata_core.knowledge_layers DROP COLUMN archived_patterns;
        DELETE FROM strata_core.schema_migrations WHERE version = {last.version};
        """,
    )
    migrations._checked.clear()


class TestSchemaBehind:
    async def test_extractive_ask_updates_the_schema_and_answers(
        self, server, settings, make_layer, restore_schema
    ):
        from tests.helpers import recorded_versions

        layer, document_id = await make_layer("extractive")
        await _one_migration_behind(settings.database_url)

        response = await _ask(server, [_hit(layer, document_id)], layers=[layer])

        assert response["status"] == "found"
        assert response["mode"] == "extractive"
        assert response["answer"][0]["text"] == NINETY
        assert 16 in await recorded_versions(settings.database_url)

    async def test_failed_update_names_the_cause_and_the_fix(
        self, server, settings, make_layer, restore_schema
    ):
        from stratakb.core import migrations

        layer, document_id = await make_layer("extractive")
        await _one_migration_behind(settings.database_url)

        with patch.object(migrations, "_load_sql", return_value="SELEC 1"):
            response = await _ask(server, [_hit(layer, document_id)], layers=[layer])

        assert response["status"] == "error"
        assert "16_archived_patterns.sql failed" in response["error"]
        assert "as the database owner (psql -f)" in response["error"]
        assert "archived_patterns does not exist" not in response["error"]
