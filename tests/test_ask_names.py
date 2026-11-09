"""strata_ask when the question names something no passage holds.

The tool is called in-process with the search (and, in extractive mode, the
cross-encoder) patched; layers, documents and receipts use the test database.
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from stratakb.rag.clause_lookup import ClauseMatches
from stratakb.rag.passages import ask_chunk
from stratakb.rag.retriever import SearchResult
from tests.helpers import DOC_TEXT, NINETY


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
def server(settings):
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(side_effect=lambda question, **_: [question]),
        ),
    ):
        yield build_server(settings)


@pytest.fixture
async def make_layer(settings):
    """Make layers in a given mode, each holding one document; yields (layer, document_id)."""
    from stratakb.core.database import get_pool
    from stratakb.knowledge.documents import store_document_version
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    db_url = settings.database_url
    names = []

    async def make(answer_mode: str, *, text: str = DOC_TEXT) -> tuple[str, int]:
        name = f"test_names_{uuid4().hex[:10]}"
        names.append(name)
        await create_layer(db_url, name=name, display_name="Names", answer_mode=answer_mode)
        pool = await get_pool(db_url)
        layer_id = await pool.fetchval(
            "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", name
        )
        async with pool.acquire() as conn, conn.transaction():
            document_id, _ = await store_document_version(
                conn, layer_id=layer_id, file_id=None, source_root="/docs",
                relative_path="contracts/acme-msa.md", text=text, page_map=None,
            )
        return name, document_id

    yield make
    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


def _hit(layer, content="Tasks run in a pool of workers.", *, similarity=0.95, **span):
    return SearchResult(
        chunk_id=f"hit-{uuid4().hex[:6]}",
        content=content,
        score=1.0,
        layer_name=layer,
        source_file="guide.md",
        source_section="section",
        doc_type="markdown",
        metadata={},
        similarity=similarity,
        document_id=span.get("document_id"),
        char_start=span.get("start"),
        char_end=span.get("end"),
    )


def _clause_chunk(layer, document_id):
    start = DOC_TEXT.index(NINETY)
    chunk = ask_chunk(
        chunk_id=None, content=NINETY, score=1.0, similarity=None, layer=layer,
        source_file="contracts/acme-msa.md", section="7 Termination", doc_type="markdown",
        metadata={}, document_id=document_id, char_start=start,
        char_end=start + len(NINETY), clause_path="7 Termination",
    )
    chunk["match"] = "exact"
    return chunk


def _looked_up(*chunks):
    return patch(
        "stratakb.rag.tools._exact_clauses",
        AsyncMock(return_value=ClauseMatches(list(chunks))),
    )


def _relevant(*words: str, score: float = 0.9):
    def score_pairs(_question, texts, _model):
        return [score if any(w in t for w in words) else 0.05 for t in texts]

    return score_pairs


async def _ask(server, results, question, **arguments) -> dict:
    with (
        patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=results)),
        patch("stratakb.rag.reranker.score_pairs", side_effect=_relevant("ninety")),
    ):
        return _payload(
            await server.call_tool("strata_ask", {"question": question, **arguments})
        )


class TestGrounded:
    async def test_absent_name_is_not_found_despite_a_high_score(self, server):
        response = await _ask(
            server, [_hit("docs")], "How does asyncio.TaskPool limit concurrency?"
        )

        assert response["status"] == "not_found"
        assert "`asyncio.TaskPool`" in response["note"]
        assert "related text" in response["note"]
        assert [p["text"] for p in response["passages"]] == ["Tasks run in a pool of workers."]

    async def test_one_present_one_absent_is_partly_found(self, server):
        response = await _ask(
            server,
            [_hit("docs", "AC04 means the account is closed.")],
            "Does AC04 differ from AC99?",
        )

        assert response["status"] == "partly_found"
        assert response["note"] == "Not in any passage: `AC99`."

    async def test_name_differing_in_case_and_spacing_is_present(self, server):
        response = await _ask(
            server, [_hit("docs", "The task pool limits workers.")], "How does TaskPool work?"
        )

        assert response["status"] == "found"
        assert "note" not in response

    async def test_name_in_the_file_path_is_present(self, server):
        hit = _hit("docs", "Limit lines to 79 characters.")
        hit = SearchResult(**{**hit.__dict__, "source_file": "peps/pep-0008.txt"})

        response = await _ask(server, [hit], "What does PEP 8 say about line length?")

        assert response["status"] == "found"

    async def test_absent_name_in_capitals_is_not_found(self, server):
        response = await _ask(
            server,
            [_hit("docs", "PYTHONHASHSEED sets the hash seed.")],
            "What does PYTHONQUILLMODE do?",
        )

        assert response["status"] == "not_found"
        assert "`PYTHONQUILLMODE`" in response["note"]

    @pytest.mark.parametrize(
        "question",
        [
            "Does HTTP send JSON?",
            "What does PEP 8 say about SEPA?",
            "HOW DO I SET PYTHONQUILLMODE?",
        ],
    )
    async def test_acronyms_keep_the_status(self, server, question):
        response = await _ask(
            server, [_hit("docs", "PEP 8 says tasks run in a pool of workers.")], question
        )

        assert response["status"] == "found"
        assert "note" not in response

    async def test_question_without_names_keeps_its_status(self, server):
        response = await _ask(server, [_hit("docs")], "q")

        assert response["status"] == "found"
        assert "note" not in response

    async def test_no_passages_still_names_what_was_asked(self, server):
        response = await _ask(server, [], "What is PyConfig_Merge?")

        assert response["status"] == "not_found"
        assert response["note"] == "Not in any passage: `PyConfig_Merge`."

    async def test_named_clause_returned_makes_absent_name_partly_found(
        self, server, make_layer
    ):
        layer, document_id = await make_layer("grounded")

        with _looked_up(_clause_chunk(layer, document_id)):
            response = await _ask(
                server, [], "What does clause 7.2 say about TaskPool?", layers=[layer]
            )

        assert response["status"] == "partly_found"
        assert response["note"] == "Not in any passage: `TaskPool`."


class TestExtractive:
    async def test_absent_name_is_not_found_with_no_receipt(self, server, make_layer):
        layer, document_id = await make_layer("extractive")
        hit = _hit(layer, "content", document_id=document_id, start=0, end=len(DOC_TEXT))

        with patch("stratakb.rag.tools.issue_receipt", AsyncMock()) as issue:
            response = await _ask(
                server, [hit], "How does asyncio.TaskPool end it in ninety days?",
                layers=[layer],
            )

        assert response["mode"] == "extractive"
        assert response["status"] == "not_found"
        assert response["answer"] == []
        assert "receipt_id" not in response
        assert "`asyncio.TaskPool`" in response["note"]
        issue.assert_not_awaited()

    async def test_one_absent_name_is_partly_found(self, server, make_layer):
        text = DOC_TEXT.replace("written notice.", "written notice under notice_period.")
        layer, document_id = await make_layer("extractive", text=text)
        hit = _hit(layer, "content", document_id=document_id, start=0, end=len(text))

        response = await _ask(
            server, [hit], "Is notice_period the same as grace_period?", layers=[layer]
        )

        assert response["status"] == "partly_found"
        assert "`grace_period`" in response["note"]
        assert "notice_period`" not in response["note"]
        assert response["receipt_id"].startswith("rcpt_")

    async def test_name_only_in_a_chosen_sentence_is_present(self, server, make_layer):
        text = DOC_TEXT.replace("written notice.", "written notice under notice_period.")
        layer, document_id = await make_layer("extractive", text=text)
        hit = _hit(layer, "content", document_id=document_id, start=0, end=len(text))

        response = await _ask(server, [hit], "What is the notice_period?", layers=[layer])

        assert response["status"] == "found"
        assert "note" not in response
        assert response["receipt_id"].startswith("rcpt_")
