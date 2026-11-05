"""The strata_ask response: cited passages packed to a token budget.

The tool tests call strata_ask in-process through the MCP server with the
search patched; the cited documents and the learning loop use the test
database. The packing, ref and citation tests are pure.
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from stratakb.rag.passages import build_passages, cite, pack_passages, passage_ref
from stratakb.rag.retriever import SearchResult

DOC_TEXT = (
    "# Master agreement\n"
    "\n"
    "Intro line.\n"
    "\n"
    "## 7 Termination\n"
    "\n"
    "7.2 Either party may end this agreement\n"
    "with ninety days' written notice.\n"
)
CLAUSE = "7.2 Either party may end this agreement\nwith ninety days' written notice."


def _payload(result) -> dict:
    """The tool's dict from a FastMCP call_tool result."""
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
        learning_enabled=True,
        reranking_enabled=False,
        strata_user="test-user",
        strata_role="viewer",
    )


@pytest.fixture
def server(settings):
    """A StrataKB server on the test database, its settings patched in everywhere."""
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.learning.tracker.get_settings", return_value=settings),
        patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
        patch("stratakb.learning.tools.write_audit_log", AsyncMock()),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(side_effect=lambda question, **_: [question]),
        ),
    ):
        yield build_server(settings)


@pytest.fixture
async def document(settings):
    """A layer holding one stored document; yields (layer, document_id)."""
    from stratakb.core.database import get_pool
    from stratakb.knowledge.documents import store_document_version
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    db_url = settings.database_url
    name = f"test_ask_{uuid4().hex[:10]}"
    await create_layer(db_url, name=name, display_name="Test Ask")
    pool = await get_pool(db_url)
    layer_id = await pool.fetchval(
        "SELECT id FROM strata_core.knowledge_layers WHERE name = $1", name
    )
    async with pool.acquire() as conn, conn.transaction():
        document_id, _ = await store_document_version(
            conn, layer_id=layer_id, file_id=None, source_root="/docs",
            relative_path="contracts/acme-msa.md", text=DOC_TEXT, page_map=None,
        )
    yield name, document_id
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


def _result(name, layer, *, similarity=0.9, content=None, document_id=None, span=None):
    start, end = span if span else (None, None)
    return SearchResult(
        chunk_id=str(uuid4()) if name is None else name,
        content=content if content is not None else f"content for {name}",
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


def _clause_result(layer, document_id, *, similarity=0.9):
    start = DOC_TEXT.index(CLAUSE)
    return _result(
        "clause", layer, similarity=similarity, content=CLAUSE,
        document_id=document_id, span=(start, start + len(CLAUSE)),
    )


async def _ask(server, results, **arguments) -> dict:
    with patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=results)):
        return _payload(await server.call_tool("strata_ask", {"question": "q", **arguments}))


async def _delete_interaction(db_url: str, interaction_id: str) -> None:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    await pool.execute(
        "DELETE FROM strata_learning.interaction_events WHERE id = $1", interaction_id
    )


class TestPacking:
    def test_budget_packs_passages(self):
        chunks = [{"content": "x" * 1000} for _ in range(10)]

        # 1,000 characters are about 250 tokens
        assert len(pack_passages(chunks, 500)) == 2
        assert len(pack_passages(chunks, 1500)) == 6
        assert pack_passages(chunks, 1500) == chunks[:6]

    def test_first_passage_always_kept(self):
        chunks = [{"content": "x" * 10_000}, {"content": "short"}]

        assert pack_passages(chunks, 100) == chunks[:1]
        assert pack_passages([], 100) == []

    def test_packing_stops_at_the_first_passage_over_budget(self):
        chunks = [{"content": "a" * 400}, {"content": "b" * 4000}, {"content": "c" * 4}]

        assert pack_passages(chunks, 200) == chunks[:1]


class TestRefAndCitation:
    def test_ref_forms(self):
        assert passage_ref({"chunk_id": "x", "document_id": 7, "char_start": 30, "char_end": 41}) \
            == "d7:30-41"
        assert passage_ref({"chunk_id": "abc", "document_id": None, "char_start": None}) == "cabc"
        assert passage_ref({"chunk_id": "abc"}) == "cabc"

    def test_pdf_citation_has_page_and_lines_on_the_page(self):
        text = "page one\nline two\n\nfirst\nsecond\nthird"
        page_two = text.index("first")
        document = {"relative_path": "specs/a.pdf", "page_map": [[0, 1], [page_two, 2]],
                    "text": text}
        start = text.index("second")
        chunk = {"chunk_id": "c", "content": "second", "layer": "specs", "doc_type": "pdf",
                 "metadata": {}, "source_file": "a.pdf", "document_id": 1,
                 "char_start": start, "char_end": start + len("second\nthird")}

        citation = cite(chunk, document)

        assert citation == {"file": "specs/a.pdf", "page": 2, "lines": "2-3",
                            "clause_path": None, "layer": "specs"}

    def test_citation_across_pages_names_the_last_page(self):
        text = "one\n\ntwo"
        document = {"relative_path": "a.pdf", "page_map": [[0, 4], [5, 5]], "text": text}
        chunk = {"chunk_id": "c", "content": text, "layer": "l", "doc_type": "pdf",
                 "metadata": {}, "document_id": 1, "char_start": 0, "char_end": len(text)}

        citation = cite(chunk, document)

        assert (citation["page"], citation["page_end"]) == (4, 5)

    def test_legacy_chunk_cited_by_file_name(self):
        chunk = {"chunk_id": "old", "content": "[l > a > Page 3] text", "layer": "l",
                 "doc_type": "pdf", "source_file": "a.pdf", "metadata": {"page": 3},
                 "document_id": None, "char_start": None, "char_end": None}

        passage = build_passages([chunk], {})[0]

        assert passage == {
            "id": "S1",
            "ref": "cold",
            "text": "[l > a > Page 3] text",
            "citation": {"file": "a.pdf", "page": 3, "lines": None, "clause_path": None,
                         "layer": "l"},
        }


class TestAskResponse:
    async def test_concise_shape(self, server, settings, document):
        layer, document_id = document

        response = await _ask(
            server,
            [_clause_result(layer, document_id), _result("other", layer, similarity=0.7)],
            layers=[layer],
        )

        assert set(response) == {"status", "mode", "passages", "interaction_id"}
        assert response["status"] == "found"
        assert response["mode"] == "grounded"
        first, second = response["passages"]
        assert set(first) == {"id", "ref", "text", "citation"}
        start = DOC_TEXT.index(CLAUSE)
        assert first["id"] == "S1"
        assert first["ref"] == f"d{document_id}:{start}-{start + len(CLAUSE)}"
        assert first["text"] == CLAUSE
        assert first["citation"] == {
            "file": "contracts/acme-msa.md",
            "page": None,
            "lines": "7-8",
            "clause_path": "Master agreement > 7 Termination",
            "layer": layer,
        }
        assert second["id"] == "S2"
        assert second["ref"] == "cother"
        assert second["citation"]["file"] == "other.md"
        await _delete_interaction(settings.database_url, response["interaction_id"])

    async def test_detailed_shape_matches_before(self, server, settings, document):
        layer, document_id = document

        response = await _ask(
            server, [_clause_result(layer, document_id)],
            layers=[layer], response_format="detailed",
        )

        assert response["status"] == "ok"
        for key in ("question", "queries_used", "confidence", "retrieved_chunks",
                    "total_chunks", "sources", "interaction_id"):
            assert key in response, key
        chunk = response["retrieved_chunks"][0]
        assert chunk["chunk_id"] == "clause"
        assert {"score", "similarity", "rerank_score", "metadata"} <= set(chunk)
        assert response["sources"][0]["file"] == "clause.md"
        assert response["passages"][0]["citation"]["file"] == "contracts/acme-msa.md"
        await _delete_interaction(settings.database_url, response["interaction_id"])

    async def test_budget_trims_the_response(self, server, settings):
        settings.answer_token_budget = 500
        results = [_result(f"c{i}", "docs", content="x" * 1000) for i in range(10)]

        response = await _ask(server, results, layers=["docs"], response_format="detailed")

        assert [c["chunk_id"] for c in response["retrieved_chunks"]] == ["c0", "c1"]
        assert response["total_chunks"] == 2
        assert [p["id"] for p in response["passages"]] == ["S1", "S2"]
        await _delete_interaction(settings.database_url, response["interaction_id"])

    async def test_not_found_has_next_step(self, server, settings):
        response = await _ask(server, [], layers=["docs"])

        assert set(response) == {"status", "mode", "passages", "next_step", "interaction_id"}
        assert response["status"] == "not_found"
        assert response["passages"] == []
        assert "stratakb add" in response["next_step"]
        await _delete_interaction(settings.database_url, response["interaction_id"])

    async def test_weak_passages_are_not_found(self, server, settings):
        response = await _ask(server, [_result("weak", "docs", similarity=0.2)], layers=["docs"])

        assert response["status"] == "not_found"
        assert [p["id"] for p in response["passages"]] == ["S1"]
        assert response["next_step"]
        await _delete_interaction(settings.database_url, response["interaction_id"])

    async def test_partly_found_has_next_step(self, server, settings):
        response = await _ask(server, [_result("some", "docs", similarity=0.5)], layers=["docs"])

        assert response["status"] == "partly_found"
        assert response["next_step"]
        await _delete_interaction(settings.database_url, response["interaction_id"])

    async def test_unknown_response_format_refused(self, server):
        response = await _ask(server, [], layers=["docs"], response_format="full")

        assert response["status"] == "error"
        assert "concise" in response["error"]

    async def test_tracker_sees_chunks(self, server, settings):
        settings.answer_token_budget = 250
        results = [_result(f"c{i}", "docs", content="x" * 1000) for i in range(3)]

        with patch(
            "stratakb.learning.tracker.Tracker._record_safely", AsyncMock()
        ) as record:
            response = await _ask(server, results, layers=["docs"])

        kwargs = record.await_args.kwargs
        assert [c["chunk_id"] for c in kwargs["chunks"]] == ["c0"]
        assert kwargs["confidence_level"] in {"high", "medium", "low"}
        assert kwargs["tool_name"] == "strata_ask"
        assert [p["id"] for p in response["passages"]] == ["S1"]

    async def test_feedback_by_passage_id(self, server, settings):
        from stratakb.core.database import get_pool

        results = [_result(f"chunk-{i}", "docs") for i in range(3)]
        response = await _ask(server, results, layers=["docs"])
        interaction_id = response["interaction_id"]

        try:
            feedback = _payload(await server.call_tool("strata_feedback", {
                "interaction_id": interaction_id,
                "helpful": True,
                "chunk_feedback": {"S2": "useful", "chunk-0": "wrong"},
            }))
            assert feedback["status"] == "ok"

            pool = await get_pool(settings.database_url)
            rows = await pool.fetch(
                "SELECT signal_name, metadata FROM strata_learning.feedback_signals "
                "WHERE interaction_id = $1 AND signal_name LIKE 'chunk_%'",
                interaction_id,
            )
            rated = {json.loads(r["metadata"])["chunk_id"]: r["signal_name"] for r in rows}
            assert rated == {"chunk-1": "chunk_useful", "chunk-0": "chunk_wrong"}

            unknown = _payload(await server.call_tool("strata_feedback", {
                "interaction_id": interaction_id,
                "helpful": False,
                "chunk_feedback": {"S9": "wrong"},
            }))
            assert unknown["status"] == "error"
            assert "S9" in unknown["error"]
        finally:
            await _delete_interaction(settings.database_url, interaction_id)

    async def test_feedback_by_passage_id_unknown_interaction(self, server):
        feedback = _payload(await server.call_tool("strata_feedback", {
            "interaction_id": str(uuid4()),
            "helpful": True,
            "chunk_feedback": {"S1": "useful"},
        }))

        assert feedback["status"] == "error"
        assert "Unknown interaction_id" in feedback["error"]
