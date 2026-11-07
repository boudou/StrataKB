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
from mcp.server.fastmcp.exceptions import ToolError

from stratakb.rag.confidence import Confidence
from stratakb.rag.passages import (
    MAX_PASSAGES,
    RESPONSE_SLACK_TOKENS,
    AskResult,
    ChunkRef,
    DocumentRef,
    SearchResponse,
    ToolFailure,
    build_passages,
    cite,
    cut_clause_note,
    estimate_tokens,
    left_out_note,
    pack_passages,
    parse_ref,
    passage_cost,
    passage_id,
    passage_number,
    passage_ref,
    sources_from,
)
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
        chunks = [{"chunk_id": f"c{i}", "content": "x" * 1000} for i in range(10)]

        # 1,000 characters are about 250 tokens, plus the passage's JSON
        cost = passage_cost(chunks[0])
        assert cost > 250 + RESPONSE_SLACK_TOKENS
        assert len(pack_passages(chunks, 500)) == 500 // cost == 1
        assert len(pack_passages(chunks, 1500)) == 1500 // cost == 4
        assert pack_passages(chunks, 1500) == chunks[:4]

    def test_cost_counts_citation(self):
        bare = {"chunk_id": "a", "content": ""}
        cited = {
            "chunk_id": "a", "content": "", "layer": "rules",
            "source_file": "rules/sepa-credit-transfer/rulebook-2025-chapter-4.md",
            "clause_path": "4 Business requirements > 4.2 Credit transfer datasets",
        }

        assert passage_cost(bare) > RESPONSE_SLACK_TOKENS
        assert passage_cost(cited) > passage_cost(bare)
        # The cost is the passage's own JSON plus the response slack
        passage = json.dumps(build_passages([cited], {})[0], indent=2, ensure_ascii=False)
        assert passage_cost(cited) == estimate_tokens(passage) + RESPONSE_SLACK_TOKENS

    def test_at_most_max_passages(self):
        chunks = [{"chunk_id": f"c{i}", "content": "x"} for i in range(60)]

        assert pack_passages(chunks, 100_000) == chunks[:MAX_PASSAGES]
        assert MAX_PASSAGES == 50

    def test_first_passage_always_kept(self):
        chunks = [{"chunk_id": "a", "content": "x" * 10_000}, {"chunk_id": "b", "content": "short"}]

        assert pack_passages(chunks, 100) == chunks[:1]
        assert pack_passages([], 100) == []

    def test_packing_stops_at_the_first_passage_over_budget(self):
        chunks = [
            {"chunk_id": "a", "content": "a" * 400},
            {"chunk_id": "b", "content": "b" * 4000},
            {"chunk_id": "c", "content": "c" * 4},
        ]

        assert pack_passages(chunks, 200) == chunks[:1]


class TestRefAndCitation:
    def test_passage_id_round_trip(self):
        for n in (1, 25, MAX_PASSAGES):
            assert passage_number(passage_id(n)) == n
        chunks = [{"chunk_id": f"c{i}", "content": "text", "layer": "l", "metadata": {}}
                  for i in range(3)]
        assert [p["id"] for p in build_passages(chunks, {})] \
            == [passage_id(1), passage_id(2), passage_id(3)]

    def test_passage_number_refuses_other_text(self):
        assert passage_number("s3") == 3
        for other in ("S0", "S01", "S", "S1 ", "chunk-1", ""):
            assert passage_number(other) is None

    def test_ref_forms(self):
        assert passage_ref({"chunk_id": "x", "document_id": 7, "char_start": 30, "char_end": 41}) \
            == "d7:30-41"
        assert passage_ref({"chunk_id": "abc", "document_id": None, "char_start": None}) == "cabc"
        assert passage_ref({"chunk_id": "abc"}) == "cabc"

    def test_parse_ref_reads_what_passage_ref_writes(self):
        chunk_id = str(uuid4())
        document = {"chunk_id": chunk_id, "document_id": 7, "char_start": 30, "char_end": 41}
        assert parse_ref(passage_ref(document)) == DocumentRef(7, 30, 41)
        assert parse_ref(passage_ref({"chunk_id": chunk_id})) == ChunkRef(chunk_id)
        assert parse_ref(passage_ref({"chunk_id": chunk_id.upper()})) == ChunkRef(chunk_id)
        for malformed in ("d7:30", "cabc", "x7:1-2", "d7:1-2 ", ""):
            assert parse_ref(malformed) is None

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
                            "clause_path": None, "layer": "specs",
                            "date": None, "date_source": None}

    def test_citation_carries_the_document_date(self):
        document = {"relative_path": "a.md", "page_map": None, "text": "one",
                    "doc_date": "2026-03-04", "doc_date_source": "front_matter"}
        chunk = {"chunk_id": "c", "content": "one", "layer": "l", "metadata": {},
                 "document_id": 1, "char_start": 0, "char_end": 3}

        citation = cite(chunk, document)

        assert (citation["date"], citation["date_source"]) == ("2026-03-04", "front_matter")

    def test_answer_item_carries_the_date(self):
        from stratakb.rag.extractive import Sentence, answer_item

        document = {"relative_path": "a.md", "page_map": None, "text": "One sentence.",
                    "text_origin": "text",
                    "doc_date": "2024-05-06", "doc_date_source": "file"}
        chunk = {"chunk_id": "c", "content": "One sentence.", "layer": "l", "metadata": {},
                 "document_id": 1, "char_start": 0, "char_end": 13}
        sentence = Sentence(1, 0, 13, "One sentence.", 0, False, 0.9)

        item = answer_item(sentence, [chunk], {1: document})

        assert (item["citation"]["date"], item["citation"]["date_source"]) == (
            "2024-05-06", "file"
        )

    def test_chunk_without_offsets_has_no_date(self):
        document = {"relative_path": "a.md", "page_map": None, "text": "one",
                    "doc_date": "2026-03-04", "doc_date_source": "front_matter"}
        chunk = {"chunk_id": "c", "content": "one", "layer": "l", "metadata": {},
                 "document_id": 1, "char_start": None, "char_end": None}

        citation = cite(chunk, document)

        assert (citation["date"], citation["date_source"]) == (None, None)

    @pytest.mark.parametrize("length", range(4))
    def test_filled_date_fits_passage_cost(self, length):
        import pydantic_core

        def tokens(response: dict) -> int:
            return estimate_tokens(pydantic_core.to_json(response, indent=2).decode())

        # Six-digit spans, across two three-digit pages
        boundary = 150_000
        document = {"relative_path": "a.pdf", "page_map": [[0, 100], [boundary, 101]],
                    "text": "x\n" * 100_000, "doc_date": "2026-03-04",
                    "doc_date_source": "front_matter"}
        chunk = {"chunk_id": "c", "content": "x" * length, "layer": "l", "doc_type": "pdf",
                 "metadata": {}, "source_file": "a.pdf", "document_id": 123_456,
                 "char_start": boundary - 10, "char_end": boundary + 10}
        passage = build_passages([chunk], {123_456: document})[0]
        assert passage["citation"]["page_end"] == 101
        assert passage["citation"]["date"] == "2026-03-04"

        response = {"status": "found", "mode": "grounded"}
        filled = tokens({**response, "passages": [passage]}) - tokens({**response, "passages": []})

        assert filled <= passage_cost(chunk)

    async def test_cited_documents_brings_the_schema_up_first(self):
        from stratakb.core.migrations import MigrationError
        from stratakb.rag.tools import _cited_documents

        chunk = {"chunk_id": "c", "document_id": 1, "char_start": 0, "char_end": 3}

        with patch(
            "stratakb.core.migrations.ensure_schema",
            AsyncMock(side_effect=MigrationError("out of date")),
        ) as ensure:
            assert await _cited_documents("postgresql://unused", [chunk]) == {}
        ensure.assert_awaited_once_with("postgresql://unused")

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
                         "layer": "l", "date": None, "date_source": None},
        }


def _answer(**fields) -> AskResult:
    defaults = {
        "question": "q",
        "queries_used": ["q"],
        "confidence": Confidence("high", 0.9, 3, "similarity"),
        "chunks": [{"chunk_id": "c1"}],
        "passages": [{"id": "S1"}],
        "status": "found",
    }
    return AskResult(**{**defaults, **fields})


class TestAskResultShapes:
    def test_detailed_key_order(self):
        answer = _answer(
            status="partly_found",
            sources=[{"file": "a.md"}],
            auto_detected_layers=["sepa"],
            routing_fallback="low_confidence",
            interaction_id="int-1",
        )

        response = answer.detailed()

        assert list(response) == [
            "status", "question", "queries_used", "confidence", "retrieved_chunks",
            "total_chunks", "answer_status", "mode", "passages", "next_step",
            "sources", "auto_detected_layers", "routing_fallback", "interaction_id",
        ]
        assert response["status"] == "ok"
        assert response["answer_status"] == "partly_found"
        assert response["mode"] == "grounded"
        assert response["total_chunks"] == 1
        assert response["routing_fallback"] == {
            "reason": "low_confidence", "searched_layers": "all",
        }

    def test_detailed_leaves_out_what_is_unset(self):
        response = _answer().detailed()

        assert list(response) == [
            "status", "question", "queries_used", "confidence", "retrieved_chunks",
            "total_chunks", "answer_status", "mode", "passages",
        ]

    def test_no_sources_asked_differs_from_none_found(self):
        assert "sources" not in _answer(sources=None).detailed()
        assert _answer(sources=[]).detailed()["sources"] == []

    def test_concise_keys(self):
        assert list(_answer().concise()) == ["status", "mode", "passages"]
        assert list(_answer(status="not_found", interaction_id="int-1").concise()) == [
            "status", "mode", "passages", "next_step", "interaction_id",
        ]
        assert _answer(status="not_found").concise()["status"] == "not_found"

    @pytest.mark.parametrize(
        ("status", "has_step"),
        [("found", False), ("partly_found", True), ("not_found", True)],
    )
    def test_next_step_only_when_not_fully_found(self, status, has_step):
        answer = _answer(status=status)

        assert ("next_step" in answer.concise()) is has_step
        assert ("next_step" in answer.detailed()) is has_step

    def test_interaction_id_in_both_shapes(self):
        answer = _answer()
        answer.interaction_id = "int-7"

        assert answer.concise()["interaction_id"] == "int-7"
        assert answer.detailed()["interaction_id"] == "int-7"

    def test_detailed_confidence_is_the_plain_dict(self):
        assert _answer().detailed()["confidence"] == {
            "level": "high", "top_score": 0.9, "relevant_sources": 3, "basis": "similarity",
        }

    def test_note_follows_the_passages(self):
        answer = _answer(status="partly_found", note="Read the rest.", interaction_id="int-1")

        assert list(answer.concise()) == [
            "status", "mode", "passages", "note", "next_step", "interaction_id",
        ]
        assert list(answer.detailed()) == [
            "status", "question", "queries_used", "confidence", "retrieved_chunks",
            "total_chunks", "answer_status", "mode", "passages", "note", "next_step",
            "interaction_id",
        ]
        assert answer.concise()["note"] == "Read the rest."
        assert answer.detailed()["note"] == "Read the rest."


def test_cut_clause_note_names_only_cut_clauses():
    def passage(path, file):
        return {"citation": {"clause_path": path, "file": file}}

    note = cut_clause_note(
        [{"cut": True}, {}],
        [passage("7 Fees", "a.md"), passage("8 Term", "b.md")],
    )

    assert note.startswith("7 Fees is only partly returned")
    assert "a.md" in note
    assert "8 Term" not in note
    assert "b.md" not in note

    unnamed = cut_clause_note([{"cut": True}], [passage(None, "a.md")])
    assert unnamed.startswith("The clause the question names is only partly returned")

    assert cut_clause_note([{}], [passage("7 Fees", "a.md")]) is None
    assert cut_clause_note([], []) is None


def test_left_out_note():
    def chunk(path, file):
        return {"source_file": file, "clause_path": path}

    assert left_out_note([], {}) is None

    assert left_out_note([chunk("7 Fees", "a.md")], {}) == (
        "1 named clause left out, over the token budget: 7 Fees in a.md. "
        "Read the files, or ask for fewer at a time."
    )
    two = left_out_note([chunk("7 Fees", "a.md"), chunk(None, "b.md")], {})
    assert two.startswith("2 named clauses left out, over the token budget: 7 Fees in a.md; b.md. ")

    assert left_out_note([], {}, 1) == (
        "More documents match than were read: 1 more matching section not read. "
        "Name a layer to narrow the question."
    )
    assert "at least 3 more matching sections" in left_out_note([], {}, 3, True)

    both = left_out_note([chunk("7 Fees", "a.md")], {}, 2)
    assert both.startswith("1 named clause left out")
    assert "ask for fewer at a time. More documents match" in both
    assert "2 more matching sections" in both


def test_left_out_note_gives_the_path_a_citation_gives():
    chunk = {
        "source_file": "a.md", "clause_path": "7 Fees", "document_id": 1,
        "char_start": 0, "char_end": 4,
    }
    documents = {1: {"text": "text", "page_map": None, "relative_path": "docs/a.md"}}

    assert "7 Fees in docs/a.md." in left_out_note([chunk], documents)


class TestSearchResponseShape:
    def test_key_order(self):
        response = SearchResponse(
            query="q",
            confidence=Confidence("low", 0.2, 0, "similarity", "careful"),
            results=[{"chunk_id": "c1"}],
            auto_detected_layers=["sepa"],
            routing_fallback="low_confidence",
            interaction_id="int-1",
        ).to_dict()

        assert list(response) == [
            "status", "query", "confidence", "results", "total",
            "auto_detected_layers", "routing_fallback", "interaction_id",
        ]
        assert response["status"] == "ok"
        assert response["total"] == 1
        assert list(response["confidence"]) == [
            "level", "top_score", "relevant_sources", "basis", "warning",
        ]
        assert response["routing_fallback"] == {
            "reason": "low_confidence", "searched_layers": "all",
        }

    def test_leaves_out_what_is_unset(self):
        response = SearchResponse(
            query="q", confidence=Confidence("high", 0.9, 3, "similarity"), results=[]
        ).to_dict()

        assert list(response) == ["status", "query", "confidence", "results", "total"]
        assert "warning" not in response["confidence"]


def test_tool_failure_dict_and_flags():
    assert ToolFailure("boom").to_dict() == {"status": "error", "error": "boom"}
    assert ToolFailure("boom").ok is False
    assert _answer().ok is True


def _source_chunk(section, *, layer="l", file="a.md", similarity=0.5, **extra):
    return {
        "layer": layer, "source_file": file, "section": section, "doc_type": "markdown",
        "similarity": similarity, "rerank_score": 0.8, **extra,
    }


class TestSourcesFrom:
    def test_one_source_per_layer_file_section_scored_on_its_first_chunk(self):
        chunks = [
            _source_chunk("s1", similarity=0.7),
            _source_chunk("s1", similarity=0.9),
            _source_chunk("s2", similarity=0.4),
            _source_chunk("s1", file="b.md", similarity=0.3),
            _source_chunk("s1", layer="m", similarity=0.2),
        ]

        assert sources_from(chunks, "similarity") == [
            {"layer": "l", "file": "a.md", "section": "s1", "doc_type": "markdown",
             "relevance_score": 0.7},
            {"layer": "l", "file": "a.md", "section": "s2", "doc_type": "markdown",
             "relevance_score": 0.4},
            {"layer": "l", "file": "b.md", "section": "s1", "doc_type": "markdown",
             "relevance_score": 0.3},
            {"layer": "m", "file": "a.md", "section": "s1", "doc_type": "markdown",
             "relevance_score": 0.2},
        ]

    def test_scores_on_the_basis_and_exact_matches_as_one(self):
        assert sources_from([_source_chunk("s")], "rerank")[0]["relevance_score"] == 0.8
        missing = _source_chunk("s", similarity=None)
        assert sources_from([missing], "similarity")[0]["relevance_score"] == 0.0
        exact = _source_chunk("s", similarity=None, match="exact")
        assert sources_from([exact], "rerank")[0]["relevance_score"] == 1.0

    def test_no_chunks_no_sources(self):
        assert sources_from([], "similarity") == []


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
            "date": None,
            "date_source": None,
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
        # Each passage costs 250 tokens of text plus its id, ref and citation
        cost = passage_cost({
            "chunk_id": "c0", "content": "x" * 1000, "source_file": "c0.md",
            "layer": "docs", "metadata": {"section_path": ["Master agreement", "7 Termination"]},
        })
        settings.answer_token_budget = 2 * cost + cost // 2
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
        with pytest.raises(ToolError, match="concise"):
            await _ask(server, [], layers=["docs"], response_format="full")

    async def test_response_format_listed_as_enum(self, server):
        tools = {t.name: t for t in await server.list_tools()}

        prop = tools["strata_ask"].inputSchema["properties"]["response_format"]
        assert prop["enum"] == ["concise", "detailed"]
        assert prop["default"] == "concise"

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

    async def test_feedback_by_lowercase_passage_id(self, server, settings):
        from stratakb.core.database import get_pool

        results = [_result(f"chunk-{i}", "docs") for i in range(3)]
        response = await _ask(server, results, layers=["docs"])
        interaction_id = response["interaction_id"]

        try:
            feedback = _payload(await server.call_tool("strata_feedback", {
                "interaction_id": interaction_id,
                "helpful": True,
                "chunk_feedback": {"s2": "useful"},
            }))
            assert feedback["status"] == "ok"

            pool = await get_pool(settings.database_url)
            rows = await pool.fetch(
                "SELECT signal_name, metadata FROM strata_learning.feedback_signals "
                "WHERE interaction_id = $1 AND signal_name LIKE 'chunk_%'",
                interaction_id,
            )
            rated = {json.loads(r["metadata"])["chunk_id"]: r["signal_name"] for r in rows}
            assert rated == {"chunk-1": "chunk_useful"}
        finally:
            await _delete_interaction(settings.database_url, interaction_id)

    async def test_feedback_on_passage_25(self, server, settings):
        from stratakb.core.database import get_pool

        settings.answer_token_budget = 10_000
        results = [_result(f"chunk-{i}", "docs") for i in range(25)]
        response = await _ask(server, results, layers=["docs"], top_k=25)
        interaction_id = response["interaction_id"]

        try:
            assert [p["id"] for p in response["passages"]][-1] == "S25"
            feedback = _payload(await server.call_tool("strata_feedback", {
                "interaction_id": interaction_id,
                "helpful": True,
                "chunk_feedback": {"S25": "useful"},
            }))
            assert feedback["status"] == "ok"

            pool = await get_pool(settings.database_url)
            rows = await pool.fetch(
                "SELECT signal_name, metadata FROM strata_learning.feedback_signals "
                "WHERE interaction_id = $1 AND signal_name LIKE 'chunk_%'",
                interaction_id,
            )
            rated = {json.loads(r["metadata"])["chunk_id"]: r["signal_name"] for r in rows}
            assert rated == {"chunk-24": "chunk_useful"}
        finally:
            await _delete_interaction(settings.database_url, interaction_id)

    async def test_feedback_stops_at_max_passages(self, server, settings):
        settings.answer_token_budget = 100_000
        results = [_result(f"chunk-{i}", "docs", content="x") for i in range(60)]
        response = await _ask(server, results, layers=["docs"], top_k=60)
        interaction_id = response["interaction_id"]

        try:
            assert len(response["passages"]) == MAX_PASSAGES
            last = _payload(await server.call_tool("strata_feedback", {
                "interaction_id": interaction_id,
                "helpful": True,
                "chunk_feedback": {f"S{MAX_PASSAGES}": "useful"},
            }))
            assert last["status"] == "ok"
            beyond = _payload(await server.call_tool("strata_feedback", {
                "interaction_id": interaction_id,
                "helpful": True,
                "chunk_feedback": {f"S{MAX_PASSAGES + 1}": "useful"},
            }))
            assert beyond["status"] == "error"
            assert f"S{MAX_PASSAGES + 1}" in beyond["error"]
        finally:
            await _delete_interaction(settings.database_url, interaction_id)

    async def test_high_confidence_survives_packing(self, server, settings):
        settings.answer_token_budget = 250
        results = [
            _result("c0", "docs", content="x" * 1000),
            _result("c1", "docs"),
            _result("c2", "docs"),
        ]

        with patch(
            "stratakb.learning.tracker.Tracker._record_safely", AsyncMock()
        ) as record:
            response = await _ask(server, results, layers=["docs"], response_format="detailed")

        assert len(response["passages"]) == 1
        assert response["confidence"]["level"] == "high"
        assert response["confidence"]["relevant_sources"] == 3
        assert response["answer_status"] == "found"
        assert record.await_args.kwargs["confidence_level"] == "high"

    async def test_status_judged_on_returned_passages(self, server, settings):
        settings.answer_token_budget = 250
        results = [
            _result("weak", "docs", similarity=0.3, content="x" * 1000),
            _result("strong", "docs", similarity=0.9),
        ]

        response = await _ask(server, results, layers=["docs"], response_format="detailed")

        assert [p["id"] for p in response["passages"]] == ["S1"]
        assert response["answer_status"] == "not_found"
        assert response["next_step"]
        assert response["confidence"]["top_score"] == 0.9
        await _delete_interaction(settings.database_url, response["interaction_id"])

    async def test_budget_counts_citations(self, server, settings):
        import pydantic_core

        results = [_result(f"c{i}", "docs", content=f"tiny {i}") for i in range(30)]

        response = await _ask(server, results, layers=["docs"], top_k=30)

        serialized = pydantic_core.to_json(response, indent=2).decode()
        # 100 tokens cover status, mode, next_step and interaction_id
        assert estimate_tokens(serialized) <= settings.answer_token_budget + 100
        assert 1 < len(response["passages"]) < 30
        await _delete_interaction(settings.database_url, response["interaction_id"])

    async def test_feedback_by_passage_id_unknown_interaction(self, server):
        feedback = _payload(await server.call_tool("strata_feedback", {
            "interaction_id": str(uuid4()),
            "helpful": True,
            "chunk_feedback": {"S1": "useful"},
        }))

        assert feedback["status"] == "error"
        assert "Unknown interaction_id" in feedback["error"]
