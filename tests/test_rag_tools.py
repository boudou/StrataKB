"""Tests for RAG tools.

Tests cover:
- Layer detection (NLP)
- Metadata hint extraction
- Confidence scoring
- strata_ask tool integration
- MCP tool registration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.helpers import (
    ISO_PRESET,
    SEPA_PRESET,
    LogitModel,
    capture_tools,
    compatible_rewrite_settings,
    example_routing_patterns,
    openai_reply_client,
    rewrite_settings,
)

# ==========================================
# Layer detection tests
# ==========================================


@pytest.fixture(scope="module")
def example_patterns():
    """Routing patterns from the example SEPA and ISO 20022 presets."""
    return example_routing_patterns(SEPA_PRESET, ISO_PRESET)


class TestLayerDetection:
    def test_detect_sepa(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "sepa" in detect_layers("SEPA Credit Transfer rejection codes", example_patterns)

    def test_detect_pain_message(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "iso20022" in detect_layers("pain.001 message structure", example_patterns)

    def test_detect_camt_message(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "iso20022" in detect_layers("camt.053 bank statement", example_patterns)

    def test_detect_iso20022(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "iso20022" in detect_layers("ISO 20022 external code sets", example_patterns)

    def test_detect_multiple_layers(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        layers = detect_layers("SEPA rulebook rules for pacs.008", example_patterns)
        assert "sepa" in layers
        assert "iso20022" in layers

    def test_no_layer_detected(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert detect_layers("general payments question", example_patterns) == []

    def test_empty_query(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert detect_layers("", example_patterns) == []

    def test_no_patterns_detects_nothing(self):
        from stratakb.rag.layer_detect import detect_layers

        assert detect_layers("SEPA Credit Transfer") == []
        assert detect_layers("SEPA Credit Transfer", {}) == []

    def test_detect_sct_inst(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "sepa" in detect_layers("SCT Inst instant payment", example_patterns)

    def test_case_insensitive(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        assert "sepa" in detect_layers("sepa credit transfer", example_patterns)
        assert "iso20022" in detect_layers("Iso 20022 Messages", example_patterns)

    def test_word_boundaries(self, example_patterns):
        from stratakb.rag.layer_detect import detect_layers

        # "epc" must not match inside another word
        assert "sepa" not in detect_layers("receptacle sizing", example_patterns)

    def test_invalid_pattern_is_dropped(self):
        from stratakb.rag.layer_detect import compile_patterns, detect_layers

        patterns = compile_patterns({"runbooks": ["(unclosed", r"\brunbook\b"]})
        assert len(patterns["runbooks"]) == 1
        assert detect_layers("which runbook covers failover", patterns) == ["runbooks"]

    def test_nested_repeat_pattern_is_dropped(self):
        # A layer stored before such patterns were refused keeps its safe ones
        from stratakb.rag.layer_detect import compile_patterns, detect_layers

        patterns = compile_patterns({"runbooks": [r"(a+)+$", r"\brunbook\b"]})
        assert len(patterns["runbooks"]) == 1
        assert detect_layers("which runbook covers failover", patterns) == ["runbooks"]

    def test_routing_word_past_the_cap_is_not_found(self):
        from stratakb.rag.layer_detect import compile_patterns, detect_layers

        patterns = compile_patterns({"runbooks": [r"\brunbook\b"]})
        filler = "filler " * 400
        assert detect_layers(filler + "runbook", patterns) == []
        assert detect_layers("runbook " + filler, patterns) == ["runbooks"]

    def test_patterns_read_at_most_the_cap(self):
        from stratakb.rag.layer_detect import MAX_PATTERN_QUESTION_CHARS, detect_layers

        class Recorder:
            lengths: list[int] = []

            def search(self, text):
                self.lengths.append(len(text))
                return None

        recorder = Recorder()
        assert detect_layers("word " * 20_000, {"runbooks": [recorder]}) == []
        assert recorder.lengths and max(recorder.lengths) <= MAX_PATTERN_QUESTION_CHARS

    def test_pattern_text_drops_a_cut_word(self):
        from stratakb.rag.layer_detect import MAX_PATTERN_QUESTION_CHARS, pattern_text

        short = "runbook " * 10
        assert pattern_text(short) == short
        cut = "x" * (MAX_PATTERN_QUESTION_CHARS - 3) + " AT-44 more"
        assert pattern_text(cut) == "x" * (MAX_PATTERN_QUESTION_CHARS - 3)
        at_space = "x" * (MAX_PATTERN_QUESTION_CHARS - 6) + " AT-4 more"
        assert pattern_text(at_space).endswith(" AT-4")
        one_word = "x" * (MAX_PATTERN_QUESTION_CHARS + 50)
        assert pattern_text(one_word) == "x" * MAX_PATTERN_QUESTION_CHARS

    def test_layer_without_valid_patterns_is_omitted(self):
        from stratakb.rag.layer_detect import compile_patterns

        assert compile_patterns({"empty": [], "broken": ["("]}) == {}


# ==========================================
# Metadata hint extraction tests
# ==========================================


class TestMetadataHints:
    def test_message_identifiers_add_no_filter(self):
        from stratakb.rag.layer_detect import extract_metadata_hints

        # Chunks carry no message_type metadata, so a filter would match nothing
        assert extract_metadata_hints("pain.001 GroupHeader fields") == {}

    def test_yaml_doc_type(self):
        from stratakb.rag.layer_detect import extract_metadata_hints

        hints = extract_metadata_hints("show me the yaml API specs")
        assert hints.get("doc_type") == "yaml"

    def test_pdf_doc_type(self):
        from stratakb.rag.layer_detect import extract_metadata_hints

        hints = extract_metadata_hints("what does the guide say about accounting")
        assert hints.get("doc_type") == "pdf"

    def test_no_hints(self):
        from stratakb.rag.layer_detect import extract_metadata_hints

        hints = extract_metadata_hints("how does payment work")
        assert hints == {}


# ==========================================
# Confidence scoring tests
# ==========================================


class TestConfidenceScoring:
    def test_high_confidence(self):
        from stratakb.rag.confidence import score_confidence

        result = score_confidence([0.92, 0.85, 0.81, 0.75], basis="similarity")
        assert result.level == "high"
        assert result.top_score == 0.92
        assert result.relevant_sources >= 3

    def test_medium_confidence(self):
        from stratakb.rag.confidence import score_confidence

        result = score_confidence([0.72, 0.65], basis="similarity")
        assert result.level == "medium"

    def test_low_confidence(self):
        from stratakb.rag.confidence import score_confidence

        result = score_confidence([0.40, 0.30], basis="similarity")
        assert result.level == "low"
        assert result.basis == "similarity"
        assert result.warning

    def test_rerank_basis_uses_rerank_thresholds(self):
        from stratakb.rag.confidence import score_confidence

        scores = [0.7, 0.65, 0.6]
        assert score_confidence(scores, basis="rerank").level == "medium"
        assert score_confidence(scores, basis="rerank").basis == "rerank"
        assert score_confidence(scores, basis="similarity").level == "high"

    def test_unknown_basis_raises(self):
        from stratakb.rag.confidence import score_confidence

        with pytest.raises(ValueError):
            score_confidence([0.9], basis="fused")

    def test_empty_scores_report_basis(self):
        from stratakb.rag.confidence import score_confidence

        assert score_confidence([], basis="rerank").basis == "rerank"
        assert score_confidence([], basis="similarity").basis == "similarity"

    def test_no_results(self):
        from stratakb.rag.confidence import score_confidence

        result = score_confidence([], basis="similarity")
        assert result.level == "low"
        assert result.warning
        assert result.relevant_sources == 0

    def test_basis_is_required(self):
        from stratakb.rag.confidence import ConfidenceThresholds, score_confidence

        with pytest.raises(TypeError):
            score_confidence([0.9])
        with pytest.raises(TypeError):
            ConfidenceThresholds()

    def test_thresholds_cannot_be_passed(self):
        from stratakb.rag.confidence import score_confidence

        with pytest.raises(TypeError):
            score_confidence([0.9], basis="similarity", thresholds=None)

    def test_single_high_result(self):
        from stratakb.rag.confidence import score_confidence

        # Only 1 relevant source — not enough for "high" (needs 3+)
        result = score_confidence([0.95], basis="similarity")
        assert result.level == "medium"

    def test_warning_only_for_low(self):
        from stratakb.rag.confidence import score_confidence

        assert score_confidence([0.92, 0.85, 0.81], basis="similarity").warning is None
        medium = score_confidence([0.72, 0.65], basis="similarity")
        assert medium.level == "medium"
        assert medium.warning is None
        assert score_confidence([0.4], basis="similarity").warning.startswith(
            "Low confidence results"
        )
        assert score_confidence([], basis="similarity").warning.startswith(
            "No results found"
        )

    def test_thresholds_are_frozen(self):
        from dataclasses import FrozenInstanceError

        from stratakb.rag.confidence import SIMILARITY_THRESHOLDS

        with pytest.raises(FrozenInstanceError):
            SIMILARITY_THRESHOLDS.high_score = 0.1

    def test_similarity_or_zero(self):
        from stratakb.rag.confidence import similarity_or_zero

        assert similarity_or_zero(None) == 0.0
        assert similarity_or_zero(-0.2) == -0.2
        assert similarity_or_zero(0.7) == 0.7

    @pytest.mark.parametrize(
        ("top_score", "basis", "expected"),
        [
            (0.6, "similarity", "found"),
            # Rounded to 4 places, as score_confidence reports it
            (0.59996, "similarity", "found"),
            (0.5999, "similarity", "partly_found"),
            (0.45, "similarity", "partly_found"),
            (0.4499, "similarity", "not_found"),
            (0.0, "similarity", "not_found"),
            (0.8, "rerank", "found"),
            (0.79, "rerank", "partly_found"),
            (0.5, "rerank", "partly_found"),
            (0.49, "rerank", "not_found"),
            (0.0, "rerank", "not_found"),
        ],
    )
    def test_answer_status(self, top_score, basis, expected):
        from stratakb.rag.confidence import answer_status

        assert answer_status(top_score, basis) == expected


class TestSearchItem:
    _KEYS = [
        "chunk_id", "content", "score", "layer", "source_file", "section",
        "doc_type", "metadata", "similarity",
    ]

    def test_retrieved_chunk(self):
        from stratakb.rag.retriever import SearchResult
        from stratakb.rag.tools import _ask_chunk, _search_item

        r = SearchResult(
            chunk_id="c1", content="text", score=0.7, layer_name="docs",
            source_file="a.md", source_section="Intro", doc_type="markdown",
            metadata={"k": "v"}, similarity=0.6, document_id=3, char_start=0, char_end=4,
        )

        item = _search_item(_ask_chunk(r, r.score))

        assert list(item) == self._KEYS
        assert item == {
            "chunk_id": "c1", "content": "text", "score": 0.7, "layer": "docs",
            "source_file": "a.md", "section": "Intro", "doc_type": "markdown",
            "metadata": {"k": "v"}, "similarity": 0.6,
        }

    def test_exact_chunk_marked_last(self):
        from stratakb.rag.clause_lookup import _exact_chunk
        from stratakb.rag.tools import _search_item

        row = {
            "layer_name": "docs", "source_file": "a.md", "path": "7 > 7.2",
            "doc_type": "markdown", "document_id": 3, "char_start": 10,
        }

        item = _search_item(_exact_chunk(row, "7.2 clause text"))

        assert list(item) == [*self._KEYS, "match"]
        assert item["chunk_id"] is None
        assert item["similarity"] is None
        assert item["score"] == 1.0
        assert item["match"] == "exact"


def _grounded_layers():
    """Layer modes come from the table; these tests run without a database."""
    return patch("stratakb.knowledge.layers.get_answer_modes", AsyncMock(return_value=["grounded"]))


def _ask_tool(settings):
    """Register the RAG tools with settings and return strata_ask."""
    from stratakb.rag.tools import register_rag_tools

    with patch("stratakb.config.get_settings", return_value=settings):
        return capture_tools(register_rag_tools, "postgresql://test", "")["strata_ask"]


class TestAskDomainContext:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured", [True, False])
    async def test_ask_fetches_domain_context_only_when_rewriting(
        self, no_sdk_env, configured
    ):
        base = {"learning_enabled": False, "reranking_enabled": False}
        settings = (
            compatible_rewrite_settings(**base) if configured else rewrite_settings(**base)
        )
        ask = _ask_tool(settings)
        routing = AsyncMock(
            return_value={"descriptions": {"sepa": "SEPA Payments: rulebooks"}, "patterns": {}}
        )
        client = openai_reply_client('["sepa rejection", "pacs.002"]')
        with (
            patch("stratakb.config.get_settings", return_value=settings),
            patch("stratakb.knowledge.layers.get_layer_routing", routing),
            # The clause lookup reads the layer routing too: only the domain
            # context fetch is counted here
            patch("stratakb.knowledge.layers.get_structure_patterns", AsyncMock(return_value=[])),
            patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[])),
            patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
            _grounded_layers(),
            patch("openai.AsyncOpenAI", return_value=client) as openai_cls,
        ):
            result = await ask(
                question="Why was it rejected?", layers=["sepa"], response_format="detailed"
            )

        assert result["status"] == "ok"
        if configured:
            routing.assert_awaited_once()
            assert result["queries_used"] == ["sepa rejection", "pacs.002"]
            system = client.chat.completions.create.call_args.kwargs["messages"][0]
            assert "**sepa**: SEPA Payments: rulebooks" in system["content"]
        else:
            routing.assert_not_awaited()
            openai_cls.assert_not_called()
            assert result["queries_used"] == ["Why was it rejected?"]


def _search_results(scores, similarities):
    from stratakb.rag.retriever import SearchResult

    return [
        SearchResult(
            chunk_id=f"c{i}", content=f"Chunk {i} text", score=score, layer_name="docs",
            source_file=f"f{i}.md", source_section="", doc_type="", similarity=similarity,
        )
        for i, (score, similarity) in enumerate(zip(scores, similarities, strict=True))
    ]


class TestAskConfidence:
    @pytest.mark.asyncio
    async def test_hybrid_confidence_uses_similarity(self, no_sdk_env):
        """Without reranking, confidence reads the cosine similarity, not the fused score."""
        settings = rewrite_settings(learning_enabled=False, reranking_enabled=False)
        ask = _ask_tool(settings)
        results = _search_results([1.0, 0.95, 0.9], [0.2, 0.15, 0.1])
        with (
            patch("stratakb.config.get_settings", return_value=settings),
            patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=results)),
            patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
            _grounded_layers(),
        ):
            result = await ask(
                question="What does AC04 mean?", layers=["docs"], response_format="detailed"
            )

        assert result["status"] == "ok"
        confidence = result["confidence"]
        assert confidence["level"] == "low"
        assert confidence["top_score"] == 0.2
        assert confidence["basis"] == "similarity"
        assert [c["similarity"] for c in result["retrieved_chunks"]] == [0.2, 0.15, 0.1]

    @pytest.mark.asyncio
    async def test_rerank_confidence_uses_probabilities(self, no_sdk_env):
        """With reranking, confidence reads the cross-encoder's probabilities (0-1)."""
        settings = rewrite_settings(learning_enabled=False, reranking_enabled=True)
        ask = _ask_tool(settings)
        results = _search_results([1.0, 0.95, 0.9], [0.2, 0.15, 0.1])
        with (
            patch("stratakb.config.get_settings", return_value=settings),
            patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=results)),
            patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
            _grounded_layers(),
            patch(
                "stratakb.rag.reranker._get_reranker",
                return_value=LogitModel([1.0, 0.8, 0.7]),
            ),
        ):
            result = await ask(
                question="What does AC04 mean?", layers=["docs"], response_format="detailed"
            )

        assert result["status"] == "ok"
        confidence = result["confidence"]
        assert confidence["level"] == "medium"
        assert confidence["top_score"] == 0.7311
        assert confidence["basis"] == "rerank"
        assert all(0 < s["relevance_score"] < 1 for s in result["sources"])
        assert [c["similarity"] for c in result["retrieved_chunks"]] == [0.2, 0.15, 0.1]


_ASK_CHUNK_KEYS = [
    "chunk_id", "content", "score", "similarity", "rerank_score",
    "layer", "source_file", "section", "doc_type", "metadata",
    "document_id", "char_start", "char_end", "clause_path",
]


def _result(name, score, similarity, *, content=None, layer="docs"):
    from stratakb.rag.retriever import SearchResult

    return SearchResult(
        chunk_id=name, content=content or f"{name} text", score=score, layer_name=layer,
        source_file=f"{name}.md", source_section="", doc_type="", similarity=similarity,
    )


async def _ask_with(by_query, *, reranking=False, logits=None):
    """Run strata_ask with the rewriter giving these queries and each query these results."""
    settings = rewrite_settings(learning_enabled=False, reranking_enabled=reranking)
    ask = _ask_tool(settings)

    async def search(_db, _key, query, *, layers, top_k):
        return by_query[query]

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(return_value=list(by_query)),
        ),
        patch("stratakb.rag.retriever.search_chunks", AsyncMock(side_effect=search)),
        patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
        _grounded_layers(),
        patch("stratakb.rag.reranker._get_reranker", return_value=LogitModel(logits or [])),
    ):
        return await ask(question="q", layers=["docs"], response_format="detailed")


class TestAskOrdering:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("swapped", [False, True])
    async def test_ask_orders_top_chunks_of_two_queries_by_similarity(
        self, no_sdk_env, swapped
    ):
        by_query = {"one": [_result("a", 1.0, 0.5)], "two": [_result("b", 1.0, 0.8)]}
        if swapped:
            by_query = dict(reversed(list(by_query.items())))
        result = await _ask_with(by_query)

        assert [c["chunk_id"] for c in result["retrieved_chunks"]] == ["b", "a"]

    @pytest.mark.asyncio
    async def test_ask_ranks_chunk_found_by_several_queries_first(self, no_sdk_env):
        by_query = {
            "one": [_result("a", 1.0, 0.9), _result("both", 0.9, 0.4)],
            "two": [_result("b", 1.0, 0.9), _result("both", 0.9, 0.4)],
        }
        result = await _ask_with(by_query)

        chunks = result["retrieved_chunks"]
        assert chunks[0]["chunk_id"] == "both"
        assert chunks[0]["score"] == 1.0
        assert all(c["score"] < 1.0 for c in chunks[1:])

    @pytest.mark.asyncio
    async def test_ask_single_query_keeps_search_order(self, no_sdk_env):
        """A chunk kept for its exact terms keeps its place despite a lower similarity."""
        by_query = {
            "q": [_result("a", 1.0, 0.3), _result("b", 0.8, 0.9), _result("c", 0.5, 0.6)],
        }
        result = await _ask_with(by_query)

        assert [c["chunk_id"] for c in result["retrieved_chunks"]] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_ask_chunk_keys_match_with_and_without_reranking(self, no_sdk_env):
        by_query = {"q": [_result("a", 1.0, 0.5), _result("b", 0.8, 0.4)]}
        plain = await _ask_with(by_query)
        reranked = await _ask_with(by_query, reranking=True, logits=[0.5, 1.0])

        for chunk in plain["retrieved_chunks"] + reranked["retrieved_chunks"]:
            assert list(chunk) == _ASK_CHUNK_KEYS
        assert all(c["rerank_score"] is None for c in plain["retrieved_chunks"])
        assert all(0 < c["rerank_score"] < 1 for c in reranked["retrieved_chunks"])
        assert plain["retrieved_chunks"][0]["score"] == 1.0
        assert [c["chunk_id"] for c in reranked["retrieved_chunks"]] == ["b", "a"]

    @pytest.mark.asyncio
    async def test_ask_source_relevance_uses_confidence_basis(self, no_sdk_env):
        by_query = {"q": [_result("a", 1.0, 0.5), _result("b", 0.8, 0.4)]}
        for reranking, field in ((False, "similarity"), (True, "rerank_score")):
            result = await _ask_with(by_query, reranking=reranking, logits=[0.5, 1.0])

            chunks = {c["source_file"]: c for c in result["retrieved_chunks"]}
            assert result["sources"]
            for source in result["sources"]:
                assert source["relevance_score"] == chunks[source["file"]][field]

    @pytest.mark.asyncio
    async def test_ask_does_not_mutate_search_results(self, no_sdk_env):
        """The search cache shares its result objects, so the ask must not rescore them."""
        by_query = {
            "one": [_result("a", 0.7, 0.5), _result("b", 0.3, 0.4)],
            "two": [_result("b", 0.6, 0.6)],
        }
        before = {q: [r.score for r in rs] for q, rs in by_query.items()}
        await _ask_with(by_query)

        assert {q: [r.score for r in rs] for q, rs in by_query.items()} == before

    @pytest.mark.asyncio
    async def test_ask_keeps_chunks_sharing_a_prefix_in_one_query(self, no_sdk_env):
        prefix = "x" * 200
        by_query = {
            "q": [
                _result("a", 1.0, 0.5, content=prefix + " alpha"),
                _result("b", 0.8, 0.4, content=prefix + " beta"),
            ],
        }
        result = await _ask_with(by_query)

        assert {c["chunk_id"] for c in result["retrieved_chunks"]} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_ask_keeps_chunks_sharing_a_prefix_across_queries(self, no_sdk_env):
        prefix = "x" * 200
        by_query = {
            "one": [_result("a", 1.0, 0.5, content=prefix + " alpha")],
            "two": [_result("b", 1.0, 0.4, content=prefix + " beta")],
        }
        result = await _ask_with(by_query)

        assert {c["chunk_id"] for c in result["retrieved_chunks"]} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_ask_keeps_identical_text_from_two_layers(self, no_sdk_env):
        text = "A shared glossary entry that two layers both store."
        by_query = {
            "q": [
                _result("docs-copy", 1.0, 0.5, content=text, layer="docs"),
                _result("glossary-copy", 0.9, 0.5, content=text, layer="glossary"),
            ],
        }
        result = await _ask_with(by_query)

        ids = {c["chunk_id"] for c in result["retrieved_chunks"]}
        assert ids == {"docs-copy", "glossary-copy"}
        assert {s["layer"] for s in result["sources"]} == {"docs", "glossary"}

    @pytest.mark.asyncio
    async def test_ask_lists_chunk_found_by_two_queries_once(self, no_sdk_env):
        by_query = {"one": [_result("both", 1.0, 0.5)], "two": [_result("both", 1.0, 0.5)]}
        result = await _ask_with(by_query)

        assert [c["chunk_id"] for c in result["retrieved_chunks"]] == ["both"]
        assert result["retrieved_chunks"][0]["score"] == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("swapped", [False, True])
    async def test_ask_tie_counts_missing_similarity_as_zero(self, no_sdk_env, swapped):
        """A chunk without an embedding ties as similarity 0.0, below a real 0.3."""
        by_query = {"one": [_result("none", 1.0, None)], "two": [_result("real", 1.0, 0.3)]}
        if swapped:
            by_query = dict(reversed(list(by_query.items())))
        result = await _ask_with(by_query)

        assert [c["chunk_id"] for c in result["retrieved_chunks"]] == ["real", "none"]
        assert result["confidence"]["top_score"] == 0.3

    @pytest.mark.asyncio
    async def test_ask_tie_ranks_missing_similarity_above_negative(self, no_sdk_env):
        by_query = {"one": [_result("neg", 1.0, -0.2)], "two": [_result("none", 1.0, None)]}
        result = await _ask_with(by_query)

        assert [c["chunk_id"] for c in result["retrieved_chunks"]] == ["none", "neg"]


# ==========================================
# MCP tool registration tests
# ==========================================


class TestRagToolRegistration:
    def test_ask_tool_registered(self, server):
        tool_names = [t.name for t in server._tool_manager.list_tools()]
        assert "strata_ask" in tool_names

    def test_search_tool_registered(self, server):
        tool_names = [t.name for t in server._tool_manager.list_tools()]
        assert "strata_search" in tool_names


# ==========================================
# User management tool registration
# ==========================================


class TestUserToolRegistration:
    def test_user_tools_registered(self, server):
        tool_names = [t.name for t in server._tool_manager.list_tools()]
        assert "strata_whoami" in tool_names
        assert "strata_admin_list_users" in tool_names
        assert "strata_admin_update_role" in tool_names
