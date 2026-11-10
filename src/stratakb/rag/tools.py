"""RAG (Retrieval-Augmented Generation) MCP tool definitions.

Registers knowledge search tools with the MCP server.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, NamedTuple

import structlog
from mcp.server.fastmcp import FastMCP

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import require_role
from stratakb.core.logging import log_tool_call
from stratakb.learning.pii import scrub_query
from stratakb.rag.clause_lookup import (
    ClauseMatches,
    cap_note,
    in_exact_span,
    question_regexes,
    without_exact_spans,
)
from stratakb.rag.clause_lookup import lookup as lookup_clauses
from stratakb.rag.confidence import (
    EXTRACTIVE_SENTENCE_THRESHOLD,
    Confidence,
    answer_status,
    score_confidence,
    similarity_or_zero,
)
from stratakb.rag.extractive import (
    LEGACY_ONLY_NOTE,
    SENTENCE_REACH,
    answer_item,
    candidate_sentences,
    clause_gap_note,
    effective_mode,
    fit_to_answer_budget,
    select_sentences,
)
from stratakb.rag.names import Name, check_names, find_names, missing_names_note
from stratakb.rag.passages import (
    AnswerMode,
    AskResult,
    ExtractiveAnswer,
    ResponseFormat,
    SearchResponse,
    ToolFailure,
    ask_chunk,
    basis_score,
    build_passages,
    cut_clause_note,
    left_out_note,
    pack_passages,
    sources_from,
)
from stratakb.rag.receipts import issue_receipt

if TYPE_CHECKING:
    from stratakb.config import Settings
    from stratakb.rag.retriever import SearchResult

logger = structlog.get_logger()


async def _detect_layers(database_url: str, query: str, learning_enabled: bool) -> list[str]:
    """Pick layers for a query from each layer's routing patterns and learned affinities."""
    from stratakb.knowledge.layers import get_layer_routing
    from stratakb.rag.layer_detect import compile_patterns, detect_layers_with_affinity

    patterns = None
    try:
        routing = await get_layer_routing(database_url)
        patterns = compile_patterns(routing["patterns"])
    except Exception as e:
        logger.warning("routing_patterns_unavailable", error_type=type(e).__name__)

    affinity_scores = None
    if learning_enabled:
        try:
            from stratakb.learning.store import LearningStore
            affinities = await LearningStore(database_url).find_layer_affinities(query)
            if affinities:
                affinity_scores = {layer: 0.8 for layer in affinities}
        except Exception as e:
            logger.warning("layer_affinities_unavailable", error_type=type(e).__name__)

    return detect_layers_with_affinity(
        query, patterns=patterns, affinity_scores=affinity_scores
    )


async def _exact_clauses(
    database_url: str, question: str, layers: list[str] | None, budget: int
) -> ClauseMatches:
    """The sections question names, looked up with the searched layers' structure patterns."""
    from stratakb.knowledge.layers import get_structure_patterns

    try:
        by_layer = await get_structure_patterns(database_url, layers)
    except Exception as e:
        logger.warning("structure_patterns_unavailable", error_type=type(e).__name__)
        by_layer = {}
    return await lookup_clauses(
        database_url, question, layers, budget, question_regexes(by_layer)
    )


async def _answer_mode(
    database_url: str,
    requested: AnswerMode | None,
    chunks: list[dict[str, Any]],
    searched_layers: list[str] | None,
) -> tuple[str, str | None]:
    """The mode an answer runs in, judged on the passages it hands back, and any note.

    With no passages there is nothing to loosen: the mode asked for, else the
    strictest of the layers searched. Modes are read from the table on each call.
    """
    from stratakb.knowledge.layers import get_answer_modes

    if chunks:
        layers = sorted({c["layer"] for c in chunks})
        return effective_mode(requested, await get_answer_modes(database_url, layers))
    if requested:
        return requested, None
    return effective_mode(None, await get_answer_modes(database_url, searched_layers))


def _routing_fallback_reason(
    result_count: int, top_k: int, confidence: Confidence
) -> str | None:
    """Decide whether an automatically routed search needs to broaden its scope."""
    if result_count < max(1, min(top_k, 50)):
        return "insufficient_results"
    if confidence.level == "low":
        return "low_confidence"
    return None


def _audit_text(text: str, store_raw: bool) -> str:
    """The question or query as an audit entry keeps it: scrubbed, then cut to 200."""
    return (text if store_raw else scrub_query(text))[:200]


def _fuse_query_results(
    batch_results: list[list[SearchResult]], k: int
) -> list[tuple[SearchResult, float]]:
    """Merge the results of several searches by rank (Reciprocal Rank Fusion).

    Each search's own scores are normalized to its top result, so they do not
    compare across searches; positions do. The searches are fused with
    fuse_rrf, all weighted equally, so a chunk found by several searches
    comes first. Ties go to the higher similarity (a missing one counts as
    0.0), then to the first seen. Each chunk is represented by its most
    similar copy. Chunks are told apart by their chunk_id, so two different
    chunks with the same text both stay. The results are never modified (the
    search cache shares them); the fused score, best 1.0, travels alongside.
    """
    from stratakb.rag.retriever import fuse_rrf

    best: dict[str, SearchResult] = {}
    rank_lists: list[list[tuple[str, int]]] = []
    for results in batch_results:
        rank_lists.append([(r.chunk_id, rank) for rank, r in enumerate(results, start=1)])
        for r in results:
            current = best.get(r.chunk_id)
            if current is None or (
                similarity_or_zero(r.similarity) > similarity_or_zero(current.similarity)
            ):
                best[r.chunk_id] = r

    fused = sorted(
        fuse_rrf(rank_lists, k=k),
        key=lambda item: (-item[1], -similarity_or_zero(best[item[0]].similarity)),
    )
    if not fused:
        return []
    top = fused[0][1]
    return [(best[cid], round(score / top, 4)) for cid, score in fused]


class _Collected(NamedTuple):
    """One strata_ask retrieval: candidates found, chunks kept, their confidence.

    confidence.basis names the scale the chunks were judged on: similarity or rerank.
    """

    candidate_count: int
    chunks: list[dict[str, Any]]
    confidence: Confidence


def _ask_chunk(r: SearchResult, score: float) -> dict[str, Any]:
    """A retrieved chunk in the strata_ask chunk shape (passages.ask_chunk)."""
    return ask_chunk(
        chunk_id=r.chunk_id,
        content=r.content,
        score=score,
        similarity=r.similarity,
        layer=r.layer_name,
        source_file=r.source_file,
        section=r.source_section,
        doc_type=r.doc_type,
        metadata=r.metadata,
        document_id=r.document_id,
        char_start=r.char_start,
        char_end=r.char_end,
        clause_path=r.clause_path,
    )


def _name_texts(chunks: list[dict[str, Any]]) -> list[str]:
    """The text a name the question asks about is looked for in: each chunk and its file."""
    return [t for c in chunks for t in (c.get("content"), c.get("source_file")) if t]


def _join_notes(*notes: str | None) -> str | None:
    return " ".join(n for n in notes if n) or None


def _cited_ends(chunks: list[dict[str, Any]], reach: int = 0) -> dict[int, int]:
    """Each cited document's id and how far to read it: its furthest span's end, plus reach."""
    ends: dict[int, int] = {}
    for c in chunks:
        if c.get("document_id") is not None and c.get("char_end") is not None:
            ends[c["document_id"]] = max(ends.get(c["document_id"], 0), c["char_end"] + reach)
    return ends


async def _read_cited_documents(
    database_url: str, ends: dict[int, int]
) -> dict[int, dict[str, Any]]:
    """The stored documents named in ends, each read up to its end, in one query.

    The schema is brought up to date first, so a database the startup update
    missed is updated here, and one that cannot be updated raises the
    MigrationError that names the cause and the fix. Any failure is raised.
    """
    from stratakb.core.database import get_pool
    from stratakb.core.migrations import ensure_schema
    from stratakb.knowledge.documents import fetch_cited_documents

    await ensure_schema(database_url)
    pool = await get_pool(database_url)
    async with pool.acquire() as conn:
        return await fetch_cited_documents(conn, ends)


async def _cited_documents(
    database_url: str, chunks: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """The stored documents the chunks cite, read in one query.

    A chunk without offsets cites no document. When the read fails, the
    passages are cited by file name only.
    """
    ends = _cited_ends(chunks)
    if not ends:
        return {}
    try:
        return await _read_cited_documents(database_url, ends)
    except Exception as e:
        logger.warning("cited_documents_fetch_failed", error_type=type(e).__name__)
        return {}


async def _extract_answer(
    database_url: str,
    question: str,
    chunks: list[dict[str, Any]],
    model_name: str,
    budget: int,
    refused_note: str | None,
    omitted_note: str | None,
    *,
    names: list[Name],
    named_clause: bool,
) -> ExtractiveAnswer:
    """Choose the sentences that answer question from the chunks, and issue their receipt.

    The best that fit the token budget are kept. Below the threshold, or with
    nothing to quote, the answer is not_found and has no receipt. A named clause
    with a part left out makes a found answer partly_found, with a note. Unlike a
    grounded answer it fails when the documents cannot be read or the model
    cannot load: there is no answer without them. The documents are read by
    _read_cited_documents, so a database the startup update missed is updated
    here, and one that cannot be updated fails with the MigrationError that
    names the cause and the fix.
    refused_note says the caller asked for a mode the layers do not allow.
    omitted_note says named clauses never reached chunks: it makes a found answer
    partly_found too, and a not_found answer keeps it, so "not in the documents"
    does not hide what was not read.
    names are the names the question asks about (names.find_names). When none
    is in the chunks or the chosen sentences, and no named clause was returned,
    the answer is not_found, before any receipt; when some are missing, a found
    answer becomes partly_found. The note names the missing ones.
    """
    ends = _cited_ends(chunks, SENTENCE_REACH)
    documents = await _read_cited_documents(database_url, ends) if ends else {}
    # Splitting and scoring are CPU-bound: kept off the event loop
    candidates = await asyncio.to_thread(candidate_sentences, chunks, documents, ends)
    try:
        sentences = await asyncio.to_thread(
            select_sentences, question, candidates, EXTRACTIVE_SENTENCE_THRESHOLD, model_name
        )
    except Exception as e:
        logger.warning("extractive_scoring_failed", error_type=type(e).__name__)
        raise RuntimeError(
            f"Extractive mode needs the reranker model; see Troubleshooting. {e}"
        ) from e
    logger.info("extractive_answer", candidates=len(candidates), sentences=len(sentences))
    if not sentences:
        legacy_note = LEGACY_ONLY_NOTE if chunks and not ends else None
        names_note = missing_names_note(check_names(names, _name_texts(chunks)), related=False)
        note = _join_notes(refused_note, names_note, legacy_note, omitted_note)
        return ExtractiveAnswer([], None, "not_found", note)
    # A sentence can reach past its chunk, so its own text counts too
    check = check_names(names, [*_name_texts(chunks), *(s.text for s in sentences)])
    names_note = missing_names_note(check, related=False)
    if check.none_present and not named_clause:
        return ExtractiveAnswer(
            [], None, "not_found", _join_notes(refused_note, names_note, omitted_note)
        )

    sentences = fit_to_answer_budget(sentences, chunks, documents, budget)
    answer = [answer_item(s, chunks, documents) for s in sentences]
    receipt_id = await issue_receipt(
        database_url,
        kind="extractive",
        mode="extractive",
        question=question,
        quotes=[
            {
                "document_id": s.document_id,
                "char_start": s.start,
                "char_end": s.end,
                "clause_path": item["citation"]["clause_path"],
            }
            for s, item in zip(sentences, answer, strict=True)
        ],
        tool_name="strata_ask",
    )
    status = answer_status(max(s.score for s in sentences), "rerank")
    gap_note = await asyncio.to_thread(clause_gap_note, chunks, documents, ends, sentences)
    if (names_note or gap_note or omitted_note) and status == "found":
        status = "partly_found"
    note = _join_notes(refused_note, names_note, gap_note, omitted_note)
    return ExtractiveAnswer(answer, receipt_id, status, note)


def _search_item(chunk: dict[str, Any]) -> dict[str, Any]:
    """A chunk in the strata_ask shape as a strata_search result.

    An exact clause chunk keeps its "match": "exact" marker, and one cut to fit
    the budget its "cut" mark, last.
    """
    item = {
        "chunk_id": chunk["chunk_id"],
        "content": chunk["content"],
        "score": chunk["score"],
        "layer": chunk["layer"],
        "source_file": chunk["source_file"],
        "section": chunk["section"],
        "doc_type": chunk["doc_type"],
        "metadata": chunk["metadata"],
        "similarity": chunk["similarity"],
    }
    if "match" in chunk:
        item["match"] = chunk["match"]
    if chunk.get("cut"):
        item["cut"] = True
    return item


def _search_confidence(results: list[SearchResult]) -> Confidence:
    """Judge strata_search results on cosine similarity, never the fused score.

    The fused score is divided by the top score, so the best result is
    always 1.0. A result without an embedding counts as similarity 0.0.
    """
    return score_confidence(
        [similarity_or_zero(r.similarity) for r in results],
        basis="similarity",
    )


def register_rag_tools(
    mcp: FastMCP,
    database_url: str,
    openai_api_key: str = "",
    *,
    settings: Settings | None = None,
) -> None:
    """Register RAG search tools on the MCP server.

    `settings` are the server's: search, ask, rewriting, the query embedding,
    raw-query audit and the tracker all use them. Without them the tracker
    uses the environment's settings, read once here, and the tools read them
    on each call.
    """

    def _current_settings() -> Settings:
        """The server's settings; without any, the environment's, read per call."""
        if settings is not None:
            return settings
        from stratakb.config import get_settings

        return get_settings()

    # Initialize learning tracker
    _tracker = None
    try:
        tracker_settings = _current_settings()
        if tracker_settings.learning_enabled:
            from stratakb.learning.store import LearningStore
            from stratakb.learning.tracker import Tracker
            _tracker = Tracker(LearningStore(database_url), tracker_settings)
    except Exception as e:
        logger.warning("learning_tracker_unavailable", error_type=type(e).__name__)

    async def _search(
        query: str,
        layers: list[str] | None,
        top_k: int,
        metadata_filter: dict[str, str] | None,
    ) -> SearchResponse | ToolFailure:
        """Run strata_search; the tool below documents the arguments."""
        from stratakb.knowledge.embeddings import embedding_key_error
        from stratakb.rag.retriever import search_chunks

        _settings = _current_settings()
        if err := embedding_key_error(_settings, openai_api_key):
            return ToolFailure(err)

        if not query or not query.strip():
            return ToolFailure("Query cannot be empty.")

        try:
            # Read before detection fills layers in: a cap note's layer advice
            layer_named = bool(layers)
            # NLP layer detection — auto-detect layers if not explicitly specified
            # Layer detection with optional learned affinity blending
            detected_layers: list[str] = []
            if not layers:
                detected_layers = await _detect_layers(
                    database_url, query, _settings.learning_enabled
                )
                if detected_layers:
                    layers = detected_layers

            results = await search_chunks(
                database_url,
                openai_api_key,
                query,
                layers=layers,
                top_k=top_k,
                metadata_filter=metadata_filter,
                settings=_settings,
            )

            confidence = _search_confidence(results)
            fallback_reason = None
            if detected_layers:
                fallback_reason = _routing_fallback_reason(len(results), top_k, confidence)
                if fallback_reason:
                    results = await search_chunks(
                        database_url, openai_api_key, query,
                        layers=None, top_k=top_k, metadata_filter=metadata_filter,
                        settings=_settings,
                    )
                    layers = None
                    confidence = _search_confidence(results)

            # A clause or section the query names comes first, verbatim
            matches = await _exact_clauses(
                database_url, query, layers, _settings.answer_token_budget
            )
            exact = matches.chunks
            found = [
                _search_item(_ask_chunk(r, r.score))
                for r in results
                if not in_exact_span(r.document_id, r.char_start, r.char_end, exact)
            ]
            items = [_search_item(e) for e in exact] + found
            items = items[: max(top_k, len(exact))]

            await write_audit_log(
                database_url,
                action="search",
                tool_name="strata_search",
                parameters={
                    "query": _audit_text(query, _settings.audit_store_raw_queries),
                    "layers": layers,
                    "top_k": top_k,
                    "routing_fallback": bool(fallback_reason),
                    "exact_matches": len(exact),
                },
                result_summary=f"Found {len(items)} results",
            )

            return SearchResponse(
                query=query,
                confidence=confidence,
                results=items,
                auto_detected_layers=detected_layers,
                routing_fallback=fallback_reason,
                note=cap_note(matches, layer_named),
            )
        except Exception as e:
            return ToolFailure(str(e))

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_search(
        query: str,
        layers: list[str] | None = None,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None,
    ) -> dict:
        """Search the knowledge base for document chunks relevant to a query.

        Runs hybrid search (vector similarity plus full-text) across the
        ingested knowledge layers and returns the most relevant chunks with
        their source file and section. Use strata_list_knowledge_layers to see
        which layers exist. The response carries a confidence level (high,
        medium or low) judged on the results' similarity, with a warning when
        low, and an interaction_id to pass to strata_feedback. A clause or
        section the query names ("clause 7.2") comes first, verbatim,
        marked "match": "exact". A result cut to fit the token budget also has
        "cut": true: it holds only the start of the clause. When more sections
        match than the first 10 returned, or the query names more than 10
        references, a top-level note gives the count of matching sections not
        returned and names the references not looked up.

        Args:
            query: Natural language search query, or exact terms and identifiers.
                   Examples: "reject reason code AC01", "pacs.008 settlement fields",
                   "how to rotate the API signing key"
            layers: Optional list of layer names to search within.
                    Examples: ["iso20022"], ["sepa", "iso20022"].
                    If omitted, layers are picked from the query, else all are searched.
            top_k: Number of results to return (1-50, default 5).
            metadata_filter: Optional key-value pairs to filter results.
                           Examples: {"doc_type": "pdf"}, {"doc_type": "xsd"}
        """
        if _tracker:
            result = await _tracker.track_search(_search, query, layers, top_k, metadata_filter)
        else:
            result = await _search(query, layers, top_k, metadata_filter)
        return result.to_dict()

    async def _ask(
        question: str,
        layers: list[str] | None,
        include_sources: bool,
        top_k: int,
        mode: AnswerMode | None = None,
    ) -> AskResult | ToolFailure:
        """Run strata_ask; the tool below documents the arguments."""
        from stratakb.knowledge.embeddings import embedding_key_error
        from stratakb.rag.query_rewriter import (
            build_domain_context,
            rewrite_configured,
            rewrite_query,
        )
        from stratakb.rag.retriever import search_chunks

        _settings = _current_settings()
        # Before layer detection and the rewrite, so a question never leaves the
        # machine for a search that cannot run
        if err := embedding_key_error(_settings, openai_api_key):
            return ToolFailure(err)

        if not question or not question.strip():
            return ToolFailure("Question cannot be empty.")

        try:
            # Read before detection fills layers in: a cap note's layer advice
            layer_named = bool(layers)
            # Auto-detect layers if not specified
            detected: list[str] = []
            if not layers:
                detected = await _detect_layers(
                    database_url, question, _settings.learning_enabled
                )
                if detected:
                    layers = detected

            # Get learning store if available
            learning_store = None
            if _settings.learning_enabled:
                try:
                    from stratakb.learning.store import LearningStore
                    learning_store = LearningStore(database_url)
                except Exception as e:
                    logger.warning("learning_store_unavailable", error_type=type(e).__name__)

            # Rewrite query into domain-specific variants
            domain_context = ""
            if rewrite_configured(_settings):
                try:
                    from stratakb.knowledge.layers import get_layer_routing
                    routing = await get_layer_routing(database_url)
                    domain_context = build_domain_context(routing["descriptions"])
                except Exception as e:
                    logger.warning("domain_context_unavailable", error_type=type(e).__name__)
            queries = await rewrite_query(
                question,
                anthropic_api_key=_settings.anthropic_api_key,
                learning_store=learning_store,
                domain_context=domain_context,
                settings=_settings,
            )

            async def _collect(
                search_layers: list[str] | None,
            ) -> _Collected:
                async def _run_search(q: str) -> list[SearchResult]:
                    return await search_chunks(
                        database_url, openai_api_key, q,
                        layers=search_layers, top_k=top_k, settings=_settings,
                    )

                batch_results = await asyncio.gather(*(_run_search(q) for q in queries))
                fused = _fuse_query_results(list(batch_results), _settings.hybrid_rrf_k)

                rerank_limit = (
                    top_k * _settings.reranking_candidate_multiplier
                    if _settings.reranking_enabled else top_k
                )
                candidates = [_ask_chunk(r, s) for r, s in fused[:rerank_limit]]
                if _settings.reranking_enabled and candidates:
                    from stratakb.rag.reranker import rerank_results

                    chunks = await asyncio.to_thread(
                        rerank_results, question, candidates, top_k,
                        _settings.reranking_model,
                    )
                    # The reranker's relevance probabilities (0-1)
                    basis = "rerank"
                else:
                    # Cosine similarity, never a fused score: that one is
                    # normalized so the top result is always 1.0
                    chunks = candidates
                    basis = "similarity"
                confidence = score_confidence(
                    [basis_score(c, basis) for c in chunks], basis=basis
                )
                return _Collected(len(candidates), chunks, confidence)

            collected = await _collect(layers)
            fallback_reason = None
            if detected:
                fallback_reason = _routing_fallback_reason(
                    len(collected.chunks), top_k, collected.confidence
                )
                if fallback_reason:
                    collected = await _collect(None)
                    layers = None
            # A clause or section the question names comes first, verbatim,
            # looked up in the layers searched; retrieved chunks inside it go
            matches = await _exact_clauses(
                database_url, question, layers, _settings.answer_token_budget
            )
            exact = matches.chunks
            # The best chunks that fit the answer budget; confidence is judged
            # on the reranked chunks, the answer status on what is returned
            chunks = pack_passages(
                exact + without_exact_spans(collected.chunks, exact),
                _settings.answer_token_budget,
            )
            confidence = collected.confidence
            basis = confidence.basis
            # Read for every chunk packed and every named clause past them
            documents = await _cited_documents(database_url, chunks + exact[len(chunks):])
            # The citations are known now, replaced_by labels too: the passages as
            # the client gets them must fit, so the last that no longer fit are dropped
            chunks = pack_passages(
                chunks, _settings.answer_token_budget, documents=documents
            )
            # Packing keeps a prefix, so the named clauses past it did not fit
            left_out = exact[len(chunks):]
            passages = build_passages(chunks, documents)
            omitted = _join_notes(
                left_out_note(left_out, documents), cap_note(matches, layer_named)
            )
            status = answer_status(
                max((basis_score(c, basis) for c in chunks), default=0.0), basis
            )
            sources = sources_from(chunks, basis) if include_sources else None
            # The names the question asks about must be in what goes back; a
            # named clause returned means it named something that is there
            names = find_names(question)
            named_clause = len(exact) > len(left_out)

            # Judged on the passages handed back, not on the layers searched
            used_mode, refused_note = await _answer_mode(database_url, mode, chunks, layers)
            extraction = None
            note = None
            if used_mode == "extractive":
                extraction = await _extract_answer(
                    database_url,
                    question,
                    chunks,
                    _settings.reranking_model,
                    _settings.answer_token_budget,
                    refused_note,
                    omitted,
                    names=names,
                    named_clause=named_clause,
                )
                status = extraction.status
            else:
                clause_note = _join_notes(cut_clause_note(chunks, passages), omitted)
                if clause_note and status == "found":
                    status = "partly_found"
                check = check_names(names, _name_texts(chunks))
                if check.none_present and not named_clause:
                    status = "not_found"
                elif check.missing and status == "found":
                    status = "partly_found"
                names_note = missing_names_note(
                    check, related=bool(passages) and status == "not_found"
                )
                note = _join_notes(names_note, clause_note)

            # Audit
            await write_audit_log(
                database_url,
                action="ask",
                tool_name="strata_ask",
                parameters={
                    "question": _audit_text(question, _settings.audit_store_raw_queries),
                    "layers": layers,
                    "queries_used": len(queries),
                    "routing_fallback": bool(fallback_reason),
                    "exact_matches": len(exact),
                    "mode": used_mode,
                },
                result_summary=(
                    f"{collected.candidate_count} chunks, {len(passages)} passages, "
                    f"confidence={confidence.level}"
                ),
            )

            return AskResult(
                question=question,
                queries_used=queries,
                confidence=confidence,
                chunks=chunks,
                passages=passages,
                status=status,
                sources=sources,
                auto_detected_layers=detected,
                routing_fallback=fallback_reason,
                extraction=extraction,
                note=note,
            )

        except Exception as e:
            return ToolFailure(str(e))

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_ask(
        question: str,
        layers: list[str] | None = None,
        include_sources: bool = True,
        top_k: int = 10,
        response_format: ResponseFormat = "concise",
        mode: AnswerMode | None = None,
    ) -> dict:
        """Ask a question of the knowledge base and get cited passages to answer from.

        This is the primary knowledge tool. It rewrites the question into the
        knowledge base's terminology, runs several searches in parallel,
        reranks the results, and returns the best passages that fit a token
        budget (ANSWER_TOKEN_BUDGET, about 1,500 tokens, each passage's id,
        ref and citation, replaced_by label too, counted with its text; at
        most 50 passages). In grounded mode the client composes the answer from the passages and
        cites them by id. When a passage comes from an extractive layer the
        server answers with the document's own sentences. A clause or section
        the question names ("clause 7.2", "Annex II") is looked up exactly and
        comes first, verbatim, once per document that has it. A named clause over
        its share of the budget comes back cut: a found answer becomes
        partly_found, with a note naming the clause and the file. A named clause
        that does not fit is left out, and so is a matching section past the
        first 10: a found answer becomes partly_found, with a note naming the
        clauses left out and their files, and counting the sections not read.
        References past the first 10 are not looked up; the note names them.
        A name the question asks about (asyncio.TaskPool, PyConfig_Merge,
        AC04, PEP 8) must be in a passage returned: when none is, the status
        is not_found and the passages are only related text; when some are
        missing, a found answer becomes partly_found. The note names them.

        The response: status is found, partly_found or not_found, with a
        next_step for the last two; mode is the mode used; passages lists S1,
        S2, ... in order of relevance, each with its exact text, a ref to the
        span of the stored document, and a citation (file path, page, lines,
        clause path, layer, date and date_source; when another file says it
        replaces this one, replaced_by; when the layer's patterns mark the file
        as archived, archived). interaction_id goes to strata_feedback, where
        passages can be rated by id ({"S1": "useful"}).

        In extractive mode answer replaces passages: sentences word for word,
        each with its ref, citation and text_origin, plus a receipt_id (open it
        with strata_get_receipt). Give them as they are, do not rephrase them.
        not_found means no sentence was relevant enough or short enough to
        quote: do not guess.

        Args:
            question: Natural language question.
                      Examples: "How does a SEPA Credit Transfer get rejected?",
                      "Which fields does pacs.008 require for settlement?",
                      "Explain the pain.001 group header"
            layers: Optional list of layer names to search within.
                    Auto-detected from question if omitted.
            include_sources: With response_format "detailed", add a sources list
                    (default true).
            top_k: Advanced: max results per search (1-50, default 10); the
                   token budget then trims them.
            response_format: "concise" (default) or "detailed", which adds scores,
                    confidence, query variants, chunk ids and metadata.
            mode: "extractive" asks for the document's own sentences on any layer.
                  "grounded" cannot loosen a result holding a passage from an
                  extractive layer: the answer stays extractive and a note says so.
        """
        if _tracker:
            result = await _tracker.track_ask(
                _ask, question, layers, include_sources, top_k, mode
            )
        else:
            result = await _ask(question, layers, include_sources, top_k, mode)
        return result.to_dict(response_format)
