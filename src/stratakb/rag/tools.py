"""RAG (Retrieval-Augmented Generation) MCP tool definitions.

Registers knowledge search tools with the MCP server.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, NamedTuple

import structlog
from mcp.server.fastmcp import FastMCP

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import get_current_user, require_role
from stratakb.core.logging import log_tool_call
from stratakb.learning.pii import scrub_query
from stratakb.rag.clause_lookup import in_exact_span, without_exact_spans
from stratakb.rag.clause_lookup import lookup as lookup_clauses
from stratakb.rag.confidence import answer_status, score_confidence, similarity_or_zero
from stratakb.rag.passages import (
    RESPONSE_FORMATS,
    ask_chunk,
    build_passages,
    next_step,
    pack_passages,
    shape_ask_response,
)

if TYPE_CHECKING:
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
    except Exception:
        pass

    affinity_scores = None
    if learning_enabled:
        try:
            from stratakb.learning.store import LearningStore
            affinities = await LearningStore(database_url).find_layer_affinities(query)
            if affinities:
                affinity_scores = {layer: 0.8 for layer in affinities}
        except Exception:
            pass

    return detect_layers_with_affinity(
        query, patterns=patterns, affinity_scores=affinity_scores
    )


def _routing_fallback_reason(
    result_count: int, top_k: int, confidence: dict[str, Any]
) -> str | None:
    """Decide whether an automatically routed search needs to broaden its scope."""
    if result_count < max(1, min(top_k, 50)):
        return "insufficient_results"
    if confidence["level"] == "low":
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
        seen: set[str] = set()
        for r in results:
            # Like fuse_rrf, only a chunk's first copy in each search counts
            if r.chunk_id in seen:
                continue
            seen.add(r.chunk_id)
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
    """One strata_ask retrieval: candidates found, chunks kept, their confidence."""

    candidate_count: int
    chunks: list[dict[str, Any]]
    confidence: dict[str, Any]


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


async def _cited_documents(
    database_url: str, chunks: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """The stored documents the chunks cite, read in one query.

    A chunk without offsets cites no document. When the read fails, the
    passages are cited by file name only.
    """
    ends: dict[int, int] = {}
    for c in chunks:
        if c.get("document_id") is not None and c.get("char_end") is not None:
            ends[c["document_id"]] = max(ends.get(c["document_id"], 0), c["char_end"])
    if not ends:
        return {}
    try:
        from stratakb.core.database import get_pool
        from stratakb.knowledge.documents import fetch_cited_documents

        pool = await get_pool(database_url)
        async with pool.acquire() as conn:
            return await fetch_cited_documents(conn, ends)
    except Exception as e:
        logger.warning("cited_documents_fetch_failed", error_type=type(e).__name__)
        return {}


def _search_item(chunk: dict[str, Any]) -> dict[str, Any]:
    """An exact clause chunk as a strata_search result, marked exact."""
    return {
        "chunk_id": None,
        "content": chunk["content"],
        "score": chunk["score"],
        "layer": chunk["layer"],
        "source_file": chunk["source_file"],
        "section": chunk["section"],
        "doc_type": chunk["doc_type"],
        "metadata": chunk["metadata"],
        "similarity": None,
        "match": "exact",
    }


def _basis_score(chunk: dict[str, Any], basis: str) -> float:
    """The score confidence judges a strata_ask chunk on, for this basis.

    An exact clause match is the passage asked for: it scores 1.0.
    """
    if chunk.get("match") == "exact":
        return 1.0
    if basis == "rerank":
        return chunk["rerank_score"]
    return similarity_or_zero(chunk["similarity"])


def _search_confidence(results: list[SearchResult]) -> dict[str, Any]:
    """Judge strata_search results on cosine similarity, never the fused score.

    The fused score is divided by the top score, so the best result is
    always 1.0. A result without an embedding counts as similarity 0.0.
    """
    return score_confidence(
        [similarity_or_zero(r.similarity) for r in results],
        basis="similarity",
    )


def register_rag_tools(
    mcp: FastMCP, database_url: str, openai_api_key: str = ""
) -> None:
    """Register RAG search tools on the MCP server."""

    # Initialize learning tracker
    _tracker = None
    try:
        from stratakb.config import get_settings
        _s = get_settings()
        if _s.learning_enabled:
            from stratakb.learning.store import LearningStore
            from stratakb.learning.tracker import Tracker
            _tracker = Tracker(LearningStore(database_url))
    except Exception:
        pass

    async def _search(
        query: str,
        layers: list[str] | None,
        top_k: int,
        metadata_filter: dict[str, str] | None,
    ) -> dict:
        """Run strata_search; the tool below documents the arguments."""
        from stratakb.config import get_settings
        from stratakb.knowledge.embeddings import embedding_key_error
        from stratakb.rag.retriever import search_chunks

        _settings = get_settings()
        if err := embedding_key_error(_settings, openai_api_key):
            return {"status": "error", "error": err}

        if not query or not query.strip():
            return {
                "status": "error",
                "error": "Query cannot be empty.",
            }

        try:
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
            )

            confidence = _search_confidence(results)
            fallback_reason = None
            if detected_layers:
                fallback_reason = _routing_fallback_reason(len(results), top_k, confidence)
                if fallback_reason:
                    results = await search_chunks(
                        database_url, openai_api_key, query,
                        layers=None, top_k=top_k, metadata_filter=metadata_filter,
                    )
                    layers = None
                    confidence = _search_confidence(results)

            # A clause or section the query names comes first, verbatim
            exact = await lookup_clauses(
                database_url, query, layers, _settings.answer_token_budget
            )
            found = [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "score": r.score,
                    "layer": r.layer_name,
                    "source_file": r.source_file,
                    "section": r.source_section,
                    "doc_type": r.doc_type,
                    "metadata": r.metadata,
                    "similarity": r.similarity,
                }
                for r in results
                if not in_exact_span(r.document_id, r.char_start, r.char_end, exact)
            ]
            items = [_search_item(e) for e in exact] + found
            items = items[: max(top_k, len(exact))]

            user_id, _ = get_current_user()
            await write_audit_log(
                database_url,
                user_id=user_id,
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

            response = {
                "status": "ok",
                "query": query,
                "confidence": confidence,
                "results": items,
                "total": len(items),
            }
            if detected_layers:
                response["auto_detected_layers"] = detected_layers
            if fallback_reason:
                response["routing_fallback"] = {
                    "reason": fallback_reason, "searched_layers": "all",
                }

            return response
        except Exception as e:
            return {"status": "error", "error": str(e)}

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
        marked "match": "exact".

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
            return await _tracker.track_search(_search, query, layers, top_k, metadata_filter)
        return await _search(query, layers, top_k, metadata_filter)

    async def _ask(
        question: str,
        layers: list[str] | None,
        include_sources: bool,
        top_k: int,
    ) -> dict:
        """Run strata_ask; the tool below documents the arguments."""
        from stratakb.config import get_settings
        from stratakb.rag.query_rewriter import (
            build_domain_context,
            rewrite_configured,
            rewrite_query,
        )
        from stratakb.rag.retriever import search_chunks

        _settings = get_settings()

        if not question or not question.strip():
            return {"status": "error", "error": "Question cannot be empty."}

        try:
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
                except Exception:
                    pass

            # Rewrite query into domain-specific variants
            domain_context = ""
            if rewrite_configured(_settings):
                try:
                    from stratakb.knowledge.layers import get_layer_routing
                    routing = await get_layer_routing(database_url)
                    domain_context = build_domain_context(routing["descriptions"])
                except Exception:
                    pass
            queries = await rewrite_query(
                question,
                anthropic_api_key=_settings.anthropic_api_key,
                learning_store=learning_store,
                domain_context=domain_context,
            )

            async def _collect(
                search_layers: list[str] | None,
            ) -> _Collected:
                async def _run_search(q: str) -> list[SearchResult]:
                    return await search_chunks(
                        database_url, openai_api_key, q,
                        layers=search_layers, top_k=top_k,
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
                    [_basis_score(c, basis) for c in chunks], basis=basis
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
            exact = await lookup_clauses(
                database_url, question, layers, _settings.answer_token_budget
            )
            # The best chunks that fit the answer budget; confidence is judged
            # on the reranked chunks, the answer status on what is returned
            chunks = pack_passages(
                exact + without_exact_spans(collected.chunks, exact),
                _settings.answer_token_budget,
            )
            confidence = collected.confidence
            basis = confidence["basis"]
            passages = build_passages(chunks, await _cited_documents(database_url, chunks))
            status = answer_status(
                score_confidence([_basis_score(c, basis) for c in chunks], basis=basis),
                len(passages),
            )

            sources = []
            if include_sources:
                seen_sources = set()
                for c in chunks:
                    source_key = f"{c['layer']}:{c['source_file']}:{c['section']}"
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        sources.append({
                            "layer": c["layer"],
                            "file": c["source_file"],
                            "section": c["section"],
                            "doc_type": c["doc_type"],
                            "relevance_score": _basis_score(c, confidence["basis"]),
                        })

            # Audit
            user_id, _ = get_current_user()
            await write_audit_log(
                database_url,
                user_id=user_id,
                action="ask",
                tool_name="strata_ask",
                parameters={
                    "question": _audit_text(question, _settings.audit_store_raw_queries),
                    "layers": layers,
                    "queries_used": len(queries),
                    "routing_fallback": bool(fallback_reason),
                    "exact_matches": len(exact),
                },
                result_summary=(
                    f"{collected.candidate_count} chunks, {len(passages)} passages, "
                    f"confidence={confidence['level']}"
                ),
            )

            response: dict = {
                "status": "ok",
                "question": question,
                "queries_used": queries,
                "confidence": confidence,
                "retrieved_chunks": chunks,
                "total_chunks": len(chunks),
                "answer_status": status,
                # Extractive answers (the document's own sentences) come later
                "mode": "grounded",
                "passages": passages,
            }
            if step := next_step(status):
                response["next_step"] = step

            if include_sources:
                response["sources"] = sources
            if detected:
                response["auto_detected_layers"] = detected
            if fallback_reason:
                response["routing_fallback"] = {
                    "reason": fallback_reason, "searched_layers": "all",
                }

            return response

        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_ask(
        question: str,
        layers: list[str] | None = None,
        include_sources: bool = True,
        top_k: int = 10,
        response_format: str = "concise",
    ) -> dict:
        """Ask a question of the knowledge base and get cited passages to answer from.

        This is the primary knowledge tool. It rewrites the question into the
        knowledge base's terminology, runs several searches in parallel,
        reranks the results, and returns the best passages that fit a token
        budget (ANSWER_TOKEN_BUDGET, about 1,500 tokens, each passage's id,
        ref and citation counted with its text; at most 50 passages). The
        server does not write an answer; the client composes it from the
        passages and cites them by id. A clause or section the question
        names ("clause 7.2", "Annex II") is looked up exactly and comes
        first, verbatim, once per document that has it.

        The response: status is found, partly_found or not_found, with a
        next_step for the last two; mode is grounded; passages lists S1, S2,
        ... in order of relevance, each with its exact text, a ref to the span
        of the stored document, and a citation (file path, page, lines, clause
        path, layer). interaction_id goes to strata_feedback, where passages
        can be rated by id ({"S1": "useful"}).

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
        """
        if response_format not in RESPONSE_FORMATS:
            return {
                "status": "error",
                "error": f"response_format must be one of {', '.join(RESPONSE_FORMATS)}.",
            }
        if _tracker:
            full = await _tracker.track_ask(_ask, question, layers, include_sources, top_k)
        else:
            full = await _ask(question, layers, include_sources, top_k)
        return shape_ask_response(full, response_format)
