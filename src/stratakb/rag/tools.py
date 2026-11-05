"""RAG (Retrieval-Augmented Generation) MCP tool definitions.

Registers knowledge search tools with the MCP server.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import get_current_user, require_role
from stratakb.core.logging import log_tool_call

if TYPE_CHECKING:
    from stratakb.rag.retriever import SearchResult


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


def _fuse_query_results(
    batch_results: list[list[SearchResult]], k: int
) -> list[tuple[SearchResult, float]]:
    """Merge the results of several searches by rank (Reciprocal Rank Fusion).

    Each search's own scores are normalized to its top result, so they do not
    compare across searches; positions do. A chunk earns 1 / (k + rank) from
    each search that finds it, so one found by several searches comes first.
    Ties go to the higher similarity, then to the first seen. Each chunk is
    represented by its most similar copy. The results are never modified (the
    search cache shares them); the fused score, best 1.0, travels alongside.
    """
    fused: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for results in batch_results:
        seen: set[str] = set()
        for rank, r in enumerate(results, start=1):
            key = str(hash(r.content[:200]))
            if key in seen:
                continue
            seen.add(key)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            current = best.get(key)
            if current is None or (r.similarity is not None and (
                current.similarity is None or r.similarity > current.similarity
            )):
                best[key] = r

    order = sorted(
        fused,
        key=lambda key: (
            -fused[key],
            best[key].similarity is None,
            -(best[key].similarity or 0.0),
        ),
    )
    if not order:
        return []
    top = fused[order[0]]
    return [(best[key], round(fused[key] / top, 4)) for key in order]


def _ask_chunk(r: SearchResult, score: float) -> dict[str, Any]:
    """The one shape of a strata_ask chunk; the reranker fills rerank_score."""
    return {
        "chunk_id": r.chunk_id,
        "content": r.content,
        "score": score,
        "similarity": r.similarity,
        "rerank_score": None,
        "layer": r.layer_name,
        "source_file": r.source_file,
        "section": r.source_section,
        "doc_type": r.doc_type,
        "metadata": r.metadata,
    }


def _basis_score(chunk: dict[str, Any], basis: str) -> float:
    """The score confidence judges a strata_ask chunk on, for this basis."""
    if basis == "rerank":
        return chunk["rerank_score"]
    return chunk["similarity"] if chunk["similarity"] is not None else 0.0


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
        from stratakb.rag.confidence import score_confidence
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

            fallback_reason = None
            if detected_layers:
                confidence = score_confidence(
                    [r.similarity for r in results if r.similarity is not None],
                    basis="similarity",
                )
                fallback_reason = _routing_fallback_reason(len(results), top_k, confidence)
                if fallback_reason:
                    results = await search_chunks(
                        database_url, openai_api_key, query,
                        layers=None, top_k=top_k, metadata_filter=metadata_filter,
                    )
                    layers = None

            user_id, _ = get_current_user()
            await write_audit_log(
                database_url,
                user_id=user_id,
                action="search",
                tool_name="strata_search",
                parameters={
                    "query": query[:200],
                    "layers": layers,
                    "top_k": top_k,
                    "routing_fallback": bool(fallback_reason),
                },
                result_summary=f"Found {len(results)} results",
            )

            response = {
                "status": "ok",
                "query": query,
                "results": [
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
                ],
                "total": len(results),
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
        which layers exist. The response carries an interaction_id to pass to
        strata_feedback.

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
        from stratakb.rag.confidence import score_confidence
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
            ) -> tuple[int, list[dict[str, Any]], str, dict[str, Any]]:
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
                candidates = [_ask_chunk(r, s) for r, s in fused][:rerank_limit]
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
                return len(candidates), chunks, basis, confidence

            candidate_count, chunks, basis, confidence = await _collect(layers)
            fallback_reason = None
            if detected:
                fallback_reason = _routing_fallback_reason(len(chunks), top_k, confidence)
                if fallback_reason:
                    candidate_count, chunks, basis, confidence = await _collect(None)
                    layers = None

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
                            "relevance_score": _basis_score(c, basis),
                        })

            # Audit
            user_id, _ = get_current_user()
            await write_audit_log(
                database_url,
                user_id=user_id,
                action="ask",
                tool_name="strata_ask",
                parameters={
                    "question": question[:200],
                    "layers": layers,
                    "queries_used": len(queries),
                    "routing_fallback": bool(fallback_reason),
                },
                result_summary=f"{candidate_count} chunks, confidence={confidence['level']}",
            )

            response: dict = {
                "status": "ok",
                "question": question,
                "queries_used": queries,
                "confidence": confidence,
                "retrieved_chunks": chunks,
                "total_chunks": len(chunks),
            }

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
    ) -> dict:
        """Ask a question of the knowledge base and get context with source citations.

        This is the primary knowledge tool. It rewrites the question into the
        knowledge base's terminology, runs several searches in parallel,
        deduplicates and reranks the results, and returns the retrieved
        context with source citations. The server does not write an answer;
        the client composes it from the returned context. The response carries
        an interaction_id to pass to strata_feedback.

        Args:
            question: Natural language question.
                      Examples: "How does a SEPA Credit Transfer get rejected?",
                      "Which fields does pacs.008 require for settlement?",
                      "Explain the pain.001 group header"
            layers: Optional list of layer names to search within.
                    Auto-detected from question if omitted.
            include_sources: Include source citations (default true).
            top_k: Max results per query (1-50, default 10).
        """
        if _tracker:
            return await _tracker.track_ask(_ask, question, layers, include_sources, top_k)
        return await _ask(question, layers, include_sources, top_k)
