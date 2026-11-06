"""Natural language layer detection for search queries.

Detects which knowledge layers a query refers to, so users can write
"search the ISO 20022 docs for pacs.008" without explicitly specifying
layers=["iso20022"].

Each layer carries its own routing patterns (regular expressions stored in
strata_core.knowledge_layers.routing_patterns, usually set by a layer preset).
Nothing here knows about any particular domain.
"""

from __future__ import annotations

import re

import structlog

from stratakb.knowledge.layers import has_nested_repeat

logger = structlog.get_logger()

# Layer patterns (routing here, references in clause_lookup) read only the
# start of a question, so a pattern that backtracks badly cannot stall a
# search on a very long one. The full question still goes to the search.
MAX_PATTERN_QUESTION_CHARS = 2000


def pattern_text(question: str) -> str:
    """The part of question that layer patterns run over.

    At most MAX_PATTERN_QUESTION_CHARS characters; a word the cut falls
    inside is dropped, so a cut id ('AT-4' from 'AT-44') never matches.
    """
    if len(question) <= MAX_PATTERN_QUESTION_CHARS:
        return question
    head = question[:MAX_PATTERN_QUESTION_CHARS]
    if question[MAX_PATTERN_QUESTION_CHARS].isspace():
        return head
    match = re.search(r"\s\S*\Z", head)
    return head[: match.start()] if match else head


def compile_patterns(raw: dict[str, list[str]]) -> dict[str, list[re.Pattern[str]]]:
    """Compile routing patterns per layer, dropping any that are invalid regex
    or repeat a repeated part (one stored before such patterns were refused)."""
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for layer, patterns in raw.items():
        layer_patterns = []
        for pattern in patterns or []:
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning("invalid_routing_pattern", layer=layer, pattern=pattern, error=str(e))
                continue
            if has_nested_repeat(pattern):
                logger.warning("unsafe_routing_pattern", layer=layer, pattern=pattern)
                continue
            layer_patterns.append(compiled_pattern)
        if layer_patterns:
            compiled[layer] = layer_patterns
    return compiled


def detect_layers(
    query: str,
    patterns: dict[str, list[re.Pattern[str]]] | None = None,
) -> list[str]:
    """Detect knowledge layers mentioned in a natural language query.

    Returns a list of layer names detected from the query text.
    Returns empty list if no specific layer is detected (search all).
    Only the first MAX_PATTERN_QUESTION_CHARS characters are read.

    Args:
        query: The user's search query.
        patterns: Compiled routing patterns per layer (see compile_patterns).

    Returns:
        List of layer name strings (e.g., ["iso20022"], ["iso20022", "sepa"]).
    """
    if not query or not patterns:
        return []

    text = pattern_text(query)
    detected = []
    for layer, layer_patterns in patterns.items():
        if any(p.search(text) for p in layer_patterns):
            detected.append(layer)

    return detected


def extract_metadata_hints(query: str) -> dict[str, str]:
    """Extract metadata filter hints from the query.

    Looks for document type references ("openapi spec", "pdf", "csv").

    Returns:
        Dict of metadata key-value pairs to use as filters.
    """
    hints: dict[str, str] = {}
    query_lower = query.lower()

    if "yaml" in query_lower or "api spec" in query_lower or "openapi" in query_lower:
        hints["doc_type"] = "yaml"
    elif "pdf" in query_lower or "guide" in query_lower or "book" in query_lower:
        hints["doc_type"] = "pdf"
    elif "csv" in query_lower:
        hints["doc_type"] = "csv"

    return hints


def detect_layers_with_affinity(
    query: str,
    patterns: dict[str, list[re.Pattern[str]]] | None = None,
    affinity_scores: dict[str, float] | None = None,
    keyword_weight: float = 0.7,
    affinity_weight: float = 0.3,
    threshold: float = 0.2,
) -> list[str]:
    """Detect layers using routing patterns blended with learned affinities.

    Args:
        query: User's search query.
        patterns: Compiled routing patterns per layer.
        affinity_scores: Dict of {layer_name: affinity_score} from learning system.
        keyword_weight: Weight for routing pattern matches.
        affinity_weight: Weight for learned affinity scores.
        threshold: Minimum blended score to include a layer.

    Returns:
        List of layer names sorted by blended score (highest first).
    """
    keyword_layers = detect_layers(query, patterns)
    scores: dict[str, float] = {}

    for layer in keyword_layers:
        scores[layer] = keyword_weight

    if affinity_scores:
        for layer, affinity in affinity_scores.items():
            scores[layer] = scores.get(layer, 0) + affinity_weight * affinity

    return [
        layer for layer, score in sorted(scores.items(), key=lambda x: -x[1])
        if score >= threshold
    ]
