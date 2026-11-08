"""Implicit feedback signal detection.

Analyzes sequential interactions within a session to detect
signals like query reformulations, layer switches, and follow-up drills.
"""

from __future__ import annotations

import re

import structlog

from stratakb.learning.synonyms import content_terms

logger = structlog.get_logger()

# Below this share of content terms in common, two questions are a change of topic
REFORMULATION_MIN_SIMILARITY = 0.3


def content_similarity(a: str, b: str) -> float:
    """Jaccard similarity of two questions' content terms; 0 if either has none."""
    terms_a = set(content_terms(a))
    terms_b = set(content_terms(b))
    if not terms_a or not terms_b:
        return 0.0
    return len(terms_a & terms_b) / len(terms_a | terms_b)


def detect_reformulation(
    current_query: str,
    previous_query: str,
    previous_confidence: str | None,
    time_delta_seconds: float,
) -> list[dict]:
    """Detect if current query is a reformulation of the previous one.

    Uses Jaccard similarity on content terms (fallback when embeddings
    unavailable), so shared function words alone never make a reformulation.
    """
    if time_delta_seconds > 60:
        return []

    similarity = content_similarity(current_query, previous_query)

    if (
        similarity >= REFORMULATION_MIN_SIMILARITY
        and current_query.lower() != previous_query.lower()
    ):
        signals = [
            {
                "signal_name": "reformulation",
                "signal_type": "implicit",
                "signal_value": -0.5,
                "metadata": {
                    "similarity": round(similarity, 3),
                    "time_delta_s": round(time_delta_seconds, 1),
                },
            }
        ]

        # Confidence mismatch: high confidence but still reformulated
        if previous_confidence == "high":
            signals.append({
                "signal_name": "confidence_mismatch",
                "signal_type": "implicit",
                "signal_value": -0.4,
                "metadata": {"expected": "high", "outcome": "reformulated"},
            })

        return signals

    return []


def detect_layer_switch(
    current_layers_explicit: list[str] | None,
    previous_layers_auto: list[str] | None,
) -> list[dict]:
    """Detect if user explicitly specified layers after auto-detection failed."""
    if not current_layers_explicit or not previous_layers_auto:
        return []

    # If user specified different layers than what was auto-detected
    if set(current_layers_explicit) != set(previous_layers_auto):
        return [
            {
                "signal_name": "layer_switch",
                "signal_type": "implicit",
                "signal_value": -0.3,
                "metadata": {
                    "auto_detected": previous_layers_auto,
                    "user_specified": current_layers_explicit,
                },
            }
        ]

    return []


def detect_follow_up(
    current_query: str,
    previous_results: list[dict] | None,
) -> list[dict]:
    """Detect if current query references entities from previous results."""
    if not previous_results:
        return []

    # Extract key terms from previous result content
    previous_terms = set()
    for r in previous_results[:5]:  # Check top 5 results
        content = r.get("content", "")
        # Extract capitalized terms (likely entity names)
        terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
        previous_terms.update(t.lower() for t in terms if len(t) > 3)

    if not previous_terms:
        return []

    query_lower = current_query.lower()
    referenced = [t for t in previous_terms if t in query_lower]

    if referenced:
        return [
            {
                "signal_name": "follow_up",
                "signal_type": "implicit",
                "signal_value": 0.5,
                "metadata": {"referenced_terms": referenced[:5]},
            }
        ]

    return []
