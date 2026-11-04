"""Implicit feedback signal detection.

Analyzes sequential interactions within a session to detect
signals like query reformulations, layer switches, and follow-up drills.
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()


def detect_reformulation(
    current_query: str,
    previous_query: str,
    previous_confidence: str | None,
    time_delta_seconds: float,
) -> list[dict]:
    """Detect if current query is a reformulation of the previous one.

    Uses Jaccard token similarity (fallback when embeddings unavailable).
    """
    if time_delta_seconds > 60:
        return []

    # Jaccard similarity on lowercased tokens
    current_tokens = set(re.findall(r"\w+", current_query.lower()))
    previous_tokens = set(re.findall(r"\w+", previous_query.lower()))

    if not current_tokens or not previous_tokens:
        return []

    intersection = current_tokens & previous_tokens
    union = current_tokens | previous_tokens
    similarity = len(intersection) / len(union) if union else 0

    if similarity >= 0.25 and current_query.lower() != previous_query.lower():
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
