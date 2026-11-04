"""Tests for implicit signal detection."""

import pytest
from stratakb.learning.signals import detect_reformulation, detect_layer_switch


def test_detect_reformulation_similar_queries():
    """Sequential similar queries with different wording = reformulation."""
    signals = detect_reformulation(
        current_query="iso20022 incorrect account number AC01",
        previous_query="what is AC01 in iso20022",
        previous_confidence="high",
        time_delta_seconds=30,
    )
    assert any(s["signal_name"] == "reformulation" for s in signals)


def test_no_reformulation_when_too_slow():
    """Queries >60s apart are not reformulations."""
    signals = detect_reformulation(
        current_query="iso20022 incorrect account number AC01",
        previous_query="what is AC01 in iso20022",
        previous_confidence="high",
        time_delta_seconds=120,
    )
    assert len(signals) == 0


def test_detect_layer_switch():
    """Explicit layer specification after auto-detect = layer switch signal."""
    signals = detect_layer_switch(
        current_layers_explicit=["runbooks"],
        previous_layers_auto=["iso20022"],
    )
    assert any(s["signal_name"] == "layer_switch" for s in signals)


def test_no_layer_switch_when_same():
    signals = detect_layer_switch(
        current_layers_explicit=["sepa"],
        previous_layers_auto=["sepa"],
    )
    assert len(signals) == 0
