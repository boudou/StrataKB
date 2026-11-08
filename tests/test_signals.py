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


def test_unrelated_questions_are_not_a_reformulation():
    """Shared function words say nothing: the reported sourdough pair."""
    signals = detect_reformulation(
        current_query="How do I configure a Kubernetes ingress?",
        previous_query="How do I bake sourdough bread?",
        previous_confidence=None,
        time_delta_seconds=30,
    )
    assert signals == []


@pytest.mark.parametrize(
    "current",
    ["How do I rotate API keys?", "How do I rotate credentials?"],
)
def test_rewording_still_fires(current):
    previous = "How do I rotate keys?"
    signals = detect_reformulation(current, previous, None, 30)
    assert [s["signal_name"] for s in signals] == ["reformulation"]
    assert detect_reformulation(current, previous, None, 120) == []


def test_shared_function_words_do_not_fire():
    assert detect_reformulation("How is it", "What is it", None, 10) == []


def test_one_shared_word_below_the_bar():
    """One shared word of four is 0.25, under the bar."""
    signals = detect_reformulation(
        "How do I configure nginx?", "How do I configure Kubernetes ingress?", None, 10
    )
    assert signals == []


def test_only_stopwords_do_not_fire():
    assert detect_reformulation("What is it", "What is that", None, 10) == []


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
