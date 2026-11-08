"""Tests for synonym extraction from reformulation chains."""

import pytest

from stratakb.learning.store import LearningStore
from stratakb.learning.synonyms import extract_synonym_candidates


def test_extract_from_reformulation_pair():
    """When user reformulates 'AC01' to 'Incorrect Account Number', extract synonym."""
    candidates = extract_synonym_candidates(
        query_a="what is AC01",
        query_b="what is Incorrect Account Number",
    )
    assert any(c["term"] == "ac01" and "incorrect" in c["synonym"].lower() for c in candidates)


def test_no_synonym_when_queries_identical():
    candidates = extract_synonym_candidates(
        query_a="SEPA rejection codes",
        query_b="SEPA rejection codes",
    )
    assert len(candidates) == 0


def test_extract_multilingual():
    """'kai' (Greek for 'and') = 'plus' from reformulation."""
    candidates = extract_synonym_candidates(
        query_a="AC01 kai AM04",
        query_b="AC01 plus AM04",
    )
    assert any(c["term"] == "kai" and c["synonym"] == "plus" for c in candidates)


def test_no_pairs_from_unrelated_questions():
    """The reported sourdough pair: no bake, bread or a."""
    candidates = extract_synonym_candidates(
        query_a="How do I bake sourdough bread?",
        query_b="How do I configure a Kubernetes ingress?",
    )
    assert candidates == []


@pytest.mark.parametrize(
    ("query_a", "query_b"),
    [
        ("why rotate keys", "what rotate keys"),
        ("is keys", "do keys"),
        ("is keys", "does keys"),
        ("of keys", "declare keys"),
        ("what is sourdough", "what is a"),
        ("rotate x keys", "rotate y keys"),
        ("rotate keys", "rotate a b"),
    ],
)
def test_stopwords_and_single_characters_are_never_mined(query_a, query_b):
    from stratakb.learning.synonyms import STOPWORDS

    for candidate in extract_synonym_candidates(query_a, query_b):
        for word in f"{candidate['term']} {candidate['synonym']}".split():
            assert word not in STOPWORDS
            assert len(word) > 1


def test_reported_stopwords_are_in_the_list():
    from stratakb.learning.synonyms import STOPWORDS

    assert {"is", "do", "does", "of", "why", "what", "a"} <= STOPWORDS


def test_redaction_placeholders_are_never_mined():
    assert extract_synonym_candidates("pay [IBAN_REDACTED] now", "pay refund now") == []
    assert extract_synonym_candidates("pay refund now", "pay [IBAN_REDACTED] now") == []
    assert extract_synonym_candidates("pay refund now", "pay [IBAN_REDACTED] wire now") == [
        {"term": "refund", "synonym": "wire"}
    ]


def test_no_positional_pairing():
    """Two changed words on each side give no way to tell what maps to what."""
    assert extract_synonym_candidates("rotate secret keys", "rotate api credentials") == []


def test_one_term_maps_to_all_the_new_words():
    assert extract_synonym_candidates(
        "AC01 reject reason", "incorrect account number reject reason"
    ) == [{"term": "ac01", "synonym": "incorrect account number"}]


def test_repeated_word_counts_once():
    assert extract_synonym_candidates("foo foo bar", "baz bar") == [
        {"term": "foo", "synonym": "baz"}
    ]


def test_apply_synonyms_adds_each_expansion_once():
    """find_synonyms ignores the layer, so one pair can come back per layer."""
    rows = [
        {"term": "q", "synonym": "x", "layer": "one", "confidence": 0.7},
        {"term": "q", "synonym": "x", "layer": "two", "confidence": 0.7},
    ]
    assert LearningStore.apply_synonyms("q", rows) == "q (x)"
