"""Tests for synonym extraction from reformulation chains."""

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
    """'kai' (Greek for 'and') = 'and' from reformulation."""
    candidates = extract_synonym_candidates(
        query_a="AC01 kai AM04",
        query_b="AC01 and AM04",
    )
    assert any(c["term"] == "kai" and c["synonym"] == "and" for c in candidates)
