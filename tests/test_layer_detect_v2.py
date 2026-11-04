from stratakb.rag.layer_detect import compile_patterns, detect_layers_with_affinity

PATTERNS = compile_patterns({
    "sepa": [r"\bsepa\b"],
    "iso20022": [r"\biso\s*20022\b", r"\b(pain|pacs|camt)\.\d{3}\b"],
})


def test_blended_detection_boosts_keyword_layers():
    """Keyword matches should be boosted, but learned affinities should also contribute."""
    result = detect_layers_with_affinity(
        query="SEPA payment processing",
        patterns=PATTERNS,
        affinity_scores={"iso20022": 0.8, "runbooks": 0.6},
    )
    assert "sepa" in result
    assert "iso20022" in result
    assert result.index("sepa") < result.index("iso20022")


def test_affinity_only_when_no_keywords():
    result = detect_layers_with_affinity(
        query="how is this handled",
        patterns=PATTERNS,
        affinity_scores={"runbooks": 0.9, "iso20022": 0.4},
    )
    assert "runbooks" in result
    assert "iso20022" not in result  # 0.3 * 0.4 = 0.12, below 0.2 threshold


def test_no_affinity_falls_back_to_keywords():
    result = detect_layers_with_affinity(query="SEPA rejection code", patterns=PATTERNS)
    assert "sepa" in result


def test_no_matches_returns_empty():
    result = detect_layers_with_affinity(query="hello world", patterns=PATTERNS)
    assert result == []


def test_no_patterns_uses_affinity_only():
    result = detect_layers_with_affinity(
        query="SEPA rejection code", affinity_scores={"runbooks": 0.9}
    )
    assert result == ["runbooks"]
