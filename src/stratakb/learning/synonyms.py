"""Synonym extraction from reformulation chains.

When a user searches for "AC01" then reformulates to
"incorrect account number", we learn that "AC01" maps to
"incorrect account number".
"""

from __future__ import annotations

import re


def extract_synonym_candidates(
    query_a: str,
    query_b: str,
) -> list[dict]:
    """Extract term→synonym pairs from a reformulation pair.

    Compares tokens in query_a vs query_b. Tokens only in query_a
    are candidate terms; tokens only in query_b are candidate synonyms.

    Returns:
        List of {term, synonym} dicts.
    """
    tokens_a = re.findall(r"\b[\w]+\b", query_a.lower())
    tokens_b = re.findall(r"\b[\w]+\b", query_b.lower())

    if tokens_a == tokens_b:
        return []

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    only_in_a = [t for t in tokens_a if t not in set_b]
    only_in_b = [t for t in tokens_b if t not in set_a]

    if not only_in_a or not only_in_b:
        return []

    candidates = []

    # Simple 1:1 mapping when both sides have single unique tokens
    if len(only_in_a) == 1 and len(only_in_b) == 1:
        candidates.append({
            "term": only_in_a[0],
            "synonym": only_in_b[0],
        })
    elif len(only_in_a) == 1 and len(only_in_b) > 1:
        # Single term maps to multi-word synonym
        # e.g., "AC01" -> "incorrect account number"
        synonym_phrase = " ".join(only_in_b)
        candidates.append({
            "term": only_in_a[0],
            "synonym": synonym_phrase,
        })
    else:
        # Multiple terms changed — try pairwise by position
        for term, synonym in zip(only_in_a, only_in_b):
            candidates.append({
                "term": term,
                "synonym": synonym,
            })

    return candidates
