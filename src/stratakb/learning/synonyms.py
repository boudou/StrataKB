"""Synonym extraction from reformulation chains.

When a user searches for "AC01 reject reason" then reformulates to
"incorrect account number reject reason", we learn that "AC01" maps to
"incorrect account number".
"""

from __future__ import annotations

import re

# English function words. They say nothing about what a question is about,
# so they neither make two questions similar nor get mined as synonyms.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "didn", "do", "does", "doesn", "don", "for", "from", "had", "has", "have",
    "how", "i", "if", "in", "is", "isn", "it", "its", "me", "my", "no", "not", "of",
    "on", "or", "our", "should", "so", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your",
})


def content_terms(text: str) -> list[str]:
    """The words that carry a question's meaning, in order, without repeats.

    Leaves out stopwords, single characters and the placeholders from
    learning/pii.py, such as [IBAN_REDACTED].
    """
    tokens = re.findall(r"\w+", text.lower())
    return list(dict.fromkeys(
        t for t in tokens
        if len(t) >= 2 and t not in STOPWORDS and not t.endswith("_redacted")
    ))


def extract_synonym_candidates(
    query_a: str,
    query_b: str,
) -> list[dict]:
    """Extract a term→synonym pair from a reformulation pair.

    Compares content terms in query_a vs query_b. A pair needs exactly one
    term only in query_a: with more, nothing says which word maps to which.
    All the terms only in query_b form the synonym.

    Returns:
        List of {term, synonym} dicts, empty or with one entry.
    """
    terms_a = content_terms(query_a)
    terms_b = content_terms(query_b)

    only_in_a = [t for t in terms_a if t not in terms_b]
    only_in_b = [t for t in terms_b if t not in terms_a]

    if len(only_in_a) != 1 or not only_in_b:
        return []

    # e.g., "AC01" -> "incorrect account number"
    return [{"term": only_in_a[0], "synonym": " ".join(only_in_b)}]
