"""PII scrubbing for query storage.

Redacts IBANs, BICs, and long account numbers from queries
before storing them in the learning tables. Short numeric codes
(like 4-6 digit screen or error codes) are preserved.
"""

from __future__ import annotations

import re

# IBAN: 2-letter country + 2 check digits + up to 30 alphanumeric
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")

# SWIFT/BIC: 8 or 11 character codes
_BIC_RE = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")

# Long numeric sequences (7+ digits = likely account numbers, NOT screen codes)
_ACCT_RE = re.compile(r"\b\d{10,34}\b")

# Known BIC-like banking terms to preserve (not actual BICs)
_BIC_WHITELIST = frozenset({
    "SEPAXXXX", "BISNXXXX", "SEPA", "BIAN", "SWIFT", "TARGET",
})


def scrub_query(query: str) -> str:
    """Remove PII (IBANs, BICs, account numbers) from a query string.

    Preserves short numeric codes (like 0401 or 40417) and known
    uppercase acronyms that look like BICs.
    """
    if not query:
        return query

    result = _IBAN_RE.sub("[IBAN_REDACTED]", query)

    def _replace_bic(match: re.Match) -> str:
        token = match.group(0)
        if token.upper() in _BIC_WHITELIST:
            return token
        return "[BIC_REDACTED]"

    result = _BIC_RE.sub(_replace_bic, result)

    # Scrub long account numbers (7+ digits; preserves 4-6 digit screen codes)
    def _replace_acct(match: re.Match) -> str:
        return "[ACCT_REDACTED]"

    result = _ACCT_RE.sub(_replace_acct, result)

    return result
