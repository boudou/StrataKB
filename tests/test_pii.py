"""Tests for PII scrubbing before learning storage."""

from stratakb.learning.pii import scrub_query


def test_iban_scrubbed():
    """Greek IBAN should be replaced with placeholder."""
    result = scrub_query("Check account GR1601101250000000012300695")
    assert "GR1601101250000000012300695" not in result
    assert "[IBAN_REDACTED]" in result


def test_bic_scrubbed():
    """BIC codes should be replaced."""
    result = scrub_query("Send via ETHNGRAA")
    assert "ETHNGRAA" not in result
    assert "[BIC_REDACTED]" in result


def test_account_number_scrubbed():
    """Long numeric sequences (likely account numbers) should be scrubbed."""
    result = scrub_query("Account 12345678901234 balance")
    assert "12345678901234" not in result
    assert "[ACCT_REDACTED]" in result


def test_screen_codes_preserved():
    """Short numeric codes like 0401, 0498 should NOT be scrubbed."""
    result = scrub_query("What is the difference between 0401 and 0498?")
    assert "0401" in result
    assert "0498" in result


def test_no_pii_unchanged():
    """Queries without PII should pass through unchanged."""
    query = "How does SEPA Credit Transfer work?"
    assert scrub_query(query) == query
