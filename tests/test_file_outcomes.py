"""Tests for FileOutcome, the counters that follow from it, and the outcome literals.

No database needed.
"""

from __future__ import annotations

import pytest


def test_file_outcome_rejects_values_outside_the_set():
    from stratakb.knowledge.ingestion import REASONS, FileOutcome

    for outcome, reason in [
        ("addded", None),
        ("failed", "read_eror"),
        ("added", "secret"),
        ("failed", None),
    ]:
        with pytest.raises(ValueError, match="Not an ingest outcome and reason"):
            FileOutcome("a.md", outcome, reason)
    for outcome, reasons in REASONS.items():
        stored = "added" if outcome == "flagged" else None
        for reason in reasons:
            assert FileOutcome("a.md", outcome, reason, stored=stored).reason == reason
    for outcome, reason, count in [
        ("added", None, 3),
        ("failed", "parse_error", 1),
        ("flagged", "truncated", 0),
    ]:
        with pytest.raises(ValueError, match="Not a count for this outcome"):
            FileOutcome("a.md", outcome, reason, count)
    assert FileOutcome("a.md", "flagged", "truncated", 5, stored="added").count == 5
    assert FileOutcome("a.md", "flagged", "probably_scanned", stored="added").count is None


def test_flagged_outcome_says_how_it_was_stored():
    from stratakb.knowledge.ingestion import FileOutcome

    for args, stored in [
        (("flagged", "truncated"), None),
        (("added", None), "added"),
        (("flagged", "truncated"), "removed"),
    ]:
        with pytest.raises(ValueError, match="Not a stored value for this outcome"):
            FileOutcome("a.md", *args, stored=stored)
    assert FileOutcome("a.md", "flagged", "truncated", stored="updated").stored == "updated"


def test_counters_follow_the_file_outcomes():
    from stratakb.knowledge.ingestion import FileOutcome, IngestionResult

    result = IngestionResult(
        layer_name="x",
        status="completed",
        files=[
            FileOutcome("added.md", "added"),
            FileOutcome("updated.md", "updated"),
            FileOutcome("unchanged.md", "unchanged"),
            FileOutcome("removed.md", "removed", "deleted"),
            FileOutcome("skipped.md", "skipped", "secret"),
            FileOutcome("failed.md", "failed", "read_error"),
            FileOutcome("flagged-new.csv", "flagged", "truncated", stored="added"),
            FileOutcome("flagged-old.csv", "flagged", "truncated", stored="updated"),
        ],
    )

    assert result.files_added == 2
    assert result.files_updated == 2
    assert result.files_unchanged == 1
    assert result.files_removed == 1
    assert result.files_skipped == 1
    assert result.files_flagged == 2
    assert result.docs_failed == 1
    assert result.docs_processed == 4


def test_counters_cannot_be_set():
    from stratakb.knowledge.ingestion import IngestionResult

    with pytest.raises(TypeError):
        IngestionResult(layer_name="x", status="completed", files_added=1)
    result = IngestionResult(layer_name="x", status="completed")
    with pytest.raises(AttributeError):
        result.files_added = 1


def test_reasons_cover_the_literals():
    from typing import get_args

    from stratakb.knowledge.ingestion import REASONS, Outcome, Reason

    assert set(REASONS) == set(get_args(Outcome))
    assert {r for rs in REASONS.values() for r in rs} - {None} == set(get_args(Reason))


def test_outcome_keys_follow_outcome():
    from typing import get_args

    from stratakb.knowledge.add_folder import OUTCOME_KEYS
    from stratakb.knowledge.ingestion import Outcome

    assert OUTCOME_KEYS == get_args(Outcome) == (
        "added", "updated", "unchanged", "removed", "skipped", "flagged", "failed",
    )


@pytest.mark.parametrize(
    ("suffix", "data", "expected"),
    [
        (".md", b"", True),
        (".pdf", b"", True),
        (".md", b"  \n\t", True),
        (".txt", b"\xef\xbb\xbf  ", True),
        (None, b"   ", True),
        (".md", b"x", False),
        (".pdf", b"   ", False),
        (".docx", b"\n", False),
    ],
)
def test_has_no_text(suffix, data, expected):
    from stratakb.knowledge.ingestion import _has_no_text
    from stratakb.knowledge.parsers.registry import FORMATS

    fmt = FORMATS[suffix] if suffix is not None else None
    assert _has_no_text(fmt, data) is expected
