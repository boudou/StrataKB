"""Ingest outcomes: what one run did with each file, and the run's result.

Every file an ingest run discovers or tracks gets one FileOutcome; REASONS
holds the outcome and reason pairs a FileOutcome may carry. IngestionResult
counts those outcomes for the tools and commands that report a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

from stratakb.knowledge.parsers.document import FlagReason

Outcome = Literal["added", "updated", "unchanged", "removed", "skipped", "flagged", "failed"]
SkippedReason = Literal["secret", "no_text"]
RemovedReason = Literal["deleted", "not_selected", SkippedReason]
FailedReason = Literal["read_error", "parse_error", "embed_error"]
Reason = Literal[RemovedReason, FlagReason, FailedReason]

# The reasons each outcome may carry; None: no reason
REASONS: dict[Outcome, frozenset[Reason | None]] = {
    "added": frozenset({None}),
    "updated": frozenset({None}),
    "unchanged": frozenset({None}),
    "removed": frozenset(get_args(RemovedReason)),
    "skipped": frozenset(get_args(SkippedReason)),
    "flagged": frozenset(get_args(FlagReason)),
    "failed": frozenset(get_args(FailedReason)),
}


@dataclass(frozen=True)
class FileOutcome:
    """What one ingest run did with one discovered or tracked file.

    outcome: "added", "updated", "unchanged", "removed", "skipped", "flagged"
    or "failed".
    reason: why a file was removed ("deleted", "not_selected" when a file list
    left out a file that still exists, "secret", "no_text"), skipped
    ("secret", "no_text"), flagged ("probably_scanned", "truncated": stored,
    and counted as added or updated) or failed ("read_error" when it could not
    be read, "parse_error", "embed_error"); None otherwise. REASONS holds the
    allowed pairs; any other pair raises ValueError.
    count: how many items the parser's limit left out of a flagged file (CSV
    rows, PDF outline entries, OpenAPI schema properties); None when not
    counted. A count on any other outcome, or below 1,
    raises ValueError.
    stored: how a flagged file was stored, "added" or "updated"; required on a
    flagged file, and on any other outcome a value raises ValueError.
    """

    relative_path: str
    outcome: Outcome
    reason: Reason | None = None
    count: int | None = None
    stored: Literal["added", "updated"] | None = None

    def __post_init__(self) -> None:
        if self.reason not in REASONS.get(self.outcome, frozenset()):
            raise ValueError(
                f"Not an ingest outcome and reason: {self.outcome!r}, {self.reason!r}"
            )
        if self.count is not None and (self.outcome != "flagged" or self.count < 1):
            raise ValueError(
                f"Not a count for this outcome: {self.outcome!r}, {self.count!r}"
            )
        if self.stored not in (("added", "updated") if self.outcome == "flagged" else (None,)):
            raise ValueError(
                f"Not a stored value for this outcome: {self.outcome!r}, {self.stored!r}"
            )


@dataclass
class IngestionResult:
    """Result of an ingestion run; files holds what the file counters count."""

    layer_name: str
    status: str  # "completed", "partial", "failed"
    chunks_created: int = 0
    # True when chunks from before per-file tracking were dropped and the layer rebuilt
    rebuilt_legacy: bool = False
    errors: list[dict] = field(default_factory=list)
    files: list[FileOutcome] = field(default_factory=list)
    duration_seconds: float = 0.0

    def _count(self, outcome: Outcome) -> int:
        return sum(1 for f in self.files if f.outcome == outcome)

    def _count_stored(self, stored: Literal["added", "updated"]) -> int:
        return sum(1 for f in self.files if f.outcome == stored or f.stored == stored)

    @property
    def docs_processed(self) -> int:
        """Files parsed and stored in this run (added + updated)."""
        return self.files_added + self.files_updated

    @property
    def docs_failed(self) -> int:
        return self._count("failed")

    @property
    def files_added(self) -> int:
        return self._count_stored("added")

    @property
    def files_updated(self) -> int:
        return self._count_stored("updated")

    @property
    def files_unchanged(self) -> int:
        return self._count("unchanged")

    @property
    def files_removed(self) -> int:
        return self._count("removed")

    @property
    def files_skipped(self) -> int:
        """Secret-looking names and files without text, not tracked."""
        return self._count("skipped")

    @property
    def files_flagged(self) -> int:
        """Files their parser flagged; also counted as added or updated."""
        return self._count("flagged")

    def flagged_rows(self) -> list[dict] | None:
        """The flagged list strata_ingest_documents and strata_apply_preset return."""
        return [
            {"file": f.relative_path, "reason": f.reason, "count": f.count}
            for f in self.files
            if f.outcome == "flagged"
        ] or None
