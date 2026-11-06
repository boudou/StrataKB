"""Clear an in-process cache when any process has changed the knowledge layers.

The query cache (rag/retriever.py) and the routing cache (knowledge/layers.py)
each own a StampWatcher. It compares the layers stamp of a database
(core.database.fetch_layers_stamp) with the last one it saw and calls the
owner's clear function when the stamp moved.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

import structlog

logger = structlog.get_logger()

Stamp = tuple[int, datetime | None]

# The layers stamp is read at most this often (seconds) per database URL
_STAMP_INTERVAL = 2.0


class StampWatcher:
    """Per-database layers stamp, read at most every _STAMP_INTERVAL seconds.

    Logs "<event>_stamp_failed" when the stamp cannot be read and
    "<event>_cleared" when it clears the cache.
    """

    def __init__(self, event: str) -> None:
        self.event = event
        self.stamps: dict[str, Stamp] = {}
        self.checked_at: dict[str, float] = {}

    async def check(
        self,
        database_url: str,
        clear: Callable[[], None],
        *,
        now: float,
        fetch: Callable[[str], Awaitable[Stamp]],
    ) -> None:
        """Call clear() when the stamp differs from the one last seen.

        The first read only records the stamp. The owner passes its clock
        reading and stamp reader, so tests can patch them per module.
        """
        last = self.checked_at.get(database_url)
        if last is not None and now - last < _STAMP_INTERVAL:
            return
        # Set before the await: concurrent callers skip, a failing DB is retried every 2 s
        self.checked_at[database_url] = now
        try:
            stamp = await fetch(database_url)
        except Exception as e:
            logger.warning(f"{self.event}_stamp_failed", error_type=type(e).__name__)
            return
        previous = self.stamps.get(database_url)
        self.stamps[database_url] = stamp
        if previous is not None and stamp != previous:
            clear()
            logger.info(f"{self.event}_cleared")
