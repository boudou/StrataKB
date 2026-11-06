"""Clear an in-process cache when any process has changed the knowledge layers.

The query cache (rag/retriever.py) and the routing cache (knowledge/layers.py)
each own a StampWatcher. It compares the layers stamp of a database
(core.database.fetch_layers_stamp) with the last one it saw and calls the
owner's clear function when the stamp moved.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import structlog

from stratakb.core.database import Stamp, fetch_layers_stamp

logger = structlog.get_logger()

# The layers stamp is read at most this often (seconds) per database URL
_STAMP_INTERVAL = 2.0


class StampWatcher:
    """Per-database layers stamp, read at most every _STAMP_INTERVAL seconds.

    Logs failed_event when the stamp cannot be read and cleared_event when it
    clears the cache. Tests can replace clock and fetch on an owner's watcher.
    """

    def __init__(
        self,
        clear: Callable[[], None],
        *,
        cleared_event: str,
        failed_event: str,
        clock: Callable[[], float] = time.monotonic,
        fetch: Callable[[str], Awaitable[Stamp]] = fetch_layers_stamp,
    ) -> None:
        self.clear = clear
        self.cleared_event = cleared_event
        self.failed_event = failed_event
        self.clock = clock
        self.fetch = fetch
        self.stamps: dict[str, Stamp] = {}
        self.checked_at: dict[str, float] = {}

    def reset(self) -> None:
        """Forget every stamp: the next check reads again and only records."""
        self.stamps.clear()
        self.checked_at.clear()

    async def check(self, database_url: str) -> None:
        """Call clear() when the stamp differs from the one last seen.

        The first read only records the stamp.
        """
        now = self.clock()
        last = self.checked_at.get(database_url)
        if last is not None and now - last < _STAMP_INTERVAL:
            return
        # Set before the await: concurrent callers skip, a failing DB is retried every 2 s
        self.checked_at[database_url] = now
        try:
            stamp = await self.fetch(database_url)
        except Exception as e:
            logger.warning(self.failed_event, error_type=type(e).__name__)
            return
        previous = self.stamps.get(database_url)
        self.stamps[database_url] = stamp
        if previous is not None and stamp != previous:
            self.clear()
            logger.info(self.cleared_event)
