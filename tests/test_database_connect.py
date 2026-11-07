"""One connection with a deadline (connect_once) in core/database.py.

No database needed: asyncpg.connect is patched, or pointed at a local TCP
listener that never answers.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stratakb.core.database import connect_once

_FAKE_URL = "postgresql://strata:secret-pw@db.example:6543/strata"


def _fake_conn() -> MagicMock:
    conn = MagicMock()
    conn.close = AsyncMock()
    return conn


class TestConnectOnce:
    @pytest.mark.asyncio
    async def test_connect_once_closes_on_success(self):
        conn = _fake_conn()
        connect = AsyncMock(return_value=conn)
        with patch("stratakb.core.database.asyncpg.connect", connect):
            async with connect_once(_FAKE_URL, timeout=3) as got:
                assert got is conn
        connect.assert_awaited_once_with(_FAKE_URL, timeout=3)
        conn.close.assert_awaited_once_with(timeout=3)
        conn.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_once_terminates_on_error(self):
        conn = _fake_conn()
        with (
            patch("stratakb.core.database.asyncpg.connect", AsyncMock(return_value=conn)),
            pytest.raises(RuntimeError, match="boom"),
        ):
            async with connect_once(_FAKE_URL, timeout=3):
                raise RuntimeError("boom")
        conn.terminate.assert_called_once()
        conn.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connect_once_terminates_a_stalled_body(self):
        """A body cancelled by a deadline drops the socket, never a graceful close."""
        conn = _fake_conn()

        async def stalled() -> None:
            async with connect_once(_FAKE_URL, timeout=3):
                await asyncio.Event().wait()

        with (
            patch("stratakb.core.database.asyncpg.connect", AsyncMock(return_value=conn)),
            pytest.raises(TimeoutError),
        ):
            await asyncio.wait_for(stalled(), 0.05)
        conn.terminate.assert_called_once()
        conn.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connect_once_times_out_a_stalled_connect(self):
        """A server that accepts TCP and never answers fails within the timeout."""
        stop = asyncio.Event()

        async def silent(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await stop.wait()
            writer.close()

        server = await asyncio.start_server(silent, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            url = f"postgresql://strata:pw@127.0.0.1:{port}/strata"
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                async with connect_once(url, timeout=0.2):
                    pass
            assert time.monotonic() - started < 1
        finally:
            stop.set()
            server.close()
            await server.wait_closed()


class TestEachSiteConnectsWithADeadline:
    @pytest.mark.asyncio
    async def test_ensure_schema(self):
        from stratakb.core import migrations

        connect = AsyncMock(side_effect=OSError("refused"))
        with (
            patch("stratakb.core.database.asyncpg.connect", connect),
            pytest.raises(OSError),
        ):
            await migrations.ensure_schema("postgresql://u:p@127.0.0.1:1/never_checked_163")
        assert connect.await_args.kwargs["timeout"] == migrations._TIMEOUT_S

    @pytest.mark.asyncio
    async def test_wait_ready(self):
        from stratakb.core import local_db

        connect = AsyncMock(side_effect=OSError("refused"))
        with (
            patch("stratakb.core.database.asyncpg.connect", connect),
            patch.object(local_db, "_READY_TIMEOUT_S", 0),
            pytest.raises(local_db.LocalDbError, match="OSError"),
        ):
            await local_db._wait_ready("postgresql://u:p@127.0.0.1:1/strata")
        assert connect.await_args.kwargs["timeout"] == 5

    @pytest.mark.asyncio
    async def test_unsafe_role_reason(self):
        from stratakb.database_access import tools

        connect = AsyncMock(side_effect=OSError("refused"))
        with (
            patch("stratakb.core.database.asyncpg.connect", connect),
            pytest.raises(OSError),
        ):
            await tools._unsafe_role_reason("postgresql://u:p@127.0.0.1:1/strata")
        assert connect.await_args.kwargs["timeout"] == tools._CHECK_TIMEOUT
