"""Tests for re-embedding stored chunks (require DB).

Tests cover:
- re_embed_layer: one layer at the column's size, batch by batch
- re_embed_all_layers: every layer at a new size, embedded before the lock
- Catch-up of chunks added meanwhile, the locked phase's own timeout
- Rollback and cleanup when something fails
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from stratakb.knowledge.vector_store import vector_column_size

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EMBED_TEXTS = "stratakb.knowledge.embeddings.embed_texts"
LAYERS = ("test_re_embed_a", "test_re_embed_b")
LATE = "test_re_embed_late"


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


def _fake_embed(value: float):
    """An embed_texts stub that returns vectors of `value` at the requested size."""
    return lambda texts, **kw: [[value] * kw["dimensions"] for _ in texts]


@contextlib.asynccontextmanager
async def _restoring_size(db_url, pool, size, layers):
    """On exit, re-embed every layer back at `size`, delete `layers` and check the size."""
    from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_all_layers

    try:
        yield
    finally:
        with patch(EMBED_TEXTS, new_callable=AsyncMock) as mock:
            mock.side_effect = _fake_embed(0.1)
            await re_embed_all_layers(db_url, new_dimensions=size)
        for layer in layers:
            await delete_layer_chunks(db_url, layer)
    assert await vector_column_size(pool) == size


async def _staging_tables(pool) -> int:
    """Temporary re-embed staging tables left in any session."""
    return await pool.fetchval(
        "SELECT count(*) FROM pg_class "
        "WHERE relname = 're_embed_staging' AND relpersistence = 't'"
    )


async def _insert_two_layers(db_url, layers, dims):
    from stratakb.knowledge.vector_store import ChunkRecord, delete_layer_chunks, insert_chunks

    for layer in layers:
        await delete_layer_chunks(db_url, layer)
        await insert_chunks(db_url, [
            ChunkRecord(
                content=f"{layer} chunk {n}", embedding=[0.1] * dims, layer_name=layer,
                metadata={"page": n}, source_file="doc.md", doc_type="md",
            )
            for n in (1, 2)
        ])


async def _chunk_locks(pool) -> int:
    """Granted locks on the chunks table that hold off ingests (and searches, when resizing)."""
    return await pool.fetchval(
        "SELECT count(*) FROM pg_locks "
        "WHERE relation = 'strata_vectors.chunks'::regclass AND granted "
        "AND mode IN ('AccessExclusiveLock', 'ShareRowExclusiveLock')"
    )


async def _insert_late(pool, layer, n, dims) -> None:
    """Insert n chunks whose fixed ids sort before any gen_random_uuid() id."""
    for i in range(1, n + 1):
        await pool.execute(
            "INSERT INTO strata_vectors.chunks "
            "(id, layer_name, content, embedding, metadata, source_file, doc_type) "
            "VALUES ($1::uuid, $2, 'late chunk', $3::vector, '{}'::jsonb, 'late.md', 'md')",
            f"00000000-0000-4000-8000-{i:012d}",
            layer,
            "[" + ",".join(["0.1"] * dims) + "]",
        )


async def _vectors(pool, layer) -> list[str]:
    rows = await pool.fetch(
        "SELECT embedding::text AS e FROM strata_vectors.chunks "
        "WHERE layer_name = $1 ORDER BY id",
        layer,
    )
    return [r["e"] for r in rows]


class TestReEmbed:
    @pytest.mark.asyncio
    async def test_re_embed_layer_rebuilds_the_prefix(self, db_url):
        """re_embed_layer embeds what ingestion embeds; old prefixed chunks are kept as stored."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown
        from stratakb.knowledge.vector_store import (
            ChunkRecord,
            chunk_embed_text,
            delete_layer_chunks,
            insert_chunks,
            re_embed_layer,
        )

        layer = "test_re_embed_prefix"
        pool = await get_pool(db_url)
        dims = await vector_column_size(pool)
        doc = parse_markdown(FIXTURES_DIR / "test_document.md")
        new_chunks = _chunk_document(doc, layer_name=layer, chunk_size=500, chunk_overlap=50)
        old_content = f"[{layer} > legacy > Page 1]\nOld chunk text"
        chunks = [*new_chunks, ChunkRecord(
            content=old_content, embedding=[], layer_name=layer,
            metadata={"page": 1}, source_file="legacy.pdf", doc_type="pdf",
        )]
        for c in chunks:
            c.embedding = [0.1] * dims

        try:
            await delete_layer_chunks(db_url, layer)
            await insert_chunks(db_url, chunks)
            with patch(EMBED_TEXTS, new_callable=AsyncMock) as mock:
                mock.side_effect = _fake_embed(0.2)
                stats = await re_embed_layer(db_url, layer, new_dimensions=dims)

            assert stats["chunks_updated"] == len(chunks)
            sent = [t for call in mock.call_args_list for t in call.args[0]]
            expected = [
                chunk_embed_text(c.layer_name, c.source_file, c.metadata, c.content)
                for c in new_chunks
            ]
            assert sorted(sent) == sorted([*expected, old_content])
        finally:
            await delete_layer_chunks(db_url, layer)

    @pytest.mark.asyncio
    async def test_re_embed_layer_refuses_another_size(self, db_url):
        """One layer cannot resize the column every layer shares."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_layer

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        try:
            await _insert_two_layers(db_url, LAYERS, size)
            before = {layer: await _vectors(pool, layer) for layer in LAYERS}
            with (
                patch(EMBED_TEXTS, new_callable=AsyncMock) as mock,
                pytest.raises(ValueError) as exc,
            ):
                await re_embed_layer(db_url, LAYERS[0], new_dimensions=size + 8)

            assert "every layer shares it" in str(exc.value)
            assert "re_embed_all_layers" in str(exc.value)
            mock.assert_not_awaited()
            assert await vector_column_size(pool) == size
            assert {layer: await _vectors(pool, layer) for layer in LAYERS} == before
        finally:
            for layer in LAYERS:
                await delete_layer_chunks(db_url, layer)

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_resizes_once(self, db_url):
        """All layers are re-embedded at the new size and stay searchable."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import re_embed_all_layers

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        new = size + 8
        async with _restoring_size(db_url, pool, size, LAYERS):
            await _insert_two_layers(db_url, LAYERS, size)
            with patch(EMBED_TEXTS, new_callable=AsyncMock) as mock:
                mock.side_effect = _fake_embed(0.2)
                stats = await re_embed_all_layers(db_url, new_dimensions=new)

            assert stats["resized"] is True
            assert stats["previous_dimensions"] == size
            assert stats["dimensions"] == new
            assert stats["layers"][LAYERS[0]] == 2
            assert stats["layers"][LAYERS[1]] == 2
            assert await vector_column_size(pool) == new
            dims = await pool.fetch(
                "SELECT DISTINCT vector_dims(embedding) AS d FROM strata_vectors.chunks "
                "WHERE layer_name = ANY($1::text[])",
                list(LAYERS),
            )
            assert [r["d"] for r in dims] == [new]
            found = await pool.fetch(
                "SELECT layer_name FROM strata_vectors.chunks "
                "ORDER BY embedding <=> $1::vector LIMIT 50",
                "[" + ",".join(["0.2"] * new) + "]",
            )
            assert set(LAYERS) <= {r["layer_name"] for r in found}
            assert await _staging_tables(pool) == 0

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_rolls_back(self, db_url):
        """A failure part-way leaves the column size and every vector as they were."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_all_layers

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        calls = 0

        def fail_second(texts, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("model went away")
            return [[0.3] * kwargs["dimensions"] for _ in texts]

        try:
            await _insert_two_layers(db_url, LAYERS, size)
            before = await _vectors(pool, LAYERS[0])
            with (
                patch(EMBED_TEXTS, new_callable=AsyncMock) as mock,
                pytest.raises(RuntimeError, match="model went away"),
            ):
                mock.side_effect = fail_second
                await re_embed_all_layers(db_url, batch_size=1, new_dimensions=size + 8)

            assert calls == 2
            assert await vector_column_size(pool) == size
            assert await _vectors(pool, LAYERS[0]) == before
            assert await _staging_tables(pool) == 0
        finally:
            for layer in LAYERS:
                await delete_layer_chunks(db_url, layer)

    @pytest.mark.asyncio
    async def test_re_embed_reads_one_batch_at_a_time(self, db_url):
        """Neither re-embed function holds more than one batch of chunk rows."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge import vector_store
        from stratakb.knowledge.vector_store import (
            ChunkRecord,
            delete_layer_chunks,
            insert_chunks,
            re_embed_all_layers,
            re_embed_layer,
        )

        layer = "test_re_embed_batches"
        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        fetched: list[int] = []
        original = vector_store._fetch_chunk_batch

        async def recording(*args, **kwargs):
            rows = await original(*args, **kwargs)
            fetched.append(len(rows))
            return rows

        try:
            await delete_layer_chunks(db_url, layer)
            await insert_chunks(db_url, [
                ChunkRecord(
                    content=f"chunk {n}", embedding=[0.1] * size, layer_name=layer,
                    metadata={"page": n}, source_file="doc.md", doc_type="md",
                )
                for n in range(5)
            ])
            with (
                patch(EMBED_TEXTS, new_callable=AsyncMock) as mock,
                patch.object(vector_store, "_fetch_chunk_batch", recording),
            ):
                mock.side_effect = _fake_embed(0.2)
                one = await re_embed_layer(db_url, layer, batch_size=2)
                assert fetched and max(fetched) <= 2
                assert sum(fetched) == 5
                fetched.clear()
                every = await re_embed_all_layers(db_url, batch_size=2, new_dimensions=size)

            assert one["chunks_updated"] == 5
            assert every["layers"][layer] == 5
            assert every["resized"] is False
            assert fetched and max(fetched) <= 2
            assert all(len(call.args[0]) <= 2 for call in mock.call_args_list)
        finally:
            await delete_layer_chunks(db_url, layer)

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_embeds_before_locking(self, db_url):
        """The model runs while no lock blocks searches or ingests on the chunks table."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import re_embed_all_layers

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        locks: list[int] = []

        async def embed(texts, **kw):
            locks.append(await _chunk_locks(pool))
            return _fake_embed(0.2)(texts, **kw)

        async with _restoring_size(db_url, pool, size, LAYERS):
            await _insert_two_layers(db_url, LAYERS, size)
            with patch(EMBED_TEXTS, new_callable=AsyncMock) as mock:
                mock.side_effect = embed
                stats = await re_embed_all_layers(db_url, batch_size=1, new_dimensions=size + 8)

            assert stats["resized"] is True
            assert locks and set(locks) == {0}

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_catches_up_new_chunks(self, db_url):
        """A chunk added while the model runs gets the new size; a deleted one is skipped."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_all_layers

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        new = size + 8
        calls = 0
        late_locks: list[int] = []

        async def embed(texts, **kw):
            nonlocal calls
            calls += 1
            if calls == 1:
                # Its id sorts first, so phase 1's cursor is already past it
                await _insert_late(pool, LATE, 1, size)
                await pool.execute(
                    "DELETE FROM strata_vectors.chunks WHERE id = "
                    "(SELECT id FROM strata_vectors.chunks WHERE layer_name = $1 "
                    "ORDER BY id DESC LIMIT 1)",
                    LAYERS[1],
                )
            if any("late chunk" in t for t in texts):
                late_locks.append(await _chunk_locks(pool))
            return _fake_embed(0.2)(texts, **kw)

        async with _restoring_size(db_url, pool, size, (*LAYERS, LATE)):
            await delete_layer_chunks(db_url, LATE)
            await _insert_two_layers(db_url, LAYERS, size)
            with patch(EMBED_TEXTS, new_callable=AsyncMock) as mock:
                mock.side_effect = embed
                stats = await re_embed_all_layers(db_url, batch_size=1, new_dimensions=new)

            assert stats["resized"] is True
            # The late chunk was embedded once, by the catch-up under the lock
            assert len(late_locks) == 1 and late_locks[0] > 0
            assert stats["layers"][LATE] == 1
            assert stats["layers"][LAYERS[0]] == 2
            assert stats["layers"][LAYERS[1]] == 1
            rows = await pool.fetch(
                "SELECT vector_dims(embedding) AS d FROM strata_vectors.chunks "
                "WHERE layer_name = ANY($1::text[])",
                [*LAYERS, LATE],
            )
            assert len(rows) == 4
            assert {r["d"] for r in rows} == {new}

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_stages_many_new_chunks_without_lock(self, db_url):
        """More new chunks than one batch are staged in an extra pass, not under the lock."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_all_layers

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        new = size + 8
        locks: list[int] = []

        async def embed(texts, **kw):
            if not locks:
                await _insert_late(pool, LATE, 3, size)
            locks.append(await _chunk_locks(pool))
            return _fake_embed(0.2)(texts, **kw)

        async with _restoring_size(db_url, pool, size, (*LAYERS, LATE)):
            await delete_layer_chunks(db_url, LATE)
            await _insert_two_layers(db_url, LAYERS, size)
            with patch(EMBED_TEXTS, new_callable=AsyncMock) as mock:
                mock.side_effect = embed
                stats = await re_embed_all_layers(db_url, batch_size=1, new_dimensions=new)

            assert stats["resized"] is True
            assert stats["layers"][LATE] == 3
            assert locks and set(locks) == {0}
            dims = await pool.fetch(
                "SELECT vector_dims(embedding) AS d FROM strata_vectors.chunks "
                "WHERE layer_name = $1",
                LATE,
            )
            assert [r["d"] for r in dims] == [new] * 3

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_locked_phase_has_own_timeout(self, db_url):
        """The locked phase outlasts the pool's command timeout, up to locked_timeout."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import re_embed_all_layers

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        short_pool = "stratakb.knowledge.vector_store.get_pool"

        async def block(blocker) -> None:
            # ACCESS SHARE lets the SHARE ROW EXCLUSIVE lock through but holds off the ALTER
            await blocker.execute("BEGIN")
            await blocker.execute("SELECT 1 FROM strata_vectors.chunks LIMIT 1")

        async def release_after_wait(blocker, hold: float) -> None:
            while not await pool.fetchval(
                "SELECT count(*) FROM pg_locks "
                "WHERE relation = 'strata_vectors.chunks'::regclass AND NOT granted"
            ):
                await asyncio.sleep(0.05)
            await asyncio.sleep(hold)
            await blocker.execute("ROLLBACK")

        async with _restoring_size(db_url, pool, size, LAYERS):
            short = await asyncpg.create_pool(db_url, min_size=1, max_size=2, command_timeout=0.5)
            blocker = await asyncpg.connect(db_url)
            try:
                await _insert_two_layers(db_url, LAYERS, size)
                with (
                    patch(EMBED_TEXTS, new_callable=AsyncMock) as mock,
                    patch(short_pool, AsyncMock(return_value=short)),
                ):
                    mock.side_effect = _fake_embed(0.2)
                    # (a) The ALTER waits 1.5 s, three times the pool's timeout, and still succeeds
                    await block(blocker)
                    release = asyncio.create_task(release_after_wait(blocker, 1.5))
                    stats = await re_embed_all_layers(db_url, new_dimensions=size + 8)
                    await asyncio.wait_for(release, 10)
                    assert stats["resized"] is True
                    assert await vector_column_size(pool) == size + 8

                    # (b) With a short locked_timeout the ALTER gives up and nothing changes
                    await block(blocker)
                    with pytest.raises(TimeoutError, match="locked_timeout"):
                        await re_embed_all_layers(db_url, new_dimensions=size, locked_timeout=0.5)
                    await blocker.execute("ROLLBACK")
                assert await vector_column_size(pool) == size + 8
                assert await _staging_tables(pool) == 0
            finally:
                if not blocker.is_closed():
                    with contextlib.suppress(Exception):
                        await blocker.execute("ROLLBACK")
                    await blocker.close()
                await short.close()

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_surfaces_error_inside_transaction(self, db_url):
        """An error in the locked phase surfaces as itself and rolls everything back."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_all_layers

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        try:
            await _insert_two_layers(db_url, LAYERS, size)
            before = await _vectors(pool, LAYERS[0])
            with (
                patch(EMBED_TEXTS, new_callable=AsyncMock) as mock,
                patch(
                    "stratakb.knowledge.vector_store._swap_staged",
                    AsyncMock(side_effect=RuntimeError("swap failed")),
                ),
                pytest.raises(RuntimeError, match="swap failed"),
            ):
                mock.side_effect = _fake_embed(0.2)
                await re_embed_all_layers(db_url, new_dimensions=size + 8)

            assert await vector_column_size(pool) == size
            assert await _vectors(pool, LAYERS[0]) == before
            assert await _staging_tables(pool) == 0
        finally:
            for layer in LAYERS:
                await delete_layer_chunks(db_url, layer)

    @pytest.mark.asyncio
    async def test_staging_table_keeps_the_first_error(self):
        """A cleanup DROP that fails does not replace the error already in flight."""
        from stratakb.knowledge.vector_store import _staging_table

        def make_conn():
            calls = 0

            async def execute(sql, *args, **kwargs):
                nonlocal calls
                calls += 1
                # Entry DROP and CREATE succeed; the exit DROP fails
                if calls == 3:
                    raise asyncpg.InterfaceError("connection is closed")

            conn = AsyncMock()
            conn.execute.side_effect = execute
            conn.is_closed = MagicMock(return_value=False)
            return conn

        with pytest.raises(RuntimeError, match="original"):
            async with _staging_table(make_conn()):
                raise RuntimeError("original")

        conn = make_conn()
        async with _staging_table(conn):
            pass
        assert conn.execute.await_count == 3

    @pytest.mark.asyncio
    async def test_re_embed_layer_stops_when_column_resized(self, db_url):
        """re_embed_layer stops when an all-layer resize changed the column under it."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import re_embed_layer

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)

        async def embed(texts, **kw):
            # What an all-layer resize does between this run's read and its write
            await pool.execute(
                "ALTER TABLE strata_vectors.chunks "
                f"ALTER COLUMN embedding TYPE vector({size + 8}) USING NULL"
            )
            return _fake_embed(0.2)(texts, **kw)

        async with _restoring_size(db_url, pool, size, LAYERS):
            await _insert_two_layers(db_url, LAYERS, size)
            with (
                patch(EMBED_TEXTS, new_callable=AsyncMock) as mock,
                pytest.raises(RuntimeError, match="resized"),
            ):
                mock.side_effect = embed
                await re_embed_layer(db_url, LAYERS[0], new_dimensions=size)
            assert await vector_column_size(pool) == size + 8

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_refuses_over_index_limit(self, db_url):
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import re_embed_all_layers

        pool = await get_pool(db_url)
        size = await vector_column_size(pool)
        with (
            patch(EMBED_TEXTS, new_callable=AsyncMock) as mock,
            pytest.raises(ValueError, match="2000"),
        ):
            await re_embed_all_layers(db_url, new_dimensions=2001)
        mock.assert_not_awaited()
        assert await vector_column_size(pool) == size
