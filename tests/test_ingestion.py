"""Tests for document ingestion pipeline.

Tests cover:
- PDF parsing with page-level tracking
- Text chunking with configurable sizes
- Vector store operations (insert, delete)
- Full ingestion pipeline orchestration
- Error handling for bad files
- MCP tool registration
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_PDF = FIXTURES_DIR / "test_doc.pdf"


# ==========================================
# PDF Parser tests
# ==========================================


class TestPDFParser:
    def test_parse_valid_pdf(self):
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        result = parse_pdf(TEST_PDF)
        assert result.error is None
        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert result.source_file == "test_doc.pdf"

    def test_page_level_tracking(self):
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        result = parse_pdf(TEST_PDF)
        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2
        assert "Customer Order" in result.pages[0].text
        assert "Operations" in result.pages[1].text

    def test_parse_nonexistent_file(self):
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        result = parse_pdf("/nonexistent/file.pdf")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_parse_non_pdf_file(self):
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        result = parse_pdf(__file__)  # This .py file
        assert result.error is not None
        assert "Not a PDF" in result.error


# ==========================================
# Chunking tests
# ==========================================


class TestChunking:
    def test_chunk_document(self):
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        doc = parse_pdf(TEST_PDF)
        chunks = _chunk_document(doc, layer_name="test", chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.layer_name == "test"
            assert chunk.doc_type == "pdf"
            assert chunk.source_file == "test_doc.pdf"
            assert "page" in chunk.metadata
            assert len(chunk.content) > 0

    def test_chunk_preserves_page_metadata(self):
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        doc = parse_pdf(TEST_PDF)
        chunks = _chunk_document(doc, layer_name="test", chunk_size=500, chunk_overlap=50)

        page_numbers = {c.metadata["page"] for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_chunk_respects_size(self):
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        doc = parse_pdf(TEST_PDF)
        chunks = _chunk_document(doc, layer_name="test", chunk_size=100, chunk_overlap=10)

        for chunk in chunks:
            # content carries no prefix, so it stays within the splitter's size
            assert len(chunk.content) <= 100


# ==========================================
# File discovery tests
# ==========================================


class TestFileDiscovery:
    def test_discover_single_file(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(TEST_PDF)
        assert len(files) == 1
        assert files[0] == TEST_PDF

    def test_discover_directory(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(FIXTURES_DIR)
        pdf_files = [f for f in files if f.suffix == ".pdf"]
        assert len(pdf_files) >= 1

    def test_discover_nonexistent(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(Path("/nonexistent"))
        assert len(files) == 0

    def test_discover_unsupported_file(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(Path(__file__))  # .py file
        assert len(files) == 0


# ==========================================
# Embedding client tests
# ==========================================


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_empty_texts_returns_empty(self):
        from stratakb.knowledge.embeddings import embed_texts

        result = await embed_texts([], api_key="fake")
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_openai_backend(self):
        from stratakb.knowledge.embeddings import embed_texts

        with pytest.raises(ValueError, match="API key"):
            await embed_texts(["hello"], api_key="", backend="openai")


# ==========================================
# Vector store tests (require DB)
# ==========================================


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_insert_and_delete_chunks(self, db_url):
        from stratakb.knowledge.vector_store import (
            ChunkRecord,
            delete_layer_chunks,
            insert_chunks,
        )

        layer = "test_vector_store"
        fake_embedding = [0.1] * 768

        chunks = [
            ChunkRecord(
                content="Test chunk content",
                embedding=fake_embedding,
                layer_name=layer,
                metadata={"page": 1},
                source_file="test.pdf",
                source_section="page_1",
                doc_type="pdf",
            ),
            ChunkRecord(
                content="Another test chunk",
                embedding=fake_embedding,
                layer_name=layer,
                metadata={"page": 2},
                source_file="test.pdf",
                source_section="page_2",
                doc_type="pdf",
            ),
        ]

        try:
            # Clean up first
            await delete_layer_chunks(db_url, layer)

            # Insert
            count = await insert_chunks(db_url, chunks)
            assert count == 2

            # Delete
            deleted = await delete_layer_chunks(db_url, layer)
            assert deleted == 2

            # Delete again — should be 0
            deleted = await delete_layer_chunks(db_url, layer)
            assert deleted == 0
        finally:
            await delete_layer_chunks(db_url, layer)

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
        dims = await pool.fetchval(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'strata_vectors.chunks'::regclass AND attname = 'embedding'"
        )
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
            with patch("stratakb.knowledge.embeddings.embed_texts", new_callable=AsyncMock) as mock:
                mock.side_effect = lambda texts, **kwargs: [[0.2] * dims for _ in texts]
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

    @staticmethod
    async def _column_size(pool) -> int:
        return await pool.fetchval(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'strata_vectors.chunks'::regclass AND attname = 'embedding'"
        )

    @staticmethod
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

    @staticmethod
    async def _vectors(pool, layer) -> list[str]:
        rows = await pool.fetch(
            "SELECT embedding::text AS e FROM strata_vectors.chunks "
            "WHERE layer_name = $1 ORDER BY id",
            layer,
        )
        return [r["e"] for r in rows]

    @pytest.mark.asyncio
    async def test_re_embed_layer_refuses_another_size(self, db_url):
        """One layer cannot resize the column every layer shares."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_layer

        layers = ("test_re_embed_a", "test_re_embed_b")
        pool = await get_pool(db_url)
        size = await self._column_size(pool)
        try:
            await self._insert_two_layers(db_url, layers, size)
            before = {layer: await self._vectors(pool, layer) for layer in layers}
            with (
                patch("stratakb.knowledge.embeddings.embed_texts", new_callable=AsyncMock) as mock,
                pytest.raises(ValueError) as exc,
            ):
                await re_embed_layer(db_url, layers[0], new_dimensions=size + 8)

            assert "every layer shares it" in str(exc.value)
            assert "re_embed_all_layers" in str(exc.value)
            mock.assert_not_awaited()
            assert await self._column_size(pool) == size
            assert {layer: await self._vectors(pool, layer) for layer in layers} == before
        finally:
            for layer in layers:
                await delete_layer_chunks(db_url, layer)

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_resizes_once(self, db_url):
        """All layers are re-embedded at the new size and stay searchable."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_all_layers

        layers = ("test_re_embed_a", "test_re_embed_b")
        pool = await get_pool(db_url)
        size = await self._column_size(pool)
        new = size + 8
        try:
            await self._insert_two_layers(db_url, layers, size)
            with patch("stratakb.knowledge.embeddings.embed_texts", new_callable=AsyncMock) as mock:
                mock.side_effect = lambda texts, **kw: [[0.2] * kw["dimensions"] for _ in texts]
                stats = await re_embed_all_layers(db_url, new_dimensions=new)

            assert stats["resized"] is True
            assert stats["previous_dimensions"] == size
            assert stats["dimensions"] == new
            assert stats["layers"][layers[0]] == 2
            assert stats["layers"][layers[1]] == 2
            assert await self._column_size(pool) == new
            dims = await pool.fetch(
                "SELECT DISTINCT vector_dims(embedding) AS d FROM strata_vectors.chunks "
                "WHERE layer_name = ANY($1::text[])",
                list(layers),
            )
            assert [r["d"] for r in dims] == [new]
            found = await pool.fetch(
                "SELECT layer_name FROM strata_vectors.chunks "
                "ORDER BY embedding <=> $1::vector LIMIT 50",
                "[" + ",".join(["0.2"] * new) + "]",
            )
            assert set(layers) <= {r["layer_name"] for r in found}
        finally:
            with patch("stratakb.knowledge.embeddings.embed_texts", new_callable=AsyncMock) as mock:
                mock.side_effect = lambda texts, **kw: [[0.1] * kw["dimensions"] for _ in texts]
                await re_embed_all_layers(db_url, new_dimensions=size)
            for layer in layers:
                await delete_layer_chunks(db_url, layer)
        assert await self._column_size(pool) == size

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_rolls_back(self, db_url):
        """A failure part-way leaves the column size and every vector as they were."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import delete_layer_chunks, re_embed_all_layers

        layers = ("test_re_embed_a", "test_re_embed_b")
        pool = await get_pool(db_url)
        size = await self._column_size(pool)
        calls = 0

        def fail_second(texts, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("model went away")
            return [[0.3] * kwargs["dimensions"] for _ in texts]

        try:
            await self._insert_two_layers(db_url, layers, size)
            before = await self._vectors(pool, layers[0])
            with (
                patch("stratakb.knowledge.embeddings.embed_texts", new_callable=AsyncMock) as mock,
                pytest.raises(RuntimeError, match="model went away"),
            ):
                mock.side_effect = fail_second
                await re_embed_all_layers(db_url, batch_size=1, new_dimensions=size + 8)

            assert calls == 2
            assert await self._column_size(pool) == size
            assert await self._vectors(pool, layers[0]) == before
        finally:
            for layer in layers:
                await delete_layer_chunks(db_url, layer)

    @pytest.mark.asyncio
    async def test_re_embed_all_layers_refuses_over_index_limit(self, db_url):
        from stratakb.core.database import get_pool
        from stratakb.knowledge.vector_store import re_embed_all_layers

        pool = await get_pool(db_url)
        size = await self._column_size(pool)
        with (
            patch("stratakb.knowledge.embeddings.embed_texts", new_callable=AsyncMock) as mock,
            pytest.raises(ValueError, match="2000"),
        ):
            await re_embed_all_layers(db_url, new_dimensions=2001)
        mock.assert_not_awaited()
        assert await self._column_size(pool) == size

    @pytest.mark.asyncio
    async def test_insert_empty_list(self, db_url):
        from stratakb.knowledge.vector_store import insert_chunks

        count = await insert_chunks(db_url, [])
        assert count == 0


# ==========================================
# Full pipeline integration test (mocked embeddings)
# ==========================================


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocked_embeddings(self, db_url):
        """Full ingestion: create layer → ingest PDF → verify chunks → cleanup."""
        from stratakb.knowledge.ingestion import ingest_documents
        from stratakb.knowledge.layers import create_layer, delete_layer
        from stratakb.knowledge.layers import LayerNotFoundError

        layer_name = "test_ingestion_pipeline"

        try:
            # Cleanup from previous runs
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

            # Create test layer
            await create_layer(
                db_url,
                name=layer_name,
                display_name="Test Ingestion",
                chunk_config={"chunk_size": 500, "chunk_overlap": 50},
            )

            fake_embedding = [0.01] * 768

            # Mock the embedding API call
            with patch(
                "stratakb.knowledge.ingestion.embed_texts",
                new_callable=AsyncMock,
            ) as mock_embed:
                # Return fake embeddings for however many chunks are created
                mock_embed.side_effect = lambda texts, **kwargs: [
                    fake_embedding for _ in texts
                ]

                result = await ingest_documents(
                    db_url,
                    "fake-api-key",
                    layer_name=layer_name,
                    source_path=str(TEST_PDF),
                )

            assert result.status in ("completed", "partial")
            assert result.docs_processed == 1
            assert result.docs_failed == 0
            assert result.chunks_created > 0
            # 2 pages averaging about 140 characters: stored, and flagged as probably scanned
            assert result.files_added == 1
            assert result.files_flagged == 1
            assert result.layer_name == layer_name

            # Verify layer stats were updated
            from stratakb.knowledge.layers import get_layer_status

            status = await get_layer_status(db_url, name=layer_name)
            assert status["doc_count"] == 1
            assert status["chunk_count"] == result.chunks_created
            assert status["last_ingested"] is not None

            # Verify ingestion run was recorded
            assert status["ingestion_run_count"] == 1

        finally:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_layer(self, db_url):
        """Ingesting into a non-existent layer should fail gracefully."""
        from stratakb.knowledge.ingestion import ingest_documents

        result = await ingest_documents(
            db_url,
            "fake-key",
            layer_name="nonexistent_layer_xyz",
            source_path=str(TEST_PDF),
        )
        assert result.status == "failed"
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_path(self, db_url):
        """Ingesting from a bad path should fail gracefully."""
        from stratakb.knowledge.ingestion import ingest_documents
        from stratakb.knowledge.layers import create_layer, delete_layer, LayerNotFoundError

        layer_name = "test_bad_path"
        try:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass
            await create_layer(db_url, name=layer_name, display_name="Test")

            result = await ingest_documents(
                db_url,
                "fake-key",
                layer_name=layer_name,
                source_path="/nonexistent/path",
            )
            assert result.status == "failed"
            assert len(result.errors) > 0
        finally:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_layer_chunk_config_is_used(self, db_url, tmp_path):
        """A layer's stored chunk_config sets the chunk size, not the defaults."""
        from stratakb.core.database import get_pool
        from stratakb.knowledge.ingestion import ingest_documents
        from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

        layer_name = "test_ingestion_chunk_config"
        source = tmp_path / "words.txt"
        source.write_text("word " * 200, encoding="utf-8")
        try:
            with contextlib.suppress(LayerNotFoundError):
                await delete_layer(db_url, name=layer_name)
            await create_layer(
                db_url,
                name=layer_name,
                display_name="Test Chunk Config",
                chunk_config={"chunk_size": 300, "chunk_overlap": 0},
            )

            with patch(
                "stratakb.knowledge.ingestion.embed_texts",
                new_callable=AsyncMock,
            ) as mock_embed:
                mock_embed.side_effect = lambda texts, **kwargs: [[0.01] * 768 for _ in texts]
                result = await ingest_documents(
                    db_url,
                    "fake-key",
                    layer_name=layer_name,
                    source_path=str(source),
                )

            assert result.docs_processed == 1
            pool = await get_pool(db_url)
            rows = await pool.fetch(
                "SELECT content FROM strata_vectors.chunks WHERE layer_name = $1",
                layer_name,
            )
            lengths = [len(r["content"]) for r in rows]
            assert len(lengths) > 1
            assert max(lengths) <= 300
        finally:
            with contextlib.suppress(LayerNotFoundError):
                await delete_layer(db_url, name=layer_name)


# ==========================================
# Contextual chunk enrichment tests
# ==========================================


class TestContextualChunkEnrichment:
    def test_context_prefix_generation(self):
        """Chunks should have a context prefix with layer, file, and section."""
        from stratakb.knowledge.vector_store import _build_context_prefix

        prefix = _build_context_prefix(
            layer_name="sepa",
            source_file="SCT_Reason_Codes_v6.0.pdf",
            page_number=12,
        )
        assert "[sepa" in prefix
        assert "SCT_Reason_Codes" in prefix
        assert "12" in prefix

    def test_context_prefix_prepended_to_content(self):
        from stratakb.knowledge.vector_store import _build_context_prefix

        prefix = _build_context_prefix("iso20022", "some_api.yaml", section="Payment Initiation")
        assert "Payment Initiation" in prefix

    def test_chunk_content_has_no_prefix(self):
        """Stored content is the page's own text; only the embedded text carries the prefix."""
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf
        from stratakb.knowledge.vector_store import chunk_embed_text

        doc = parse_pdf(TEST_PDF)
        chunks = _chunk_document(doc, layer_name="sepa", chunk_size=500, chunk_overlap=50)
        pages = {p.page_number: p.text for p in doc.pages}

        assert len(chunks) > 0
        for chunk in chunks:
            assert not chunk.content.startswith("["), chunk.content[:60]
            assert chunk.content in pages[chunk.metadata["page"]]
            embedded = chunk_embed_text(
                chunk.layer_name, chunk.source_file, chunk.metadata, chunk.content
            )
            assert embedded.startswith("[sepa > test_doc > Page ")
            assert embedded.endswith(chunk.content)

    def test_markdown_section_path_in_metadata_and_prefix(self):
        """A Markdown section's heading chain is in the chunk metadata and the embedded prefix."""
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.markdown_parser import parse_markdown
        from stratakb.knowledge.vector_store import chunk_embed_text

        doc = parse_markdown(FIXTURES_DIR / "test_document.md")
        chunks = _chunk_document(doc, layer_name="docs", chunk_size=500, chunk_overlap=50)

        kc = [c for c in chunks if c.content.startswith("### Key Capabilities")]
        assert len(kc) == 1
        path = ["Customer Order Service", "Overview", "Key Capabilities"]
        assert kc[0].metadata["section_path"] == path
        c = kc[0]
        embedded = chunk_embed_text(c.layer_name, c.source_file, c.metadata, c.content)
        assert embedded.startswith(
            "[docs > test_document > Customer Order Service > Overview > Key Capabilities]\n"
        )
        assert "## Overview" not in kc[0].content

    @pytest.mark.asyncio
    async def test_embed_chunks_sends_prefixed_text(self):
        """The embedder gets the prefix plus content; the stored content stays as it is."""
        from stratakb.knowledge.ingestion import _embed_chunks
        from stratakb.knowledge.vector_store import ChunkRecord

        chunks = [
            ChunkRecord(
                content="body", embedding=[], layer_name="l",
                metadata={"page": 1}, source_file="f.pdf",
            ),
            ChunkRecord(
                content="sec", embedding=[], layer_name="l",
                metadata={"page": 1, "section_path": ["A", "B"]}, source_file="g.md",
            ),
        ]
        with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
            mock.side_effect = lambda texts, **kwargs: [[0.0] for _ in texts]
            await _embed_chunks("key", chunks)
        assert mock.call_args.args[0] == ["[l > f > Page 1]\nbody", "[l > g > A > B]\nsec"]
        assert [c.content for c in chunks] == ["body", "sec"]

    def test_prefix_contains_filename_stem(self):
        """Context prefix must include the filename stem (without extension)."""
        from stratakb.knowledge.vector_store import _build_context_prefix

        prefix = _build_context_prefix(
            layer_name="iso20022",
            source_file="RTGS_UDFS_R2026_JUN.pdf",
            page_number=5,
        )
        assert "RTGS_UDFS_R2026_JUN" in prefix
        assert ".pdf" not in prefix


# ==========================================
# MCP tool registration
# ==========================================


class TestIngestionToolRegistration:
    def test_ingest_tool_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_ingest_documents" in tool_names
