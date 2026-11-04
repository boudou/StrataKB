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
            # Chunks can slightly exceed size due to splitter behavior
            assert len(chunk.content) <= 200  # generous bound


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


# ==========================================
# Contextual chunk enrichment tests
# ==========================================


class TestContextualChunkEnrichment:
    def test_context_prefix_generation(self):
        """Chunks should have a context prefix with layer, file, and section."""
        from stratakb.knowledge.ingestion import _build_context_prefix

        prefix = _build_context_prefix(
            layer_name="sepa",
            source_file="SCT_Reason_Codes_v6.0.pdf",
            page_number=12,
        )
        assert "[sepa" in prefix
        assert "SCT_Reason_Codes" in prefix
        assert "12" in prefix

    def test_context_prefix_prepended_to_content(self):
        from stratakb.knowledge.ingestion import _build_context_prefix

        prefix = _build_context_prefix("iso20022", "some_api.yaml", section="Payment Initiation")
        assert "Payment Initiation" in prefix

    def test_chunk_content_has_prefix(self):
        """Chunks produced by _chunk_document must start with the context prefix."""
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        doc = parse_pdf(TEST_PDF)
        chunks = _chunk_document(doc, layer_name="sepa", chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content.startswith("[sepa >"), (
                f"Expected prefix '[sepa >' but got: {chunk.content[:60]!r}"
            )

    def test_prefix_contains_filename_stem(self):
        """Context prefix must include the filename stem (without extension)."""
        from stratakb.knowledge.ingestion import _build_context_prefix

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
