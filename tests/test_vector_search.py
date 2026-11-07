"""Tests for vector search retriever and MCP tool.

Tests cover:
- SearchResult dataclass
- Retriever: search_chunks with mocked embeddings and DB
- Retriever: layer filtering, metadata filtering, top_k clamping
- Retriever: empty query handling, sanitize key
- MCP tool registration and strata_search tool
- Integration: end-to-end search with real DB (mocked embeddings)
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import capture_tools, mock_pool


# ==========================================
# Unit tests — retriever logic
# ==========================================


class TestSearchResult:
    def test_search_result_defaults(self):
        from stratakb.rag.retriever import SearchResult

        r = SearchResult(
            chunk_id="00000000-0000-0000-0000-000000000001",
            content="test",
            score=0.95,
            layer_name="iso20022",
            source_file="test.yaml",
            source_section="section1",
            doc_type="yaml",
        )
        assert r.chunk_id == "00000000-0000-0000-0000-000000000001"
        assert r.content == "test"
        assert r.score == 0.95
        assert r.metadata == {}

    def test_search_result_with_metadata(self):
        from stratakb.rag.retriever import SearchResult

        r = SearchResult(
            chunk_id="00000000-0000-0000-0000-000000000002",
            content="test",
            score=0.85,
            layer_name="sepa",
            source_file="codes.csv",
            source_section="",
            doc_type="csv",
            metadata={"scheme": "SCT"},
        )
        assert r.metadata == {"scheme": "SCT"}


class TestSanitizeKey:
    def test_alphanumeric_unchanged(self):
        from stratakb.rag.retriever import _sanitize_key

        assert _sanitize_key("doc_type") == "doc_type"

    def test_strips_special_chars(self):
        from stratakb.rag.retriever import _sanitize_key

        assert _sanitize_key("key'; DROP TABLE") == "keyDROPTABLE"

    def test_empty_string(self):
        from stratakb.rag.retriever import _sanitize_key

        assert _sanitize_key("") == ""


def _mock_settings(**overrides):
    """Create a mock Settings object with hybrid search disabled by default for unit tests."""
    defaults = {
        "hybrid_search_enabled": False,
        "hnsw_ef_search": 40,
        "cache_enabled": False,
        "reranking_enabled": False,
        "embedding_backend": "local",
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class TestSearchChunksUnit:
    @pytest.fixture(autouse=True)
    def _no_stamp_read(self):
        """The fake URL has no database: never read the layers stamp."""
        with patch(
            "stratakb.rag.retriever._watcher.fetch",
            new_callable=AsyncMock,
            return_value=(0, None),
        ):
            yield

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        from stratakb.rag.retriever import search_chunks

        results = await search_chunks("postgresql://test", "sk-test", "")
        assert results == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self):
        from stratakb.rag.retriever import search_chunks

        results = await search_chunks("postgresql://test", "sk-test", "   ")
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_clamped_to_range(self):
        """top_k should be clamped between 1 and 50."""
        from stratakb.rag.retriever import search_chunks

        fake_embedding = [0.01] * 768
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        with patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            return_value=[fake_embedding],
        ), patch(
            "stratakb.rag.retriever.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool(mock_conn),
        ), patch(
            "stratakb.config.get_settings",
            return_value=_mock_settings(),
        ):
            # top_k=0 should become 1
            await search_chunks(
                "postgresql://test", "sk-test", "test query", top_k=0
            )
            call_args = mock_conn.fetch.call_args
            assert call_args[0][2] == 1  # $2 = top_k, clamped to 1

            # top_k=100 should become 50
            await search_chunks(
                "postgresql://test", "sk-test", "test query", top_k=100
            )
            call_args = mock_conn.fetch.call_args
            assert call_args[0][2] == 50

    @pytest.mark.asyncio
    async def test_layer_filter_adds_where_clause(self):
        """Layer filter should add IN clause to SQL."""
        from stratakb.rag.retriever import search_chunks

        fake_embedding = [0.01] * 768
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        with patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            return_value=[fake_embedding],
        ), patch(
            "stratakb.rag.retriever.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool(mock_conn),
        ), patch(
            "stratakb.config.get_settings",
            return_value=_mock_settings(),
        ):
            await search_chunks(
                "postgresql://test",
                "sk-test",
                "test query",
                layers=["iso20022", "sepa"],
            )
            sql = mock_conn.fetch.call_args[0][0]
            assert "layer_name IN" in sql
            params = mock_conn.fetch.call_args[0]
            assert "iso20022" in params
            assert "sepa" in params

    @pytest.mark.asyncio
    async def test_metadata_filter_adds_where_clause(self):
        """Metadata filter should add JSONB conditions."""
        from stratakb.rag.retriever import search_chunks

        fake_embedding = [0.01] * 768
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        with patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            return_value=[fake_embedding],
        ), patch(
            "stratakb.rag.retriever.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool(mock_conn),
        ), patch(
            "stratakb.config.get_settings",
            return_value=_mock_settings(),
        ):
            await search_chunks(
                "postgresql://test",
                "sk-test",
                "test query",
                metadata_filter={"doc_type": "yaml"},
            )
            sql = mock_conn.fetch.call_args[0][0]
            assert "metadata->>'doc_type'" in sql

    @pytest.mark.asyncio
    @pytest.mark.parametrize("filters", [
        {},
        {"layers": ["sepa"]},
        {"layers": ["sepa"], "metadata_filter": {"doc_type": "yaml"}},
    ])
    async def test_vector_sql_skips_chunks_without_embedding(self, filters):
        """A chunk without an embedding has no cosine distance: the vector query leaves it out."""
        from stratakb.rag.retriever import search_chunks

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        with patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            return_value=[[0.01] * 768],
        ), patch(
            "stratakb.rag.retriever.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool(mock_conn),
        ), patch(
            "stratakb.config.get_settings",
            return_value=_mock_settings(),
        ):
            await search_chunks("postgresql://test", "sk-test", "test query", **filters)
            sql = mock_conn.fetch.call_args[0][0]
            assert "WHERE embedding IS NOT NULL" in sql

    @pytest.mark.asyncio
    async def test_results_parsed_correctly(self):
        """Results from DB should be parsed into SearchResult objects."""
        from stratakb.rag.retriever import search_chunks

        fake_embedding = [0.01] * 768
        import json

        mock_row = {
            "id": "00000000-0000-0000-0000-000000000099",
            "content": "Test content about payment initiation",
            "score": 0.9523,
            "layer_name": "iso20022",
            "source_file": "test.yaml",
            "source_section": "Operations",
            "doc_type": "yaml",
            "metadata": json.dumps({"api": "Payment Initiation"}),
        }

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [mock_row]

        with patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            return_value=[fake_embedding],
        ), patch(
            "stratakb.rag.retriever.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool(mock_conn),
        ), patch(
            "stratakb.config.get_settings",
            return_value=_mock_settings(),
        ):
            results = await search_chunks(
                "postgresql://test", "sk-test", "payment initiation"
            )

            assert len(results) == 1
            assert results[0].chunk_id == "00000000-0000-0000-0000-000000000099"
            assert results[0].content == "Test content about payment initiation"
            assert results[0].score == 0.9523
            assert results[0].layer_name == "iso20022"
            assert results[0].source_file == "test.yaml"
            assert results[0].doc_type == "yaml"
            assert results[0].metadata == {"api": "Payment Initiation"}


# ==========================================
# MCP tool registration tests
# ==========================================


class TestRAGToolRegistration:
    def test_strata_search_tool_registered(self):
        from stratakb.rag.tools import register_rag_tools

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda f: f
        register_rag_tools(mock_mcp, "postgresql://test", "sk-test")

        # strata_search + strata_ask = 2 RAG tools
        assert mock_mcp.tool.call_count == 2

    def test_total_tools_including_rag(self):
        """All tools (knowledge + RAG + admin) should be registered."""
        from stratakb.knowledge.tools import register_knowledge_tools
        from stratakb.rag.tools import register_rag_tools

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda f: f
        register_knowledge_tools(mock_mcp, "postgresql://test", "sk-test")
        register_rag_tools(mock_mcp, "postgresql://test", "sk-test")

        # 7 knowledge tools + 2 RAG tools (strata_search + strata_ask) = 9
        assert mock_mcp.tool.call_count == 9


class TestSearchTool:
    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        from stratakb.rag.tools import register_rag_tools

        fn = capture_tools(register_rag_tools, "postgresql://test", "sk-test")["strata_search"]
        result = await fn(query="")
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error_openai_backend(self):
        from stratakb.rag.tools import register_rag_tools

        fn = capture_tools(register_rag_tools, "postgresql://test", "")["strata_search"]

        with patch("stratakb.config.get_settings") as mock_settings:
            mock_settings.return_value.embedding_backend = "openai"
            mock_settings.return_value.embedding_base_url = ""
            mock_settings.return_value.strata_user = "test"
            mock_settings.return_value.strata_role = "admin"
            result = await fn(query="test query")
        assert result["status"] == "error"
        assert "API key" in result["error"]


# ==========================================
# Integration tests (require DB)
# ==========================================


class TestVectorSearchIntegration:
    @pytest.mark.asyncio
    async def test_search_with_ingested_data(self, db_url, vacuumed_chunks):
        """End-to-end: ingest data with fake embeddings, then search."""
        from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer
        from stratakb.knowledge.vector_store import ChunkRecord, insert_chunks
        from stratakb.rag.retriever import search_chunks

        layer_name = "test_search_integration"
        fake_embedding = [1.0] * 384 + [0.0] * 384  # query direction

        try:
            # Cleanup
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

            await create_layer(
                db_url,
                name=layer_name,
                display_name="Test Search",
            )

            # Insert test chunks with embeddings that differ in direction
            # (uniform vectors like [0.1]*N all point the same way, giving equal cosine similarity)
            close_embedding = [1.0] * 384 + [0.0] * 384  # same direction as query
            mid_embedding = [1.0] * 192 + [0.0] * 192 + [1.0] * 192 + [0.0] * 192
            far_embedding = [0.0] * 384 + [1.0] * 384  # orthogonal to query

            chunks = [
                ChunkRecord(
                    content="SEPA Credit Transfer pain.001 message format",
                    embedding=close_embedding,
                    layer_name=layer_name,
                    source_file="sepa_guide.pdf",
                    source_section="Chapter 3",
                    doc_type="pdf",
                    metadata={"message_type": "pain.001", "scheme": "SCT"},
                ),
                ChunkRecord(
                    content="The payment initiation API handles payment processing",
                    embedding=mid_embedding,
                    layer_name=layer_name,
                    source_file="payments_api.yaml",
                    source_section="Operations",
                    doc_type="yaml",
                    metadata={"service_domain": "PaymentExecution"},
                ),
                ChunkRecord(
                    content="Rejection code AC01 IncorrectAccountNumber invalid IBAN",
                    embedding=far_embedding,
                    layer_name=layer_name,
                    source_file="codes.csv",
                    source_section="Row 1",
                    doc_type="csv",
                    metadata={"code_type": "rejection", "scheme": "SCT"},
                ),
            ]
            await insert_chunks(db_url, chunks)

            # Search with mocked embedding (returns embedding close to chunk 1)
            with patch(
                "stratakb.rag.retriever.embed_texts",
                new_callable=AsyncMock,
                return_value=[fake_embedding],
            ):
                results = await search_chunks(
                    db_url, "fake-key", "SEPA pain.001", top_k=3
                )

            assert len(results) == 3
            # First result should be closest to the fake_embedding ([0.1]*3072)
            assert results[0].content == "SEPA Credit Transfer pain.001 message format"
            assert results[0].score > 0
            assert results[0].layer_name == layer_name
            assert results[0].source_file == "sepa_guide.pdf"
            assert results[0].doc_type == "pdf"
            assert results[0].metadata["message_type"] == "pain.001"

        finally:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_search_layer_filtering(self, db_url):
        """Search with layer filter should only return chunks from that layer."""
        from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer
        from stratakb.knowledge.vector_store import ChunkRecord, insert_chunks
        from stratakb.rag.retriever import search_chunks

        layer_a = "test_search_layer_a"
        layer_b = "test_search_layer_b"
        fake_embedding = [0.1] * 768

        try:
            for name in [layer_a, layer_b]:
                try:
                    await delete_layer(db_url, name=name)
                except LayerNotFoundError:
                    pass
                await create_layer(db_url, name=name, display_name=f"Test {name}")

            await insert_chunks(db_url, [
                ChunkRecord(
                    content="Layer A content",
                    embedding=fake_embedding,
                    layer_name=layer_a,
                    doc_type="pdf",
                ),
                ChunkRecord(
                    content="Layer B content",
                    embedding=fake_embedding,
                    layer_name=layer_b,
                    doc_type="pdf",
                ),
            ])

            with patch(
                "stratakb.rag.retriever.embed_texts",
                new_callable=AsyncMock,
                return_value=[fake_embedding],
            ):
                # Search only layer A
                results = await search_chunks(
                    db_url, "fake-key", "content", layers=[layer_a]
                )
                assert all(r.layer_name == layer_a for r in results)
                assert len(results) == 1

                # Search only layer B
                results = await search_chunks(
                    db_url, "fake-key", "content", layers=[layer_b]
                )
                assert all(r.layer_name == layer_b for r in results)
                assert len(results) == 1

                # Search both layers
                results = await search_chunks(
                    db_url, "fake-key", "content", layers=[layer_a, layer_b]
                )
                assert len(results) == 2

        finally:
            for name in [layer_a, layer_b]:
                try:
                    await delete_layer(db_url, name=name)
                except LayerNotFoundError:
                    pass

    @pytest.mark.asyncio
    async def test_search_metadata_filtering(self, db_url):
        """Search with metadata filter should only return matching chunks."""
        from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer
        from stratakb.knowledge.vector_store import ChunkRecord, insert_chunks
        from stratakb.rag.retriever import search_chunks

        layer_name = "test_search_metadata"
        fake_embedding = [0.1] * 768

        try:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

            await create_layer(
                db_url, name=layer_name, display_name="Test Metadata"
            )

            await insert_chunks(db_url, [
                ChunkRecord(
                    content="YAML content",
                    embedding=fake_embedding,
                    layer_name=layer_name,
                    doc_type="yaml",
                    metadata={"doc_type": "yaml"},
                ),
                ChunkRecord(
                    content="CSV content",
                    embedding=fake_embedding,
                    layer_name=layer_name,
                    doc_type="csv",
                    metadata={"doc_type": "csv"},
                ),
            ])

            with patch(
                "stratakb.rag.retriever.embed_texts",
                new_callable=AsyncMock,
                return_value=[fake_embedding],
            ):
                results = await search_chunks(
                    db_url,
                    "fake-key",
                    "content",
                    metadata_filter={"doc_type": "yaml"},
                )
                assert len(results) == 1
                assert results[0].metadata["doc_type"] == "yaml"

        finally:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_search_no_results(self, db_url):
        """Search on non-existent layer returns empty results."""
        from stratakb.rag.retriever import search_chunks

        fake_embedding = [0.1] * 768
        with patch(
            "stratakb.rag.retriever.embed_texts",
            new_callable=AsyncMock,
            return_value=[fake_embedding],
        ):
            results = await search_chunks(
                db_url,
                "fake-key",
                "nonexistent query",
                layers=["nonexistent_layer_xyz"],
            )
            assert results == []

    @pytest.mark.asyncio
    async def test_search_result_format(self, db_url):
        """Verify all expected fields are present in search results."""
        from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer
        from stratakb.knowledge.vector_store import ChunkRecord, insert_chunks
        from stratakb.rag.retriever import search_chunks

        layer_name = "test_search_format"
        fake_embedding = [0.1] * 768

        try:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

            await create_layer(
                db_url, name=layer_name, display_name="Test Format"
            )

            await insert_chunks(db_url, [
                ChunkRecord(
                    content="Format test content",
                    embedding=fake_embedding,
                    layer_name=layer_name,
                    source_file="test.pdf",
                    source_section="Section 1",
                    doc_type="pdf",
                    metadata={"key": "value"},
                ),
            ])

            with patch(
                "stratakb.rag.retriever.embed_texts",
                new_callable=AsyncMock,
                return_value=[fake_embedding],
            ):
                results = await search_chunks(db_url, "fake-key", "test")

            assert len(results) >= 1
            r = results[0]
            # Verify all required fields
            assert isinstance(r.content, str) and len(r.content) > 0
            assert isinstance(r.score, float) and 0 <= r.score <= 1
            assert isinstance(r.layer_name, str)
            assert isinstance(r.source_file, str)
            assert isinstance(r.source_section, str)
            assert isinstance(r.doc_type, str)
            assert isinstance(r.metadata, dict)

        finally:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_vector_search_skips_null_embedding(self, db_url, monkeypatch):
        """A chunk with a NULL embedding is left out of a vector-only search, not a crash."""
        import uuid

        from stratakb.config import get_settings
        from stratakb.core.database import get_pool
        from stratakb.rag import retriever
        from stratakb.rag.retriever import search_chunks

        settings = get_settings()
        vector_only = settings.model_copy(update={"hybrid_search_enabled": False})
        monkeypatch.setattr("stratakb.config.get_settings", lambda: vector_only)
        monkeypatch.setattr("stratakb.rag.retriever._get_cache", lambda: None)

        original = retriever._search_fetch

        async def _fetch_without_index(pool, settings, sql, params):
            async with pool.acquire() as conn:
                # The HNSW index never returns a NULL vector; a sequential scan does
                await conn.execute("SET enable_indexscan = off")
                try:
                    one_conn = SimpleNamespace(acquire=lambda: contextlib.nullcontext(conn))
                    return await original(one_conn, settings, sql, params)
                finally:
                    await conn.execute("RESET enable_indexscan")

        monkeypatch.setattr("stratakb.rag.retriever._search_fetch", _fetch_without_index)

        pool = await get_pool(db_url)
        layer = f"nullemb_{uuid.uuid4().hex[:8]}"
        query = [0.1] * settings.embedding_dimensions
        vector = "[" + ",".join(str(x) for x in query) + "]"
        insert = """
            INSERT INTO strata_vectors.chunks (layer_name, content, embedding, source_file)
            VALUES ($1, $2, $3::vector, $4)
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(insert, layer, "First embedded note.", vector, "a")
                await conn.execute(insert, layer, "Second embedded note.", vector, "b")
                await conn.execute(insert, layer, "A note without an embedding.", None, "c")

            with patch(
                "stratakb.rag.retriever.embed_texts",
                new_callable=AsyncMock,
                return_value=[query],
            ):
                results = await search_chunks(db_url, None, "note", layers=[layer], top_k=5)

            assert sorted(r.source_file for r in results) == ["a", "b"]
            assert all(r.similarity is not None for r in results)
        finally:
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM strata_vectors.chunks WHERE layer_name = $1", layer)
