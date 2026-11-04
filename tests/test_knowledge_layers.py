"""Tests for knowledge layer configuration system.

Tests cover:
- Layer CRUD operations (create, list, get_status, delete)
- Validation (unique names, required fields, chunk config)
- MCP tool registration and invocation
"""

from __future__ import annotations

import pytest

from stratakb.knowledge.layers import (
    LayerExistsError,
    LayerNotFoundError,
    LayerValidationError,
    _validate_chunk_config,
    _validate_name,
)


# ==========================================
# Name validation tests
# ==========================================


class TestValidateName:
    def test_valid_names(self):
        for name in ["runbooks", "sepa", "api_docs", "product_v2_docs", "iso_20022"]:
            _validate_name(name)  # Should not raise

    def test_empty_name_raises(self):
        with pytest.raises(LayerValidationError, match="Invalid layer name"):
            _validate_name("")

    def test_too_short_name_raises(self):
        with pytest.raises(LayerValidationError, match="Invalid layer name"):
            _validate_name("ab")

    def test_uppercase_raises(self):
        with pytest.raises(LayerValidationError, match="Invalid layer name"):
            _validate_name("SEPA")

    def test_starts_with_number_raises(self):
        with pytest.raises(LayerValidationError, match="Invalid layer name"):
            _validate_name("1sepa")

    def test_special_chars_raises(self):
        with pytest.raises(LayerValidationError, match="Invalid layer name"):
            _validate_name("api-docs")

    def test_spaces_raises(self):
        with pytest.raises(LayerValidationError, match="Invalid layer name"):
            _validate_name("api docs")


# ==========================================
# Chunk config validation tests
# ==========================================


class TestValidateChunkConfig:
    def test_defaults(self):
        result = _validate_chunk_config({})
        assert result["chunk_size"] == 1000
        assert result["chunk_overlap"] == 200
        assert result["splitting_strategy"] == "recursive"

    def test_custom_values(self):
        result = _validate_chunk_config(
            {"chunk_size": 500, "chunk_overlap": 100, "splitting_strategy": "markdown"}
        )
        assert result["chunk_size"] == 500
        assert result["chunk_overlap"] == 100

    def test_chunk_size_too_small_raises(self):
        with pytest.raises(LayerValidationError, match="chunk_size"):
            _validate_chunk_config({"chunk_size": 50})

    def test_negative_overlap_raises(self):
        with pytest.raises(LayerValidationError, match="chunk_overlap"):
            _validate_chunk_config({"chunk_overlap": -1})

    def test_overlap_exceeds_size_raises(self):
        with pytest.raises(LayerValidationError, match="chunk_overlap must be less"):
            _validate_chunk_config({"chunk_size": 500, "chunk_overlap": 500})

    def test_non_dict_raises(self):
        with pytest.raises(LayerValidationError, match="must be a JSON object"):
            _validate_chunk_config("invalid")


# ==========================================
# MCP server import / tool registration
# ==========================================


class TestToolRegistration:
    def test_mcp_server_imports_with_knowledge_tools(self):
        """MCP server should import and register knowledge tools."""
        from stratakb.mcp_server import mcp

        assert mcp is not None
        assert mcp.name == "StrataKB"

    def test_knowledge_tools_registered(self):
        """All 4 knowledge layer tools should be registered."""
        from stratakb.mcp_server import mcp

        # FastMCP stores tools internally — verify by listing them
        # The tool names should be accessible
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_list_knowledge_layers" in tool_names
        assert "strata_create_knowledge_layer" in tool_names
        assert "strata_get_layer_status" in tool_names
        assert "strata_delete_knowledge_layer" in tool_names


# ==========================================
# Integration tests (require database)
# ==========================================


@pytest.fixture
def db_url():
    """Database URL from settings."""
    from stratakb.config import get_settings
    return get_settings().database_url


class TestLayerCRUDIntegration:
    """Integration tests that require a running PostgreSQL database.

    These tests are marked with pytest.mark.integration and can be skipped
    when no database is available.
    """

    @pytest.mark.asyncio
    async def test_create_list_delete_lifecycle(self, db_url):
        """Full lifecycle: create → list → status → delete."""
        from stratakb.knowledge.layers import (
            create_layer,
            delete_layer,
            get_layer_status,
            list_layers,
        )

        layer_name = "test_layer_lifecycle"

        try:
            # Clean up from any previous failed run
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

            # Create
            created = await create_layer(
                db_url,
                name=layer_name,
                display_name="Test Layer",
                description="A test knowledge layer",
                metadata_schema={"test_field": "string"},
                chunk_config={"chunk_size": 500, "chunk_overlap": 100},
            )
            assert created["name"] == layer_name
            assert created["display_name"] == "Test Layer"
            assert created["status"] == "active"
            assert created["doc_count"] == 0

            # List — should contain our layer
            layers = await list_layers(db_url)
            names = [l["name"] for l in layers]
            assert layer_name in names

            # Status
            status = await get_layer_status(db_url, name=layer_name)
            assert status["name"] == layer_name
            assert status["chunk_count"] == 0
            assert status["ingestion_run_count"] == 0

            # Delete
            result = await delete_layer(db_url, name=layer_name)
            assert result["deleted"] is True
            assert result["layer_name"] == layer_name

            # Verify deleted
            with pytest.raises(LayerNotFoundError):
                await get_layer_status(db_url, name=layer_name)

        finally:
            # Cleanup
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_create_duplicate_raises(self, db_url):
        """Creating a layer with an existing name should raise."""
        from stratakb.knowledge.layers import create_layer, delete_layer

        layer_name = "test_duplicate"

        try:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

            await create_layer(
                db_url,
                name=layer_name,
                display_name="First",
            )

            with pytest.raises(LayerExistsError, match="already exists"):
                await create_layer(
                    db_url,
                    name=layer_name,
                    display_name="Second",
                )
        finally:
            try:
                await delete_layer(db_url, name=layer_name)
            except LayerNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_get_nonexistent_layer_raises(self, db_url):
        """Getting status of a non-existent layer should raise."""
        from stratakb.knowledge.layers import get_layer_status

        with pytest.raises(LayerNotFoundError):
            await get_layer_status(db_url, name="nonexistent_layer_xyz")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_layer_raises(self, db_url):
        """Deleting a non-existent layer should raise."""
        from stratakb.knowledge.layers import delete_layer

        with pytest.raises(LayerNotFoundError):
            await delete_layer(db_url, name="nonexistent_layer_xyz")

    @pytest.mark.asyncio
    async def test_create_with_invalid_name_raises(self, db_url):
        """Creating a layer with invalid name should raise validation error."""
        from stratakb.knowledge.layers import create_layer

        with pytest.raises(LayerValidationError):
            await create_layer(db_url, name="INVALID", display_name="Bad")

    @pytest.mark.asyncio
    async def test_create_without_display_name_raises(self, db_url):
        """Creating a layer without display_name should raise."""
        from stratakb.knowledge.layers import create_layer

        with pytest.raises(LayerValidationError, match="display_name"):
            await create_layer(db_url, name="valid_name", display_name="")
