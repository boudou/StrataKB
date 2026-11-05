"""Tests for knowledge layer configuration system.

Tests cover:
- Layer CRUD operations (create, list, get_status, delete)
- Validation (unique names, required fields, chunk config)
- MCP tool registration and invocation
"""

from __future__ import annotations

import contextlib

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
            listed = next(row for row in layers if row["name"] == layer_name)

            # Status
            status = await get_layer_status(db_url, name=layer_name)
            assert status["name"] == layer_name
            assert status["chunk_count"] == 0
            assert status["ingestion_run_count"] == 0

            # The jsonb columns come back decoded, not as JSON text
            for layer in (created, listed, status):
                assert layer["chunk_config"] == {
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "splitting_strategy": "recursive",
                }
                assert layer["metadata_schema"] == {"test_field": "string"}
                assert layer["routing_patterns"] == []

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


# ==========================================
# Layer origin, owner and root folder (require database)
# ==========================================

# Roots live under a prefix no other test uses, since overlap checks scan every layer
ROOT = "/strata_test_50"


async def _drop_layers(db_url, *names):
    from stratakb.knowledge.layers import delete_layer

    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


async def _add_file_row(db_url, layer_name, source_root, relative_path="a.md"):
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    await pool.execute(
        """
        INSERT INTO strata_core.ingested_files
            (layer_id, source_root, relative_path, content_hash)
        SELECT id, $2, $3, $4 FROM strata_core.knowledge_layers WHERE name = $1
        """,
        layer_name,
        source_root,
        relative_path,
        "0" * 64,
    )


async def _add_legacy_chunk(db_url, layer_name):
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    await pool.execute(
        "INSERT INTO strata_vectors.chunks (layer_name, content) VALUES ($1, $2)",
        layer_name,
        "legacy chunk",
    )


async def _updated_at(db_url, layer_name):
    from stratakb.knowledge.layers import get_layer_status

    return (await get_layer_status(db_url, name=layer_name))["updated_at"]


class TestFolderFieldsIntegration:
    """origin, owner and root_path on layers, and the folder helpers."""

    async def test_fields_round_trip(self, db_url):
        from stratakb.knowledge.layers import create_layer, get_layer_status, list_layers

        name = "test_folder_fields"
        root = f"{ROOT}/r"
        await _drop_layers(db_url, name)
        try:
            created = await create_layer(
                db_url, name=name, display_name="Folder", origin="manual",
                owner="u1", root_path=root,
            )
            expected = {"origin": "manual", "owner": "u1", "root_path": root}
            assert {k: created[k] for k in expected} == expected

            listed = next(
                layer for layer in await list_layers(db_url) if layer["name"] == name
            )
            assert {k: listed[k] for k in expected} == expected

            status = await get_layer_status(db_url, name=name)
            assert {k: status[k] for k in expected} == expected
        finally:
            await _drop_layers(db_url, name)

    async def test_fields_default_to_none(self, db_url):
        from stratakb.knowledge.layers import create_layer

        name = "test_folder_none"
        await _drop_layers(db_url, name)
        try:
            created = await create_layer(db_url, name=name, display_name="None")
            assert created["origin"] is None
            assert created["owner"] is None
            assert created["root_path"] is None
        finally:
            await _drop_layers(db_url, name)

    async def test_same_root_raises_root_taken(self, db_url):
        from stratakb.knowledge.layers import RootTakenError, create_layer

        root = f"{ROOT}/taken"
        await _drop_layers(db_url, "test_root_first", "test_root_second")
        try:
            await create_layer(db_url, name="test_root_first", display_name="A", root_path=root)
            with pytest.raises(RootTakenError):
                await create_layer(
                    db_url, name="test_root_second", display_name="B", root_path=root
                )
            with pytest.raises(LayerExistsError):
                await create_layer(db_url, name="test_root_first", display_name="C")
        finally:
            await _drop_layers(db_url, "test_root_first", "test_root_second")

    async def test_find_layer_by_root(self, db_url):
        from stratakb.knowledge.layers import create_layer, find_layer_by_root

        name = "test_find_root"
        root = f"{ROOT}/find"
        await _drop_layers(db_url, name)
        try:
            await create_layer(db_url, name=name, display_name="Find", root_path=root)
            found = await find_layer_by_root(db_url, root)
            assert found is not None and found["name"] == name
            assert isinstance(found["chunk_config"], dict)
            assert await find_layer_by_root(db_url, f"{ROOT}/other") is None
        finally:
            await _drop_layers(db_url, name)

    async def test_find_overlapping_roots(self, db_url):
        from stratakb.knowledge.layers import create_layer, find_overlapping_roots

        layers = {
            "test_overlap_a": f"{ROOT}/p/a",
            "test_overlap_abc": f"{ROOT}/p/a/b/c",
            "test_overlap_bc": f"{ROOT}/p/a/bc",
            "test_overlap_ab": f"{ROOT}/p/a/b",
        }
        await _drop_layers(db_url, *layers)
        try:
            for name, root in layers.items():
                await create_layer(db_url, name=name, display_name=name, root_path=root)
            overlapping = await find_overlapping_roots(db_url, f"{ROOT}/p/a/b")
            assert sorted(layer["root_path"] for layer in overlapping) == [
                f"{ROOT}/p/a",
                f"{ROOT}/p/a/b/c",
            ]
        finally:
            await _drop_layers(db_url, *layers)

    async def test_find_overlapping_roots_treats_underscore_literally(self, db_url):
        from stratakb.knowledge.layers import create_layer, find_overlapping_roots

        name = "test_overlap_literal"
        await _drop_layers(db_url, name)
        try:
            await create_layer(
                db_url, name=name, display_name="Literal", root_path=f"{ROOT}/x_y"
            )
            # LIKE would read "_" as any character and match /x_y against /xzy/...
            assert await find_overlapping_roots(db_url, f"{ROOT}/xzy/sub") == []
        finally:
            await _drop_layers(db_url, name)

    @pytest.mark.parametrize(
        ("setup", "expected"),
        [
            ("empty", True),
            ("other_root", False),
            ("legacy", False),
            ("this_root", True),
            ("subfolder", False),
        ],
    )
    async def test_layer_holds_folder(self, db_url, setup, expected):
        from stratakb.knowledge.layers import create_layer, layer_holds_folder

        name = "test_holds_folder"
        root = f"{ROOT}/docs"
        await _drop_layers(db_url, name)
        try:
            await create_layer(db_url, name=name, display_name="Holds", origin="manual")
            if setup == "other_root":
                await _add_file_row(db_url, name, f"{ROOT}/elsewhere")
            elif setup == "legacy":
                await _add_legacy_chunk(db_url, name)
            elif setup == "this_root":
                await _add_file_row(db_url, name, root, "a.md")
                await _add_file_row(db_url, name, root, "sub/b.md")
            elif setup == "subfolder":
                await _add_file_row(db_url, name, f"{root}/a")
            assert await layer_holds_folder(db_url, name, root) is expected
        finally:
            await _drop_layers(db_url, name)

    async def test_layer_holds_folder_unknown_layer(self, db_url):
        from stratakb.knowledge.layers import layer_holds_folder

        with pytest.raises(LayerNotFoundError):
            await layer_holds_folder(db_url, "nonexistent_layer_xyz", f"{ROOT}/docs")

    @pytest.mark.parametrize(
        ("setter", "column", "value"),
        [
            ("set_root_path", "root_path", f"{ROOT}/set"),
            ("set_description", "description", "New description"),
            ("set_origin", "origin", "preset:test"),
            ("set_layer_answer_mode", "answer_mode", "extractive"),
        ],
    )
    async def test_setters(self, db_url, setter, column, value):
        from stratakb.knowledge import layers

        name = "test_setter"
        await _drop_layers(db_url, name)
        try:
            await layers.create_layer(db_url, name=name, display_name="Setter")
            before = await _updated_at(db_url, name)
            await getattr(layers, setter)(db_url, name, value)
            status = await layers.get_layer_status(db_url, name=name)
            assert status[column] == value
            assert status["updated_at"] > before

            with pytest.raises(LayerNotFoundError):
                await getattr(layers, setter)(db_url, "nonexistent_layer_xyz", value)
        finally:
            await _drop_layers(db_url, name)

    async def test_set_root_path_onto_taken_root_raises(self, db_url):
        from stratakb.knowledge.layers import RootTakenError, create_layer, set_root_path

        root = f"{ROOT}/owned"
        await _drop_layers(db_url, "test_root_owner", "test_root_mover")
        try:
            await create_layer(db_url, name="test_root_owner", display_name="A", root_path=root)
            await create_layer(db_url, name="test_root_mover", display_name="B")
            with pytest.raises(RootTakenError):
                await set_root_path(db_url, "test_root_mover", root)
        finally:
            await _drop_layers(db_url, "test_root_owner", "test_root_mover")

    async def test_origin_too_long_rejected(self, db_url):
        from stratakb.knowledge.layers import create_layer, set_origin

        with pytest.raises(LayerValidationError, match="origin"):
            await create_layer(
                db_url, name="test_long_origin", display_name="Long", origin="x" * 101
            )
        with pytest.raises(LayerValidationError, match="origin"):
            await set_origin(db_url, "test_long_origin", "x" * 101)

    async def test_listing_shows_answer_mode_default_grounded(self, db_url):
        from unittest.mock import AsyncMock, patch

        from stratakb.knowledge.layers import create_layer, list_layers
        from stratakb.mcp_server import mcp

        name = "test_answer_mode_default"
        await _drop_layers(db_url, name)
        try:
            created = await create_layer(db_url, name=name, display_name="Default")
            assert created["answer_mode"] == "grounded"
            rows = await list_layers(db_url)
            assert next(r for r in rows if r["name"] == name)["answer_mode"] == "grounded"

            # The tool passes the stored rows through, so a viewer sees the mode.
            # The tool's own URL is fixed at import, so it lists the rows read above.
            with (
                patch("stratakb.core.auth.get_current_user", return_value=("u", "viewer")),
                patch("stratakb.knowledge.tools.get_current_user", return_value=("u", "viewer")),
                patch(
                    "stratakb.knowledge.tools.list_layers",
                    new_callable=AsyncMock,
                    return_value=rows,
                ),
            ):
                listed = await mcp._tool_manager.get_tool("strata_list_knowledge_layers").fn()
            layer = next(layer for layer in listed["layers"] if layer["name"] == name)
            assert layer["answer_mode"] == "grounded"
        finally:
            await _drop_layers(db_url, name)

    async def test_invalid_answer_mode_refused(self, db_url):
        import asyncpg

        from stratakb.core.database import get_pool
        from stratakb.knowledge.layers import create_layer, set_layer_answer_mode

        name = "test_answer_mode_bad"
        await _drop_layers(db_url, name)
        try:
            with pytest.raises(LayerValidationError, match="answer_mode"):
                await create_layer(db_url, name=name, display_name="Bad", answer_mode="free")
            await create_layer(db_url, name=name, display_name="Bad")
            with pytest.raises(LayerValidationError, match="answer_mode"):
                await set_layer_answer_mode(db_url, name, "free")
            # The column refuses it too, for writes that bypass the setters
            pool = await get_pool(db_url)
            with pytest.raises(asyncpg.CheckViolationError):
                await pool.execute(
                    "UPDATE strata_core.knowledge_layers SET answer_mode = 'free' "
                    "WHERE name = $1",
                    name,
                )
        finally:
            await _drop_layers(db_url, name)

    async def test_create_layer_checks_schema_once(self, db_url):
        from unittest.mock import AsyncMock, patch

        import asyncpg

        from stratakb.core import migrations
        from stratakb.knowledge.layers import create_layer

        names = ("test_schema_one", "test_schema_two")
        await _drop_layers(db_url, *names)
        migrations._checked.discard(db_url)
        try:
            with patch(
                "stratakb.core.migrations.asyncpg.connect",
                new_callable=AsyncMock,
                side_effect=asyncpg.connect,
            ) as mock_connect:
                for name in names:
                    await create_layer(db_url, name=name, display_name=name)
            mock_connect.assert_awaited_once()
            assert db_url in migrations._checked
        finally:
            await _drop_layers(db_url, *names)

    async def test_create_tool_records_manual_origin_and_owner(self, db_url):
        from unittest.mock import AsyncMock, patch

        from stratakb.knowledge.layers import get_layer_status
        from stratakb.mcp_server import mcp

        name = "test_tool_origin"
        await _drop_layers(db_url, name)
        try:
            with (
                patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
                patch(
                    "stratakb.knowledge.tools.get_current_user",
                    return_value=("tester", "admin"),
                ),
                patch("stratakb.knowledge.tools.write_audit_log", new_callable=AsyncMock),
            ):
                await mcp.call_tool(
                    "strata_create_knowledge_layer", {"name": name, "display_name": "Tool"}
                )
            status = await get_layer_status(db_url, name=name)
            assert status["origin"] == "manual"
            assert status["owner"] == "tester"
        finally:
            await _drop_layers(db_url, name)


class TestLayerFieldsForCaller:
    """Viewers and operators do not see who made a layer or its server folder."""

    LAYER = {"name": "x", "origin": "manual", "owner": "u1", "root_path": "/srv/docs"}

    @pytest.mark.parametrize("role", ["viewer", "operator"])
    async def test_below_admin_drops_owner_and_root(self, role):
        from unittest.mock import AsyncMock, patch

        from stratakb.mcp_server import mcp

        with (
            patch("stratakb.core.auth.get_current_user", return_value=("u", role)),
            patch("stratakb.knowledge.tools.get_current_user", return_value=("u", role)),
            patch(
                "stratakb.knowledge.tools.list_layers",
                new_callable=AsyncMock,
                return_value=[dict(self.LAYER)],
            ),
            patch(
                "stratakb.knowledge.tools.get_layer_status",
                new_callable=AsyncMock,
                return_value=dict(self.LAYER),
            ),
        ):
            listed = await mcp._tool_manager.get_tool("strata_list_knowledge_layers").fn()
            status = await mcp._tool_manager.get_tool("strata_get_layer_status").fn(name="x")

        for layer in (listed["layers"][0], status["layer"]):
            assert layer == {"name": "x", "origin": "manual"}

    async def test_admin_sees_owner_and_root(self):
        from unittest.mock import AsyncMock, patch

        from stratakb.mcp_server import mcp

        with (
            patch("stratakb.core.auth.get_current_user", return_value=("u", "admin")),
            patch("stratakb.knowledge.tools.get_current_user", return_value=("u", "admin")),
            patch(
                "stratakb.knowledge.tools.get_layer_status",
                new_callable=AsyncMock,
                return_value=dict(self.LAYER),
            ),
        ):
            status = await mcp._tool_manager.get_tool("strata_get_layer_status").fn(name="x")

        assert status["layer"] == self.LAYER


class TestUpdateLayer:
    """update_layer sets only the settings it is given (require database)."""

    async def test_update_layer_sets_only_given_columns(self, db_url):
        from stratakb.knowledge.layers import create_layer, get_layer_status, update_layer

        name = "test_update_layer"
        await _drop_layers(db_url, name)
        try:
            await create_layer(
                db_url, name=name, display_name="Update", description="old",
                chunk_config={"chunk_size": 800, "chunk_overlap": 100},
                routing_patterns=["old"], answer_mode="extractive", origin="preset:x",
            )
            before = await get_layer_status(db_url, name=name)
            updated = await update_layer(
                db_url, name, description="new", routing_patterns=["new", "newer"]
            )
            after = await get_layer_status(db_url, name=name)

            assert updated["description"] == "new"
            assert after["description"] == "new"
            assert after["routing_patterns"] == ["new", "newer"]
            assert after["chunk_config"] == before["chunk_config"]
            assert after["answer_mode"] == "extractive"
            assert after["origin"] == "preset:x"
            assert after["display_name"] == "Update"
            assert after["updated_at"] > before["updated_at"]
        finally:
            await _drop_layers(db_url, name)

    async def test_update_layer_writes_jsonb_and_strips_display_name(self, db_url):
        from stratakb.knowledge.layers import create_layer, get_layer_status, update_layer

        name = "test_update_layer_json"
        await _drop_layers(db_url, name)
        try:
            await create_layer(db_url, name=name, display_name="Json")
            await update_layer(
                db_url, name, display_name="  Renamed  ",
                metadata_schema={"type": "object"},
                chunk_config={"chunk_size": 500, "chunk_overlap": 50},
                answer_mode="extractive",
            )
            after = await get_layer_status(db_url, name=name)
            assert after["display_name"] == "Renamed"
            assert after["metadata_schema"] == {"type": "object"}
            assert after["chunk_config"] == {
                "chunk_size": 500, "chunk_overlap": 50, "splitting_strategy": "recursive",
            }
            assert after["answer_mode"] == "extractive"
        finally:
            await _drop_layers(db_url, name)

    async def test_update_layer_missing_layer(self, db_url):
        from stratakb.knowledge.layers import update_layer

        await _drop_layers(db_url, "test_update_missing")
        with pytest.raises(LayerNotFoundError):
            await update_layer(db_url, "test_update_missing", description="x")

    async def test_update_layer_validates(self, db_url):
        from stratakb.knowledge.layers import create_layer, get_layer_status, update_layer

        name = "test_update_layer_bad"
        await _drop_layers(db_url, name)
        try:
            await create_layer(db_url, name=name, display_name="Bad", description="kept")
            with pytest.raises(LayerValidationError):
                await update_layer(
                    db_url, name, description="changed",
                    chunk_config={"chunk_size": 200, "chunk_overlap": 200},
                )
            with pytest.raises(LayerValidationError):
                await update_layer(db_url, name, routing_patterns=["(unclosed"])
            assert (await get_layer_status(db_url, name=name))["description"] == "kept"
        finally:
            await _drop_layers(db_url, name)

    async def test_update_layer_without_settings_writes_nothing(self, db_url):
        from stratakb.knowledge.layers import create_layer, get_layer_status, update_layer

        name = "test_update_layer_noop"
        await _drop_layers(db_url, name)
        try:
            await create_layer(db_url, name=name, display_name="Noop")
            before = await get_layer_status(db_url, name=name)
            result = await update_layer(db_url, name)
            assert result["updated_at"] == before["updated_at"]
        finally:
            await _drop_layers(db_url, name)
