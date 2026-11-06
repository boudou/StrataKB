"""Tests for layer presets: loading, dry-run parsing and layer creation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from stratakb.knowledge.layers import LayerValidationError
from stratakb.knowledge.presets import (
    create_preset_layer,
    dry_run_parse,
    load_preset,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
SEPA_PRESET = EXAMPLES_DIR / "sepa" / "preset.yaml"
ISO_PRESET = EXAMPLES_DIR / "iso20022" / "preset.yaml"
PEPS_PRESET = EXAMPLES_DIR / "peps" / "preset.yaml"


def _write_preset(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "preset.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _stored_layer(preset, **overrides) -> dict:
    """A stored layer whose settings match the preset, except for `overrides`."""
    return {
        "name": preset.name,
        "origin": None,
        "display_name": preset.display_name,
        "description": preset.description,
        "metadata_schema": preset.metadata_schema,
        "chunk_config": preset.chunk_config,
        "routing_patterns": preset.routing_patterns,
        "structure_patterns": preset.structure_patterns,
        "answer_mode": "grounded",
        **overrides,
    }


class TestLoadPreset:
    @pytest.mark.parametrize("path", [SEPA_PRESET, ISO_PRESET, PEPS_PRESET])
    def test_example_presets_load(self, path):
        preset = load_preset(path)
        assert preset.name in ("sepa", "iso20022", "python_peps")
        assert preset.display_name
        assert preset.description
        assert preset.routing_patterns
        assert preset.sources
        assert preset.root == path.parent

    def test_peps_preset_source(self):
        preset = load_preset(PEPS_PRESET)
        assert [s.path for s in preset.sources] == ["source/peps"]
        assert preset.chunk_config["chunk_size"] == 1500
        assert preset.chunk_config["chunk_overlap"] == 200

    def test_chunk_config_defaults_applied(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        preset = load_preset(path)
        assert preset.chunk_config["chunk_size"] == 1000
        assert preset.chunk_config["chunk_overlap"] == 200
        assert preset.routing_patterns == []

    def test_root_key_is_relative_to_preset(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "root: ../shared-docs\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        assert load_preset(path).root == tmp_path / "../shared-docs"

    def test_missing_sources_rejected(self, tmp_path):
        path = _write_preset(tmp_path, "name: runbooks\ndisplay_name: Ops Runbooks\n")
        with pytest.raises(LayerValidationError, match="sources"):
            load_preset(path)

    def test_invalid_name_rejected(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: Ops-Runbooks\n"
            "display_name: Ops Runbooks\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        with pytest.raises(LayerValidationError):
            load_preset(path)

    def test_invalid_routing_pattern_rejected(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "routing_patterns: ['(unclosed']\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        with pytest.raises(LayerValidationError, match="routing pattern"):
            load_preset(path)

    def test_structure_patterns_loaded(self, tmp_path):
        assert load_preset(SEPA_PRESET).structure_patterns == [
            "^Identification:\\s*(?P<label>AT-[A-Z]?\\d{1,3})\\b",
        ]
        assert load_preset(ISO_PRESET).structure_patterns == [
            "^(?P<label>(?:pain|pacs|camt)\\.\\d{3}\\.\\d{3}\\.\\d{2})\\b",
        ]
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        assert load_preset(path).structure_patterns == []

    @pytest.mark.parametrize("pattern, message", [
        ("'(unclosed'", "structure pattern"),
        ("'\\s*'", "empty line"),
        ("42", "list of strings"),
        # Nested unbounded repeats could stall an ingest on a long line
        ("'^(?P<label>(AT-\\d+)+)'", "repeats a repeated part"),
        ("'^(\\w+\\s?)*:'", "repeats a repeated part"),
    ])
    def test_invalid_structure_pattern_refused(self, tmp_path, pattern, message):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            f"structure_patterns: [{pattern}]\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        with pytest.raises(LayerValidationError, match=message):
            load_preset(path)

    def test_source_without_path_rejected(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "sources:\n"
            "  - {name: docs}\n"
        ))
        with pytest.raises(LayerValidationError, match="path"):
            load_preset(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_preset(tmp_path / "nope.yaml")

    def test_answer_mode_absent_is_none(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        assert load_preset(path).answer_mode is None

    def test_answer_mode_empty_is_none(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "answer_mode:\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        assert load_preset(path).answer_mode is None

    def test_metadata_schema_must_be_mapping(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "metadata_schema: [a, b]\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        with pytest.raises(LayerValidationError, match="metadata_schema"):
            load_preset(path)

    @pytest.mark.parametrize("display_name", ['"   "', "42"])
    def test_blank_display_name_rejected(self, tmp_path, display_name):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            f"display_name: {display_name}\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        with pytest.raises(LayerValidationError, match="display_name"):
            load_preset(path)

    def test_answer_mode_invalid_refused(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "answer_mode: verbatim\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        with pytest.raises(LayerValidationError, match="answer_mode"):
            load_preset(path)


class TestPepsRouting:
    """The PEPs preset's routing patterns, compiled as the server compiles them."""

    @staticmethod
    def _patterns(*paths: Path) -> dict:
        from stratakb.rag.layer_detect import compile_patterns

        presets = [load_preset(p) for p in paths]
        return compile_patterns({p.name: p.routing_patterns for p in presets})

    @pytest.mark.parametrize("query", [
        "PEP 8",
        "pep-484",
        "PEP8",
        "What does PEP 572 say?",
        "Python Enhancement Proposal",
        "python enhancement proposals on typing",
    ])
    def test_routed(self, query):
        from stratakb.rag.layer_detect import detect_layers

        assert detect_layers(query, self._patterns(PEPS_PRESET)) == ["python_peps"]

    @pytest.mark.parametrize(
        "query", ["pepper", "step 8", "pep 12345", "general payments question"]
    )
    def test_not_routed(self, query):
        from stratakb.rag.layer_detect import detect_layers

        assert detect_layers(query, self._patterns(PEPS_PRESET)) == []

    def test_alongside_the_other_examples(self):
        from stratakb.rag.layer_detect import detect_layers

        patterns = self._patterns(SEPA_PRESET, ISO_PRESET, PEPS_PRESET)
        assert detect_layers("PEP 8 line length", patterns) == ["python_peps"]
        assert "python_peps" not in detect_layers("pacs.008 settlement amount", patterns)


class TestDryRun:
    def test_sample_reason_codes_parse(self):
        result = dry_run_parse(load_preset(SEPA_PRESET))
        codes = result.group_stats["reason_codes"]
        assert codes["files"] == 1
        assert codes["parsed"] == 1
        assert codes["chunks"] == 12  # one chunk per code row

    def test_missing_folders_report_download_link(self):
        result = dry_run_parse(load_preset(SEPA_PRESET), root=Path("/nonexistent-root"))
        assert result.total_files == 0
        rulebook_errors = [e for e in result.errors if e["group"] == "rulebooks"]
        assert rulebook_errors
        assert "europeanpaymentscouncil.eu" in rulebook_errors[0]["error"]

    def test_custom_root(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "failover.md").write_text("# Failover\n\nSwitch traffic to the standby region.\n")
        preset_path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        result = dry_run_parse(load_preset(preset_path))
        assert result.parsed_files == 1
        assert result.total_chunks >= 1
        assert result.errors == []

    def test_secret_names_not_parsed(self, tmp_path):
        from stratakb.knowledge import presets

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "failover.md").write_text("# Failover\n\nSwitch traffic to the standby region.\n")
        (docs / "client_secret.json").write_text('{"client_secret": "x"}')
        preset_path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        with patch(
            "stratakb.knowledge.presets._parse_file", side_effect=presets._parse_file
        ) as spy:
            result = dry_run_parse(load_preset(preset_path))

        assert [c.args[0].name for c in spy.call_args_list] == ["failover.md"]
        assert result.group_stats["docs"]["files"] == 1
        assert result.parsed_files == 1
        assert result.errors == []


class TestCreatePresetLayer:
    async def test_passes_preset_settings_to_create_layer(self):
        preset = load_preset(ISO_PRESET)
        with patch(
            "stratakb.knowledge.presets.create_layer", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = {"name": "iso20022"}
            result = await create_preset_layer("postgresql://test", preset)

        assert result["status"] == "created"
        kwargs = mock_create.call_args.kwargs
        assert kwargs["name"] == "iso20022"
        assert kwargs["routing_patterns"] == preset.routing_patterns
        assert kwargs["chunk_config"]["chunk_size"] == 1500
        assert kwargs["origin"] == "preset:iso20022"
        assert kwargs["answer_mode"] == "grounded"

    async def test_new_layer_without_answer_mode_is_grounded(self):
        preset = load_preset(SEPA_PRESET)
        assert preset.answer_mode is None
        with patch(
            "stratakb.knowledge.presets.create_layer", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = {"name": "sepa"}
            result = await create_preset_layer("postgresql://test", preset)

        assert result == {"status": "created", "layer": {"name": "sepa"}, "changes": []}
        assert mock_create.call_args.kwargs["answer_mode"] == "grounded"
        assert mock_create.call_args.kwargs["structure_patterns"] == preset.structure_patterns

    async def test_structure_patterns_stored(self, tmp_path):
        import contextlib

        from stratakb.config import get_settings
        from stratakb.knowledge.layers import (
            LayerNotFoundError,
            delete_layer,
            get_layer_status,
            list_layers,
        )

        db_url = get_settings().database_url
        name = "test_preset_structure"
        body = (
            f"name: {name}\n"
            "display_name: Structure\n"
            "structure_patterns: [{patterns}]\n"
            "sources:\n"
            "  - {{name: docs, path: docs}}\n"
        )
        path = _write_preset(tmp_path, body.format(patterns="'(?P<label>AT-\\d+)\\b'"))
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)
        try:
            result = await create_preset_layer(db_url, load_preset(path))
            assert result["layer"]["structure_patterns"] == ["(?P<label>AT-\\d+)\\b"]
            listed = next(layer for layer in await list_layers(db_url) if layer["name"] == name)
            assert listed["structure_patterns"] == ["(?P<label>AT-\\d+)\\b"]

            # Applying the preset again brings the patterns up to the YAML
            _write_preset(tmp_path, body.format(patterns="'(?P<label>DS-\\d+)\\b'"))
            again = await create_preset_layer(db_url, load_preset(path))
            assert again["changes"] == ["structure_patterns"]
            stored = await get_layer_status(db_url, name=name)
            assert stored["structure_patterns"] == ["(?P<label>DS-\\d+)\\b"]
        finally:
            with contextlib.suppress(LayerNotFoundError):
                await delete_layer(db_url, name=name)

    async def _create_existing(self, existing_origin, existing_mode="grounded", preset=None,
                               update_error=None, **stored):
        """Apply a preset to a mocked existing layer that matches it, except for `stored`."""
        from stratakb.knowledge.layers import LayerExistsError

        preset = preset or load_preset(SEPA_PRESET)
        existing = _stored_layer(
            preset, origin=existing_origin, answer_mode=existing_mode, **stored
        )
        with (
            patch(
                "stratakb.knowledge.presets.create_layer",
                new_callable=AsyncMock,
                side_effect=LayerExistsError("exists"),
            ),
            patch(
                "stratakb.knowledge.presets.get_layer",
                new_callable=AsyncMock,
                return_value=existing,
            ),
            patch(
                "stratakb.knowledge.presets.update_layer",
                new_callable=AsyncMock,
                side_effect=update_error,
            ) as mock_update,
        ):
            result = await create_preset_layer("postgresql://test", preset)
        self.mock_update = mock_update
        return result

    async def test_existing_layer_reported(self):
        result = await self._create_existing("preset:sepa")

        assert result["status"] == "exists"
        assert result["changes"] == []

    async def test_existing_layer_unchanged_not_written(self):
        await self._create_existing("preset:sepa")

        self.mock_update.assert_not_awaited()

    async def test_existing_layer_updated(self):
        result = await self._create_existing("preset:sepa", description="Old description")

        assert result["status"] == "updated"
        assert result["changes"] == ["description"]
        preset = load_preset(SEPA_PRESET)
        self.mock_update.assert_awaited_once_with(
            "postgresql://test", "sepa", claim_origin="preset:sepa",
            description=preset.description,
        )

    async def test_existing_layer_backfills_origin(self):
        result = await self._create_existing(None, routing_patterns=["old"])

        assert result["status"] == "updated"
        assert result["changes"] == ["routing_patterns"]
        # One write: the origin and the changed settings together
        preset = load_preset(SEPA_PRESET)
        self.mock_update.assert_awaited_once_with(
            "postgresql://test", "sepa", claim_origin="preset:sepa",
            routing_patterns=preset.routing_patterns,
        )

    async def test_existing_layer_without_origin_unchanged_claims_origin(self):
        result = await self._create_existing(None)

        assert result["status"] == "exists"
        assert result["changes"] == []
        self.mock_update.assert_awaited_once_with(
            "postgresql://test", "sepa", claim_origin="preset:sepa"
        )

    async def test_origin_taken_during_apply_refused(self):
        from stratakb.knowledge.layers import LayerOriginError

        result = await self._create_existing(
            None, description="Old", update_error=LayerOriginError("taken", "manual")
        )

        assert result["status"] == "refused"
        assert result["changes"] == []
        assert "origin 'manual'" in result["message"]

    @pytest.mark.parametrize("origin", ["manual", "folder", "preset:other"])
    async def test_existing_layer_other_origin_refused(self, origin):
        result = await self._create_existing(origin, description="Old")

        assert result["status"] == "refused"
        assert result["changes"] == []
        assert f"origin '{origin}'" in result["message"]
        self.mock_update.assert_not_awaited()

    async def test_missing_answer_mode_keeps_layer_mode(self):
        result = await self._create_existing("preset:sepa", existing_mode="extractive")

        assert result["status"] == "exists"
        self.mock_update.assert_not_awaited()

    async def test_existing_layer_takes_preset_answer_mode(self):
        preset = load_preset(SEPA_PRESET)
        preset.answer_mode = "extractive"
        result = await self._create_existing("preset:sepa", preset=preset)

        assert result["changes"] == ["answer_mode"]
        self.mock_update.assert_awaited_once_with(
            "postgresql://test", "sepa", claim_origin="preset:sepa", answer_mode="extractive"
        )

    async def test_answer_mode_extractive_stored(self, tmp_path):
        import contextlib

        from stratakb.config import get_settings
        from stratakb.knowledge.layers import LayerNotFoundError, delete_layer, list_layers

        db_url = get_settings().database_url
        name = "test_preset_extractive"
        path = _write_preset(tmp_path, (
            f"name: {name}\n"
            "display_name: Extractive\n"
            "answer_mode: extractive\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        preset = load_preset(path)
        assert preset.answer_mode == "extractive"
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)
        try:
            result = await create_preset_layer(db_url, preset)
            assert result["layer"]["answer_mode"] == "extractive"
            listed = next(layer for layer in await list_layers(db_url) if layer["name"] == name)
            assert listed["answer_mode"] == "extractive"
        finally:
            with contextlib.suppress(LayerNotFoundError):
                await delete_layer(db_url, name=name)


RUNBOOKS_YAML = (
    "name: {name}\n"
    "display_name: Ops Runbooks\n"
    "description: {description}\n"
    "chunk_config: {{chunk_size: {chunk_size}, chunk_overlap: 100}}\n"
    "routing_patterns: [{pattern}]\n"
    "sources:\n"
    "  - {{name: docs, path: docs}}\n"
)


class TestReapplyPresetIntegration:
    """Applying a preset again brings its layer up to the YAML (require database)."""

    @pytest.fixture
    async def db_url(self):
        import contextlib

        from stratakb.config import get_settings
        from stratakb.knowledge.layers import LayerNotFoundError, delete_layer

        db_url = get_settings().database_url
        yield db_url
        for name in ("test_preset_reapply", "test_preset_manual", "test_preset_race"):
            with contextlib.suppress(LayerNotFoundError):
                await delete_layer(db_url, name=name)

    async def test_reapply_updates_stored_settings(self, db_url, tmp_path):
        import contextlib

        from stratakb.knowledge.layers import (
            LayerNotFoundError,
            delete_layer,
            get_layer_status,
            set_layer_answer_mode,
        )

        name = "test_preset_reapply"
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)
        path = _write_preset(tmp_path, RUNBOOKS_YAML.format(
            name=name, description="First", chunk_size=800, pattern="'\\bops\\b'",
        ))
        assert (await create_preset_layer(db_url, load_preset(path)))["status"] == "created"
        # A mode set another way survives a preset that leaves answer_mode out
        await set_layer_answer_mode(db_url, name, "extractive")

        _write_preset(tmp_path, RUNBOOKS_YAML.format(
            name=name, description="Second", chunk_size=1200, pattern="'\\brunbook\\b'",
        ))
        result = await create_preset_layer(db_url, load_preset(path))
        assert result["status"] == "updated"
        assert result["changes"] == ["description", "chunk_config", "routing_patterns"]

        stored = await get_layer_status(db_url, name=name)
        assert stored["description"] == "Second"
        assert stored["chunk_config"]["chunk_size"] == 1200
        assert stored["routing_patterns"] == ["\\brunbook\\b"]
        assert stored["answer_mode"] == "extractive"
        assert stored["origin"] == f"preset:{name}"

        again = await create_preset_layer(db_url, load_preset(path))
        assert again["status"] == "exists"
        assert again["changes"] == []

    async def test_reapply_refuses_manual_layer(self, db_url, tmp_path):
        import contextlib

        from stratakb.knowledge.layers import (
            LayerNotFoundError,
            create_layer,
            delete_layer,
            get_layer_status,
        )

        name = "test_preset_manual"
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)
        await create_layer(
            db_url, name=name, display_name="Manual", description="Kept", origin="manual"
        )
        before = await get_layer_status(db_url, name=name)
        path = _write_preset(tmp_path, RUNBOOKS_YAML.format(
            name=name, description="Replaced", chunk_size=800, pattern="ops",
        ))

        result = await create_preset_layer(db_url, load_preset(path))

        assert result["status"] == "refused"
        after = await get_layer_status(db_url, name=name)
        assert after == before

    async def test_concurrent_set_origin_not_overwritten(self, db_url, tmp_path):
        import contextlib

        from stratakb.knowledge import layers
        from stratakb.knowledge.layers import (
            LayerNotFoundError,
            create_layer,
            delete_layer,
            get_layer,
            set_origin,
        )

        name = "test_preset_race"
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)
        # Made before origins were recorded: the preset may adopt it
        await create_layer(db_url, name=name, display_name="Race", description="Kept")
        path = _write_preset(tmp_path, RUNBOOKS_YAML.format(
            name=name, description="Replaced", chunk_size=800, pattern="ops",
        ))

        async def read_then_race(database_url, *, name):
            stale = await layers.get_layer(database_url, name=name)
            # Another process records an origin between the preset's read and its write
            await set_origin(database_url, name, "manual")
            return stale

        with patch("stratakb.knowledge.presets.get_layer", side_effect=read_then_race):
            result = await create_preset_layer(db_url, load_preset(path))

        assert result["status"] == "refused"
        assert "origin 'manual'" in result["message"]
        stored = await get_layer(db_url, name=name)
        assert stored["origin"] == "manual"
        assert stored["description"] == "Kept"


class TestPresetLayerChanges:
    """preset_layer_changes compares stored settings with the preset (no database)."""

    def test_nothing_changed(self):
        from stratakb.knowledge.presets import preset_layer_changes

        preset = load_preset(SEPA_PRESET)
        assert preset_layer_changes(_stored_layer(preset), preset) == []

    @pytest.mark.parametrize("key, stored", [
        ("display_name", "Old name"),
        ("description", "Old description"),
        ("metadata_schema", {"old": True}),
        ("chunk_config", {"chunk_size": 1, "chunk_overlap": 0, "splitting_strategy": "x"}),
        ("routing_patterns", ["old"]),
        ("structure_patterns", []),
    ])
    def test_one_setting_changed(self, key, stored):
        from stratakb.knowledge.presets import preset_layer_changes

        preset = load_preset(SEPA_PRESET)
        assert preset_layer_changes(_stored_layer(preset, **{key: stored}), preset) == [key]

    def test_display_name_compared_stripped(self, tmp_path):
        from stratakb.knowledge.presets import preset_layer_changes

        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: '  Ops Runbooks  '\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        preset = load_preset(path)
        stored = _stored_layer(preset, display_name="Ops Runbooks")
        assert preset_layer_changes(stored, preset) == []

    def test_answer_mode_none_never_reported(self):
        from stratakb.knowledge.presets import preset_layer_changes

        preset = load_preset(SEPA_PRESET)
        assert preset.answer_mode is None
        stored = _stored_layer(preset, answer_mode="extractive")
        assert preset_layer_changes(stored, preset) == []
        preset.answer_mode = "grounded"
        assert preset_layer_changes(stored, preset) == ["answer_mode"]


class TestApplyPresetTool:
    """strata_apply_preset refuses another origin and reports the layer check."""

    @staticmethod
    def _admin():
        return (
            patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
            patch("stratakb.knowledge.tools.get_current_user", return_value=("tester", "admin")),
        )

    async def _call(self, **kwargs):
        from stratakb.mcp_server import mcp

        tool = mcp._tool_manager.get_tool("strata_apply_preset")
        return await tool.fn(preset_path=str(SEPA_PRESET), **kwargs)

    async def test_other_origin_returns_error_without_ingest(self):
        auth, tools_auth = self._admin()
        with (
            auth, tools_auth,
            patch("stratakb.knowledge.embeddings.embedding_key_error", return_value=None),
            patch(
                "stratakb.knowledge.presets.create_preset_layer",
                new_callable=AsyncMock,
                return_value={"status": "refused", "changes": [], "message": "not ours"},
            ),
            patch("stratakb.knowledge.presets.ingest_preset", new_callable=AsyncMock) as ingest,
            patch("stratakb.knowledge.tools.write_audit_log", new_callable=AsyncMock) as audit,
        ):
            result = await self._call()

        assert result == {"status": "error", "error": "not ours"}
        ingest.assert_not_awaited()
        audit.assert_not_awaited()

    async def test_layer_error_returns_error(self):
        auth, tools_auth = self._admin()
        with (
            auth, tools_auth,
            patch("stratakb.knowledge.embeddings.embedding_key_error", return_value=None),
            patch(
                "stratakb.knowledge.presets.create_preset_layer",
                new_callable=AsyncMock,
                side_effect=LayerValidationError("display_name is required."),
            ),
            patch("stratakb.knowledge.presets.ingest_preset", new_callable=AsyncMock) as ingest,
            patch("stratakb.knowledge.tools.write_audit_log", new_callable=AsyncMock) as audit,
        ):
            result = await self._call()

        assert result == {"status": "error", "error": "display_name is required."}
        ingest.assert_not_awaited()
        audit.assert_not_awaited()

    async def test_real_run_reports_layer_changes(self):
        auth, tools_auth = self._admin()
        with (
            auth, tools_auth,
            patch("stratakb.knowledge.embeddings.embedding_key_error", return_value=None),
            patch(
                "stratakb.knowledge.presets.create_preset_layer",
                new_callable=AsyncMock,
                return_value={"status": "updated", "changes": ["description"], "message": ""},
            ),
            patch(
                "stratakb.knowledge.presets.ingest_preset",
                new_callable=AsyncMock,
                return_value={"total_docs": 0, "total_chunks": 0},
            ),
            patch("stratakb.knowledge.tools.write_audit_log", new_callable=AsyncMock),
        ):
            result = await self._call()

        assert result["status"] == "completed"
        assert result["layer"] == "updated"
        assert result["layer_changes"] == ["description"]

    async def _audited_run(self, layer_result):
        auth, tools_auth = self._admin()
        with (
            auth, tools_auth,
            patch("stratakb.knowledge.embeddings.embedding_key_error", return_value=None),
            patch(
                "stratakb.knowledge.presets.create_preset_layer",
                new_callable=AsyncMock,
                return_value=layer_result,
            ),
            patch(
                "stratakb.knowledge.presets.ingest_preset",
                new_callable=AsyncMock,
                return_value={"total_docs": 4, "total_chunks": 31},
            ),
            patch("stratakb.knowledge.tools.write_audit_log", new_callable=AsyncMock) as audit,
        ):
            await self._call()
        audit.assert_awaited_once()
        return audit.await_args.kwargs["result_summary"]

    async def test_audit_names_changed_settings(self):
        summary = await self._audited_run(
            {"status": "updated", "changes": ["description", "routing_patterns"], "message": ""}
        )
        assert summary == (
            "Layer 'sepa': updated (description, routing_patterns), ingested 4 docs, 31 chunks"
        )

    async def test_audit_has_no_setting_values(self):
        description = load_preset(SEPA_PRESET).description
        assert description
        summary = await self._audited_run(
            {"status": "updated", "changes": ["description"], "message": "ZZ-old-description-7f3"}
        )
        assert "(description)" in summary
        assert "ZZ-old-description-7f3" not in summary
        assert description not in summary

    @pytest.mark.parametrize("status", ["created", "exists"])
    async def test_audit_unchanged_without_changes(self, status):
        summary = await self._audited_run({"status": status, "changes": [], "message": ""})
        assert summary == f"Layer 'sepa': {status}, ingested 4 docs, 31 chunks"

    async def _dry_run(self, stored=None, side_effect=None):
        auth, tools_auth = self._admin()
        with (
            auth, tools_auth,
            patch(
                "stratakb.knowledge.presets.get_layer",
                new_callable=AsyncMock,
                return_value=stored,
                side_effect=side_effect,
            ),
        ):
            return await self._call(dry_run=True)

    async def test_dry_run_would_update(self):
        preset = load_preset(SEPA_PRESET)
        stored = _stored_layer(
            preset, origin="preset:sepa", description="Old", routing_patterns=["old"]
        )
        result = await self._dry_run(stored)

        assert result["status"] == "dry_run_complete"
        assert result["layer_check"] == {
            "status": "would_update", "changes": ["description", "routing_patterns"],
        }

    async def test_dry_run_would_claim(self):
        # The real run writes the origin of a matching layer that has none
        preset = load_preset(SEPA_PRESET)
        result = await self._dry_run(_stored_layer(preset))

        assert result["layer_check"] == {"status": "would_claim", "changes": []}

    async def test_dry_run_unclaimed_would_update(self):
        preset = load_preset(SEPA_PRESET)
        result = await self._dry_run(_stored_layer(preset, description="Old"))

        assert result["layer_check"] == {"status": "would_update", "changes": ["description"]}

    async def test_dry_run_own_layer_unchanged(self):
        preset = load_preset(SEPA_PRESET)
        result = await self._dry_run(_stored_layer(preset, origin="preset:sepa"))

        assert result["layer_check"] == {"status": "unchanged", "changes": []}

    async def test_dry_run_would_create(self):
        from stratakb.knowledge.layers import LayerNotFoundError

        result = await self._dry_run(side_effect=LayerNotFoundError("missing"))

        assert result["layer_check"] == {"status": "would_create", "changes": []}

    async def test_dry_run_refused(self):
        result = await self._dry_run({"origin": "manual"})

        assert result["layer_check"]["status"] == "refused"
        assert result["layer_check"]["changes"] == []

    async def test_dry_run_unknown_when_lookup_fails(self):
        result = await self._dry_run(side_effect=OSError("connection refused"))

        assert result["layer_check"] == {"status": "unknown", "changes": []}

    async def test_dry_run_check_times_out(self):
        import asyncio
        import time

        async def hang(*args, **kwargs):
            await asyncio.sleep(5)

        with patch("stratakb.knowledge.tools._LAYER_CHECK_TIMEOUT_S", 0.05):
            started = time.monotonic()
            result = await self._dry_run(side_effect=hang)
            elapsed = time.monotonic() - started

        assert result["layer_check"] == {"status": "unknown", "changes": []}
        assert elapsed < 2


class TestApplyPresetScript:
    """scripts/apply_preset.py reports layer changes and exits 1 on a refused layer."""

    @staticmethod
    def _run(layer_result, capsys):
        import importlib.util
        import json
        import sys

        script = Path(__file__).parent.parent / "scripts" / "apply_preset.py"
        spec = importlib.util.spec_from_file_location("apply_preset_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        code = 0
        with (
            patch.object(sys, "argv", ["apply_preset.py", str(SEPA_PRESET)]),
            patch.object(module, "configure_logging"),
            patch.object(module, "close_pool", new_callable=AsyncMock),
            patch.object(
                module, "create_preset_layer", new_callable=AsyncMock, return_value=layer_result
            ),
            patch.object(
                module, "ingest_preset", new_callable=AsyncMock, return_value={"total_docs": 0}
            ) as ingest,
        ):
            try:
                module.main()
            except SystemExit as e:
                code = e.code
        return code, json.loads(capsys.readouterr().out), ingest

    def test_reports_layer_changes(self, capsys):
        code, out, ingest = self._run(
            {"status": "updated", "changes": ["chunk_config"], "message": ""}, capsys
        )
        assert code == 0
        assert out["layer"] == "updated"
        assert out["layer_changes"] == ["chunk_config"]
        ingest.assert_awaited_once()

    def test_refused_exits_1_without_ingest(self, capsys):
        code, out, ingest = self._run(
            {"status": "refused", "changes": [], "message": "not ours"}, capsys
        )
        assert code == 1
        assert out == {"layer": "refused", "error": "not ours"}
        ingest.assert_not_awaited()


class TestPresetToolRegistration:
    def test_apply_preset_tool_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_apply_preset" in tool_names
        assert "strata_setup_sepa_layer" not in tool_names
