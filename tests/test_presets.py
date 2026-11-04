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

    def test_answer_mode_defaults_to_grounded(self, tmp_path):
        path = _write_preset(tmp_path, (
            "name: runbooks\n"
            "display_name: Ops Runbooks\n"
            "sources:\n"
            "  - {name: docs, path: docs}\n"
        ))
        assert load_preset(path).answer_mode == "grounded"

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

    async def _create_existing(self, existing_origin, existing_mode="grounded", preset=None):
        from stratakb.knowledge.layers import LayerExistsError

        preset = preset or load_preset(SEPA_PRESET)
        with (
            patch(
                "stratakb.knowledge.presets.create_layer",
                new_callable=AsyncMock,
                side_effect=LayerExistsError("exists"),
            ),
            patch(
                "stratakb.knowledge.presets.get_layer_status",
                new_callable=AsyncMock,
                return_value={
                    "name": "sepa", "origin": existing_origin, "answer_mode": existing_mode,
                },
            ),
            patch(
                "stratakb.knowledge.presets.set_origin", new_callable=AsyncMock
            ) as mock_set_origin,
            patch(
                "stratakb.knowledge.presets.set_layer_answer_mode", new_callable=AsyncMock
            ) as mock_set_mode,
        ):
            result = await create_preset_layer("postgresql://test", preset)
        self.mock_set_mode = mock_set_mode
        return result, mock_set_origin

    async def test_existing_layer_reported(self):
        result, _ = await self._create_existing("preset:sepa")

        assert result["status"] == "exists"

    async def test_existing_layer_backfills_origin(self):
        result, mock_set_origin = await self._create_existing(None)

        assert result["status"] == "exists"
        mock_set_origin.assert_awaited_once_with("postgresql://test", "sepa", "preset:sepa")

    async def test_existing_layer_keeps_other_origin(self):
        result, mock_set_origin = await self._create_existing("manual")

        assert result["status"] == "exists"
        mock_set_origin.assert_not_awaited()

    async def test_existing_layer_keeps_same_answer_mode(self):
        await self._create_existing("preset:sepa")

        self.mock_set_mode.assert_not_awaited()

    async def test_existing_layer_takes_preset_answer_mode(self):
        preset = load_preset(SEPA_PRESET)
        preset.answer_mode = "extractive"
        await self._create_existing("preset:sepa", preset=preset)

        self.mock_set_mode.assert_awaited_once_with("postgresql://test", "sepa", "extractive")

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


class TestPresetToolRegistration:
    def test_apply_preset_tool_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_apply_preset" in tool_names
        assert "strata_setup_sepa_layer" not in tool_names
