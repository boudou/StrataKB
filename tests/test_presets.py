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


def _write_preset(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "preset.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestLoadPreset:
    @pytest.mark.parametrize("path", [SEPA_PRESET, ISO_PRESET])
    def test_example_presets_load(self, path):
        preset = load_preset(path)
        assert preset.name in ("sepa", "iso20022")
        assert preset.display_name
        assert preset.description
        assert preset.routing_patterns
        assert preset.sources
        assert preset.root == path.parent

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

    async def test_existing_layer_reported(self):
        from stratakb.knowledge.layers import LayerExistsError

        preset = load_preset(SEPA_PRESET)
        with patch(
            "stratakb.knowledge.presets.create_layer",
            new_callable=AsyncMock,
            side_effect=LayerExistsError("exists"),
        ):
            result = await create_preset_layer("postgresql://test", preset)

        assert result["status"] == "exists"


class TestPresetToolRegistration:
    def test_apply_preset_tool_registered(self):
        from stratakb.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "strata_apply_preset" in tool_names
        assert "strata_setup_sepa_layer" not in tool_names
