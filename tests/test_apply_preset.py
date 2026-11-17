"""Tests for scripts/apply_preset.py: the same apply flow and audit entries as the tool."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from structlog.testing import capture_logs

from stratakb.config import Settings
from stratakb.knowledge.layers import LayerValidationError
from tests.helpers import SEPA_PRESET, capture_tools, openai_settings

SCRIPT = Path(__file__).parent.parent / "scripts" / "apply_preset.py"


def _local_settings() -> Settings:
    return Settings(_env_file=None, embedding_backend="local")


def _write_preset(tmp_path: Path, line: str = "['old/']") -> Path:
    """A preset file with an archived_patterns line."""
    path = tmp_path / "preset.yaml"
    path.write_text(
        f"name: archived_preset\ndisplay_name: Archived Preset\narchived_patterns: {line}\n"
        "sources:\n  - {name: docs, path: docs}\n",
        encoding="utf-8",
    )
    return path


class TestApplyPresetScript:
    """scripts/apply_preset.py reports layer changes and exits 1 on a refused layer."""

    @staticmethod
    def _run(
        layer_result,
        capsys,
        settings=None,
        extra_args=(),
        preset=SEPA_PRESET,
        create_side_effect=None,
    ):
        if settings is None:
            settings = _local_settings()
        spec = importlib.util.spec_from_file_location("apply_preset_script", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if create_side_effect is not None:
            create = AsyncMock(side_effect=create_side_effect)
        else:
            create = AsyncMock(return_value=layer_result)
        ingest = AsyncMock(return_value={"total_docs": 0, "total_chunks": 0})
        audit = AsyncMock()
        code = 0
        with (
            patch.object(sys, "argv", ["apply_preset.py", str(preset), *extra_args]),
            # The script's settings, not the shell's config
            patch.object(module, "get_settings", return_value=settings),
            patch.object(module, "configure_logging"),
            # Keeps the parsers' log lines off stdout, which holds only the JSON
            capture_logs(),
            patch.object(module, "close_pool", new_callable=AsyncMock),
            patch("stratakb.knowledge.presets.create_preset_layer", create),
            patch("stratakb.knowledge.presets.ingest_preset", ingest),
            patch("stratakb.knowledge.presets.write_audit_log", audit),
        ):
            try:
                module.main()
            except SystemExit as e:
                code = e.code
        captured = capsys.readouterr()
        return code, json.loads(captured.out), ingest, create, audit, captured.err

    def test_reports_layer_changes(self, capsys):
        settings = _local_settings()
        code, out, ingest, _, _, _ = self._run(
            {"status": "updated", "changes": ["chunk_config"], "message": ""},
            capsys,
            settings=settings,
        )
        assert code == 0
        assert out["layer"] == "updated"
        assert out["layer_changes"] == ["chunk_config"]
        ingest.assert_awaited_once()
        # The check and the embedding use the same settings
        assert ingest.await_args.kwargs["settings"] is settings

    def test_refused_exits_1_without_ingest(self, capsys):
        code, out, ingest, _, _, _ = self._run(
            {"status": "refused", "changes": [], "message": "not ours"}, capsys
        )
        assert code == 1
        assert out == {"layer": "refused", "error": "not ours"}
        ingest.assert_not_awaited()

    def test_missing_key_exits_1_before_any_write(self, capsys):
        code, out, ingest, create, audit, _ = self._run(
            {"status": "created", "changes": [], "message": ""},
            capsys,
            settings=openai_settings(openai_api_key=""),
        )
        assert code == 1
        assert out["layer"] == "refused"
        assert "API key" in out["error"]
        create.assert_not_awaited()
        ingest.assert_not_awaited()
        audit.assert_not_awaited()

    def test_dry_run_needs_no_key(self, capsys):
        code, out, _, create, audit, _ = self._run(
            {"status": "created", "changes": [], "message": ""},
            capsys,
            settings=openai_settings(openai_api_key=""),
            extra_args=("--dry-run",),
        )
        assert code == 0
        assert out["layer"] != "refused"
        create.assert_not_awaited()
        audit.assert_not_awaited()

    def test_script_writes_apply_preset_entry(self, capsys):
        code, _, _, _, audit, _ = self._run(
            {"status": "updated", "changes": ["description"], "message": ""}, capsys
        )
        assert code == 0
        audit.assert_awaited_once()
        assert audit.await_args.kwargs == {
            "action": "apply_preset",
            "tool_name": "script:apply_preset",
            "parameters": {"preset_path": str(SEPA_PRESET), "docs_root": None},
            "result_summary": "Layer 'sepa': updated (description), ingested 0 docs, 0 chunks",
        }

    def test_script_writes_archived_patterns_entry(self, capsys, tmp_path):
        code, _, _, _, audit, _ = self._run(
            {"status": "updated", "changes": ["archived_patterns"], "message": ""},
            capsys,
            preset=_write_preset(tmp_path),
        )
        assert code == 0
        assert [c.kwargs["action"] for c in audit.await_args_list] == [
            "set_archived_patterns",
            "apply_preset",
        ]
        assert audit.await_args_list[0].kwargs == {
            "action": "set_archived_patterns",
            "tool_name": "script:apply_preset",
            "parameters": {"layer": "archived_preset", "patterns": ["old/"]},
        }

    async def test_script_and_tool_write_the_same_entries(self, capsys, tmp_path):
        from stratakb.knowledge.tools import register_knowledge_tools

        path = _write_preset(tmp_path)
        layer_result = {"status": "updated", "changes": ["archived_patterns"], "message": ""}
        # The script runs its own event loop, so it goes in a thread
        _, _, _, _, script_audit, _ = await asyncio.to_thread(
            self._run, layer_result, capsys, preset=path
        )

        fn = capture_tools(register_knowledge_tools, "postgresql://test", "")[
            "strata_apply_preset"
        ]
        tool_audit = AsyncMock()
        with (
            patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
            patch("stratakb.config.get_settings", return_value=_local_settings()),
            patch(
                "stratakb.knowledge.presets.create_preset_layer",
                AsyncMock(return_value=layer_result),
            ),
            patch(
                "stratakb.knowledge.presets.ingest_preset",
                AsyncMock(return_value={"total_docs": 0, "total_chunks": 0}),
            ),
            patch("stratakb.knowledge.presets.write_audit_log", tool_audit),
        ):
            result = await fn(preset_path=str(path))
        assert result["status"] == "completed"

        def entries(audit, tool_name):
            calls = [dict(c.kwargs) for c in audit.await_args_list]
            assert [c.pop("tool_name") for c in calls] == [tool_name, tool_name]
            return calls

        assert entries(script_audit, "script:apply_preset") == entries(
            tool_audit, "strata_apply_preset"
        )

    def test_script_refused_writes_no_audit(self, capsys):
        code, out, ingest, _, audit, err = self._run(
            {"status": "refused", "changes": [], "message": "not ours"}, capsys
        )
        assert code == 1
        assert out == {"layer": "refused", "error": "not ours"}
        audit.assert_not_awaited()
        ingest.assert_not_awaited()
        assert "Traceback" not in err

    def test_script_layer_error_exits_1(self, capsys):
        code, out, ingest, _, audit, err = self._run(
            None, capsys, create_side_effect=LayerValidationError("display_name is required.")
        )
        assert code == 1
        assert out == {"layer": "refused", "error": "display_name is required."}
        audit.assert_not_awaited()
        ingest.assert_not_awaited()
        assert "Traceback" not in err

    def test_script_missing_preset_exits_1(self, capsys, tmp_path):
        path = tmp_path / "missing.yaml"
        code, out, _, create, audit, _ = self._run(None, capsys, preset=path)
        assert code == 1
        assert out == {"layer": "refused", "error": f"Preset not found: {path}"}
        create.assert_not_awaited()
        audit.assert_not_awaited()


def _bad_inputs():
    """Bad inputs as (id, build): build(tmp_path) gives the preset path and docs root."""

    def write(tmp_path, data):
        path = tmp_path / "preset.yaml"
        path.write_bytes(data)
        return str(path), None

    def file_root(tmp_path):
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")
        return str(SEPA_PRESET), str(tmp_path / "file.txt")

    sources = b"sources:\n  - {name: docs, path: docs}\n"
    return [
        ("missing file", lambda t: (str(t / "missing.yaml"), None)),
        ("folder", lambda t: (str(t), None)),
        ("not utf-8", lambda t: write(t, b"name: caf\xe9\n")),
        ("invalid name", lambda t: write(t, b"name: Ops-Runbooks\ndisplay_name: Ops\n" + sources)),
        ("name not a string", lambda t: write(t, b"name: 123\ndisplay_name: Ops\n" + sources)),
        ("sources not a list", lambda t: write(t, b"name: ops\ndisplay_name: Ops\nsources: 5\n")),
        ("missing docs root", lambda t: (str(SEPA_PRESET), str(t / "nowhere"))),
        ("docs root is a file", file_root),
    ]


class TestToolAndScriptRefuseAlike:
    """The tool and the script run the same input checks, with the same refusal text."""

    @staticmethod
    async def _tool(layer_result=None, check=None, **kwargs):
        from stratakb.knowledge.tools import register_knowledge_tools

        fn = capture_tools(register_knowledge_tools, "postgresql://test", "")[
            "strata_apply_preset"
        ]
        create = AsyncMock(return_value=layer_result)
        ingest = AsyncMock(return_value={"total_docs": 0, "total_chunks": 0})
        audit = AsyncMock()
        check = check or AsyncMock(return_value={"status": "would_create", "changes": []})
        with (
            patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
            patch("stratakb.config.get_settings", return_value=_local_settings()),
            patch("stratakb.knowledge.presets.create_preset_layer", create),
            patch("stratakb.knowledge.presets.ingest_preset", ingest),
            patch("stratakb.knowledge.presets.write_audit_log", audit),
            patch("stratakb.knowledge.presets.check_preset_layer", check),
            capture_logs(),
        ):
            result = await fn(**kwargs)
        return result, create, ingest, audit

    @pytest.mark.parametrize("dry_run", [False, True], ids=["apply", "dry_run"])
    @pytest.mark.parametrize(
        "build", [b for _, b in _bad_inputs()], ids=[i for i, _ in _bad_inputs()]
    )
    async def test_same_refusal(self, capsys, tmp_path, build, dry_run):
        preset_path, docs_root = build(tmp_path)
        args = []
        if docs_root is not None:
            args += ["--docs-root", docs_root]
        if dry_run:
            args.append("--dry-run")
        code, out, s_ingest, s_create, s_audit, err = await asyncio.to_thread(
            TestApplyPresetScript._run, None, capsys, preset=preset_path, extra_args=args
        )
        tool, t_create, t_ingest, t_audit = await self._tool(
            preset_path=preset_path, docs_root=docs_root, dry_run=dry_run
        )

        assert tool["status"] == "error"
        assert out == {"layer": "refused", "error": tool["error"]}
        assert code == 1
        assert "Traceback" not in err
        for mock in (s_create, s_ingest, s_audit, t_create, t_ingest, t_audit):
            mock.assert_not_awaited()

    async def test_same_dry_run_report(self, capsys):
        code, out, *_ = await asyncio.to_thread(
            TestApplyPresetScript._run, None, capsys, extra_args=("--dry-run",)
        )
        tool, *_ = await self._tool(preset_path=str(SEPA_PRESET), dry_run=True)

        assert code == 0
        assert tool.pop("layer_check") == {"status": "would_create", "changes": []}
        assert json.loads(json.dumps(tool)) == out
        assert out["status"] == "dry_run_complete"
        assert out["layer"] == "sepa"
        assert out["group_stats"]

    def test_dry_run_of_a_preset_named_refused_exits_0(self, capsys, tmp_path):
        path = tmp_path / "preset.yaml"
        path.write_text(
            "name: refused\ndisplay_name: Refused\nsources:\n  - {name: docs, path: docs}\n",
            encoding="utf-8",
        )
        code, out, *_ = TestApplyPresetScript._run(
            None, capsys, preset=path, extra_args=("--dry-run",)
        )
        assert code == 0
        assert out["layer"] == "refused"
        assert out["status"] == "dry_run_complete"
