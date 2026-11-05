"""Tests for ingest errors and logs: no absolute paths, short messages.

All tests need the database; embeddings are mocked.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.test_ingestion_sync import FAKE_EMBEDDING, _ingest, _outcomes, _write


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def layer(db_url):
    """A fresh layer with a unique name, deleted afterwards."""
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    name = f"test_errors_{uuid4().hex[:10]}"
    await create_layer(
        db_url,
        name=name,
        display_name="Test Errors",
        chunk_config={"chunk_size": 500, "chunk_overlap": 50},
    )
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


def _denied_read_bytes(name: str):
    """A Path.read_bytes that raises a real-looking PermissionError, naming
    the absolute path, for files called name."""
    original_read = Path.read_bytes

    def read_bytes(self):
        if self.name == name:
            raise PermissionError(13, "Permission denied", str(self))
        return original_read(self)

    return read_bytes


async def test_tool_reports_read_error(layer, tmp_path):
    from stratakb.mcp_server import mcp

    _write(tmp_path / "locked.md", "# Locked\n\nLocked content.\n")
    _write(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
    with (
        patch("stratakb.core.auth.get_current_user", return_value=("tester", "admin")),
        patch("stratakb.knowledge.tools.get_current_user", return_value=("tester", "admin")),
        patch("stratakb.knowledge.tools.write_audit_log", new_callable=AsyncMock),
        patch(
            "stratakb.knowledge.ingestion.embed_texts",
            new_callable=AsyncMock,
            side_effect=lambda texts, **_: [FAKE_EMBEDDING for _ in texts],
        ),
        patch.object(Path, "read_bytes", _denied_read_bytes("locked.md")),
    ):
        tool = mcp._tool_manager.get_tool("strata_ingest_documents")
        result = await tool.fn(layer_name=layer, source_path=str(tmp_path))

    assert result["status"] == "partial"
    assert result["errors"] == [{
        "file": "locked.md",
        "error": "[Errno 13] Permission denied: 'locked.md'",
        "reason": "read_error",
    }]
    assert str(tmp_path.resolve()) not in json.dumps(result)


async def _last_error_log(db_url: str, layer: str) -> list:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    value = await pool.fetchval(
        """
        SELECT r.error_log FROM strata_core.ingestion_runs r
        JOIN strata_core.knowledge_layers l ON l.id = r.layer_id
        WHERE l.name = $1 ORDER BY r.completed_at DESC LIMIT 1
        """,
        layer,
    )
    return json.loads(value) if isinstance(value, str) else value


class TestErrorsWithoutAbsolutePaths:
    async def test_read_error_names_the_relative_path(self, db_url, layer, tmp_path):
        _write(tmp_path / "sub" / "locked.md", "# Locked\n\nLocked content.\n")

        with patch.object(Path, "read_bytes", _denied_read_bytes("locked.md")):
            result, _ = await _ingest(db_url, layer, tmp_path)

        expected = [{
            "file": "sub/locked.md",
            "error": "[Errno 13] Permission denied: 'sub/locked.md'",
            "reason": "read_error",
        }]
        assert result.errors == expected
        assert await _last_error_log(db_url, layer) == expected

    async def test_parse_error_names_the_relative_path(self, db_url, layer, tmp_path):
        from stratakb.knowledge.parsers.pdf_parser import ParsedDocument

        _write(tmp_path / "bad.md", "# Bad\n\nBad content.\n")
        root = tmp_path.resolve()

        def failing_parse(file_path):
            return ParsedDocument(
                source_file=file_path.name,
                error=f"Not readable: {file_path} (under {root})",
            )

        with patch("stratakb.knowledge.ingestion._parse_file", side_effect=failing_parse):
            result, _ = await _ingest(db_url, layer, tmp_path)

        assert result.errors == [
            {"file": "bad.md", "error": "Not readable: bad.md (under .)", "reason": "parse_error"}
        ]
        assert str(root) not in json.dumps(await _last_error_log(db_url, layer))

    async def test_long_error_is_cut(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import _ERROR_MAX_CHARS

        _write(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        with patch(
            "stratakb.knowledge.ingestion._store_file",
            new_callable=AsyncMock,
            side_effect=RuntimeError("x" * 1000),
        ):
            result, _ = await _ingest(db_url, layer, tmp_path)

        error = result.errors[0]["error"]
        assert len(error) == _ERROR_MAX_CHARS
        assert error.endswith("...")

    async def test_embed_and_store_failures_log_no_exception_text(
        self, db_url, layer, tmp_path
    ):
        from structlog.testing import capture_logs

        from stratakb.knowledge import ingestion

        marker = "distinctive-7f3a-exception-text"
        _write(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        _write(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
        original_store = ingestion._store_file

        async def failing_store(*args, **kwargs):
            if kwargs["relative_path"] == "b.md":
                raise RuntimeError(f"store {marker}")
            return await original_store(*args, **kwargs)

        async def failing_embed(texts, **_):
            if any("Alpha" in t for t in texts):
                raise RuntimeError(f"embed {marker}")
            return [FAKE_EMBEDDING for _ in texts]

        with (
            capture_logs() as logs,
            patch("stratakb.knowledge.ingestion._store_file", side_effect=failing_store),
            patch("stratakb.knowledge.ingestion.embed_texts", side_effect=failing_embed),
        ):
            result = await ingestion.ingest_documents(
                db_url, "fake-key", layer_name=layer, source_path=str(tmp_path)
            )

        assert _outcomes(result) == {
            "a.md": ("failed", "embed_error"), "b.md": ("failed", "embed_error"),
        }
        failed = [e for e in logs if e["event"] == "file_ingestion_failed"]
        assert [(e["file"], e["error_type"]) for e in failed] == [
            ("a.md", "RuntimeError"), ("b.md", "RuntimeError"),
        ]
        assert marker not in repr(logs)
