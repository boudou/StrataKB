"""Tests for ingest errors and logs: no absolute paths, short messages.

All tests need the database; embeddings are mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tests.helpers import (
    FAKE_EMBEDDING,
    as_admin,
    call_tool,
    fake_openai_client,
    file_outcomes,
    ingest,
    openai_settings,
    write_file,
)


def _denied_read_bytes(name: str):
    """A Path.read_bytes that raises a real-looking PermissionError, naming
    the absolute path, for files called name."""
    original_read = Path.read_bytes

    def read_bytes(self):
        if self.name == name:
            raise PermissionError(13, "Permission denied", str(self))
        return original_read(self)

    return read_bytes


async def test_tool_reports_read_error(server, layer, tmp_path):
    write_file(tmp_path / "locked.md", "# Locked\n\nLocked content.\n")
    write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
    with as_admin(), patch.object(Path, "read_bytes", _denied_read_bytes("locked.md")):
        result = await call_tool(
            server, "strata_ingest_documents", layer_name=layer, source_path=str(tmp_path)
        )

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
        write_file(tmp_path / "sub" / "locked.md", "# Locked\n\nLocked content.\n")

        with patch.object(Path, "read_bytes", _denied_read_bytes("locked.md")):
            result = await ingest(db_url, layer, tmp_path)

        expected = [{
            "file": "sub/locked.md",
            "error": "[Errno 13] Permission denied: 'sub/locked.md'",
            "reason": "read_error",
        }]
        assert result.errors == expected
        assert await _last_error_log(db_url, layer) == expected

    async def test_parse_error_names_the_relative_path(self, db_url, layer, tmp_path):
        from stratakb.knowledge.parsers.pdf_parser import ParsedDocument

        write_file(tmp_path / "bad.md", "# Bad\n\nBad content.\n")
        root = tmp_path.resolve()

        def failing_parse(file_path):
            return ParsedDocument(
                source_file=file_path.name,
                error=f"Not readable: {file_path} (under {root})",
            )

        with patch("stratakb.knowledge.ingestion._parse_file", side_effect=failing_parse):
            result = await ingest(db_url, layer, tmp_path)

        assert result.errors == [
            {"file": "bad.md", "error": "Not readable: bad.md (under .)", "reason": "parse_error"}
        ]
        assert str(root) not in json.dumps(await _last_error_log(db_url, layer))

    async def test_long_error_is_cut(self, db_url, layer, tmp_path):
        from stratakb.knowledge.ingestion import _ERROR_MAX_CHARS

        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")

        with patch(
            "stratakb.knowledge.ingestion._store_file",
            new_callable=AsyncMock,
            side_effect=RuntimeError("x" * 1000),
        ):
            result = await ingest(db_url, layer, tmp_path)

        error = result.errors[0]["error"]
        assert len(error) == _ERROR_MAX_CHARS
        assert error.endswith("...")

    async def test_embed_and_store_failures_log_no_exception_text(
        self, db_url, layer, tmp_path
    ):
        from structlog.testing import capture_logs

        from stratakb.knowledge import ingestion

        marker = "distinctive-7f3a-exception-text"
        write_file(tmp_path / "a.md", "# Alpha\n\nAlpha content.\n")
        write_file(tmp_path / "b.md", "# Beta\n\nBeta content.\n")
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

        assert file_outcomes(result) == {
            "a.md": ("failed", "embed_error"), "b.md": ("failed", "embed_error"),
        }
        failed = [e for e in logs if e["event"] == "file_ingestion_failed"]
        assert [(e["file"], e["error_type"]) for e in failed] == [
            ("a.md", "RuntimeError"), ("b.md", "RuntimeError"),
        ]
        assert marker not in repr(logs)


class TestLogsWithoutExceptionText:
    """No database needed: the parsers and embed_texts are called directly."""

    def test_corrupt_docx_logs_no_path(self, tmp_path):
        from structlog.testing import capture_logs

        from stratakb.knowledge.parsers.docx_parser import parse_docx

        file_path = tmp_path / "broken.docx"
        file_path.write_bytes(b"not a zip")
        with capture_logs() as logs:
            doc = parse_docx(file_path)

        assert doc.error
        events = [e for e in logs if e["event"] == "docx_parse_error"]
        assert len(events) == 1
        assert events[0]["error_type"] == "PackageNotFoundError"
        assert "error" not in events[0]
        assert str(tmp_path) not in repr(logs)

    def test_corrupt_pdf_logs_no_path(self, tmp_path):
        from structlog.testing import capture_logs

        from stratakb.knowledge.parsers.pdf_parser import parse_pdf

        file_path = tmp_path / "broken.pdf"
        file_path.write_bytes(b"%PDF-1.4 garbage")
        with (
            capture_logs() as logs,
            patch("pypdf.PdfReader", side_effect=ValueError(f"cannot open {file_path}")),
        ):
            doc = parse_pdf(file_path)

        assert str(file_path) in doc.error
        events = [e for e in logs if e["event"] == "pdf_parse_error"]
        assert len(events) == 1
        assert events[0]["error_type"] == "ValueError"
        assert "error" not in events[0]
        assert str(tmp_path) not in repr(logs)

    async def test_embedding_api_error_logs_type_only(self):
        from structlog.testing import capture_logs

        from stratakb.knowledge.embeddings import embed_texts

        marker = "distinctive-91c2-provider-text"
        client = fake_openai_client()
        client.embeddings.create = AsyncMock(side_effect=RuntimeError(marker))
        with (
            capture_logs() as logs,
            patch("stratakb.config.get_settings", return_value=openai_settings()),
            patch("openai.AsyncOpenAI", return_value=client),
            pytest.raises(RuntimeError),
        ):
            await embed_texts(["x"], api_key="sk-test")

        events = [e for e in logs if e["event"] == "openai_embedding_api_error"]
        assert len(events) == 1
        assert events[0]["error_type"] == "RuntimeError"
        assert "error" not in events[0]
        assert marker not in repr(logs)


class TestFileErrorBoundaries:
    ROOT = Path("/data/docs")
    FILE = Path("/data/docs/a.md")

    def _redact(self, message: str) -> str:
        from stratakb.knowledge.ingestion import _file_error

        return _file_error(message, file_path=self.FILE, relative_path="a.md", root=self.ROOT)

    @pytest.mark.parametrize(
        "message", ["see /data/docs2/x", "see /data/docs.bak/x", "see /mnt/data/docs/x"]
    )
    def test_longer_names_stay(self, message):
        with patch("stratakb.knowledge.ingestion._case_insensitive_fs", return_value=False):
            assert self._redact(message) == message

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("open /data/docs/b.md failed", "open ./b.md failed"),
            ("open /data/docs/a.md failed", "open a.md failed"),
            ("bad path '/data/docs'", "bad path '.'"),
            ("(under /data/docs)", "(under .)"),
            ("nothing at /data/docs.", "nothing at .."),
        ],
    )
    def test_path_at_a_boundary_is_replaced(self, message, expected):
        with patch("stratakb.knowledge.ingestion._case_insensitive_fs", return_value=False):
            assert self._redact(message) == expected

    def test_other_case_is_replaced_on_a_case_insensitive_fs(self):
        with patch("stratakb.knowledge.ingestion._case_insensitive_fs", return_value=True):
            assert self._redact("open /DATA/Docs/a.md failed") == "open a.md failed"

    def test_other_case_stays_on_a_case_sensitive_fs(self):
        with patch("stratakb.knowledge.ingestion._case_insensitive_fs", return_value=False):
            assert self._redact("open /DATA/Docs/a.md failed") == "open /DATA/Docs/a.md failed"

    @pytest.mark.skipif(
        sys.platform not in ("win32", "linux"), reason="case sensitivity varies by volume"
    )
    def test_case_insensitive_fs_on_this_platform(self, tmp_path):
        from stratakb.knowledge.ingestion import _case_insensitive_fs

        assert _case_insensitive_fs(tmp_path) is (sys.platform == "win32")
