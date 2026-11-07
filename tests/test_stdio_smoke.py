"""Drive the server over stdio the way an MCP client does.

Starts ``python -m stratakb`` as a subprocess, sends initialize, tools/list, a
dry-run strata_apply_preset, prompts/list and prompts/get, and checks that every
line on stdout is a JSON-RPC message: anything else there breaks the client's
session. Needs no database and no embedding model.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SEPA_PRESET = REPO / "examples" / "sepa" / "preset.yaml"
TIMEOUT = 30


class _Server:
    def __init__(self):
        env = {
            **os.environ,
            "STRATA_ROLE": "admin",
            # Nothing listens here: the protocol must not need the database
            "DATABASE_URL": "postgresql://strata:none@127.0.0.1:9/strata",
            # No local model to load: importing it can take a minute and
            # hold up the first tool call (#3)
            "EMBEDDING_BACKEND": "openai",
            "EMBEDDING_BASE_URL": "http://127.0.0.1:9/v1",
            "EMBEDDING_MODEL": "smoke-test",
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        env.pop("READONLY_DATABASE_URL", None)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "stratakb"],
            cwd=REPO,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        self.lines: list[str] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self._queue.put(line)
        self._queue.put(None)

    def send(self, message: dict):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        self.proc.stdin.flush()

    def reply(self, request_id: int) -> dict:
        """Read stdout until the reply to request_id arrives."""
        while True:
            try:
                line = self._queue.get(timeout=TIMEOUT)
            except queue.Empty:
                pytest.fail(f"no reply to request {request_id} within {TIMEOUT} s")
            if line is None:
                pytest.fail(f"server closed stdout before replying to request {request_id}")
            self.lines.append(line)
            message = json.loads(line) if line.strip().startswith("{") else {}
            if message.get("id") == request_id:
                return message

    def close(self) -> list[str]:
        """Close stdin, wait for the exit and return every stdout line."""
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            pytest.fail(f"server did not exit within {TIMEOUT} s after stdin closed")
        while (line := self._queue.get(timeout=TIMEOUT)) is not None:
            self.lines.append(line)
        return self.lines


def test_stdio_session_carries_only_json_rpc():
    server = _Server()
    try:
        server.send({
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "smoke-test", "version": "0"},
            },
        })
        init = server.reply(1)
        assert init["result"]["serverInfo"]["name"]
        assert "strata_verify" in init["result"]["instructions"]

        server.send({"method": "notifications/initialized"})
        server.send({"id": 2, "method": "tools/list"})
        names = {tool["name"] for tool in server.reply(2)["result"]["tools"]}
        assert {"strata_search", "strata_ask", "strata_apply_preset"} <= names

        server.send({
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "strata_apply_preset",
                "arguments": {"preset_path": str(SEPA_PRESET), "dry_run": True},
            },
        })
        call = server.reply(3)["result"]
        assert not call.get("isError"), call
        assert json.loads(call["content"][0]["text"])["status"] == "dry_run_complete"

        server.send({"id": 4, "method": "prompts/list"})
        prompts = {prompt["name"] for prompt in server.reply(4)["result"]["prompts"]}
        assert "answer_from_documents" in prompts

        server.send({
            "id": 5,
            "method": "prompts/get",
            "params": {
                "name": "answer_from_documents",
                "arguments": {"question": "How?", "layers": "a, b,"},
            },
        })
        text = server.reply(5)["result"]["messages"][0]["content"]["text"]
        assert "How?" in text
        assert '["a", "b"]' in text
    finally:
        lines = server.close()

    for line in lines:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            pytest.fail(f"stdout carries a line that is not JSON-RPC: {line!r}")
        assert message.get("jsonrpc") == "2.0", f"not a JSON-RPC message: {line!r}"
