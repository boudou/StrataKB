"""End-to-end search and ask with the real embedding and reranking models.

Nothing is patched: the example PEPs are embedded with the configured
local model, and strata_search and strata_ask run through the MCP tools. The
models are large, so these tests run only with STRATA_E2E=1 (the CI e2e job
sets it), the PEPs fetched (see examples/README.md) and the database up.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from uuid import uuid4

import pytest

from tests.test_example_peps import CLONE_HINT, EXAMPLE_PEPS, PEPS_PRESET, PEPS_SOURCE

pytestmark = pytest.mark.skipif(
    os.environ.get("STRATA_E2E") != "1",
    reason=(
        "e2e tests load the real embedding model; run with STRATA_E2E=1, "
        "the PEPs fetched and the database up"
    ),
)

QUESTION = "What is the maximum line length?"


def _payload(result) -> dict:
    """The tool's dict from a FastMCP call_tool result."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return result
    return json.loads(result[0].text)


@pytest.fixture
async def peps_layer(tmp_path):
    """A layer holding the example PEPs, embedded for real; deleted afterwards."""
    if not all((PEPS_SOURCE / name).is_file() for name in EXAMPLE_PEPS):
        pytest.skip(CLONE_HINT)

    from stratakb.config import get_settings
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer
    from stratakb.knowledge.presets import create_preset_layer, ingest_preset, load_preset

    db_url = get_settings().database_url
    docs = tmp_path / "source" / "peps"
    docs.mkdir(parents=True)
    for name in EXAMPLE_PEPS:
        shutil.copy2(PEPS_SOURCE / name, docs / name)

    preset = load_preset(PEPS_PRESET)
    preset.name = f"test_e2e_{uuid4().hex[:10]}"
    try:
        created = await create_preset_layer(db_url, preset)
        assert created["status"] == "created"
        result = await ingest_preset(db_url, "", preset, tmp_path)
        assert result["total_docs"] == len(EXAMPLE_PEPS)
        assert result["total_errors"] == 0
        yield preset.name
    finally:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=preset.name)


async def test_search_finds_pep8_for_line_length(server, peps_layer):
    response = _payload(
        await server.call_tool("strata_search", {"query": QUESTION, "layers": [peps_layer]})
    )

    assert response["status"] == "ok", response
    files = [r["source_file"] for r in response["results"][:3]]
    assert "pep-0008.rst" in files, files


async def test_ask_cites_pep8_with_a_confidence(server, peps_layer):
    response = _payload(
        await server.call_tool(
            "strata_ask",
            {"question": QUESTION, "layers": [peps_layer], "response_format": "detailed"},
        )
    )

    assert response["status"] == "ok", response
    assert response["total_chunks"] > 0
    assert response["confidence"]["level"] in {"high", "medium", "low"}
    assert "pep-0008.rst" in [s["file"] for s in response["sources"][:3]], response["sources"]
