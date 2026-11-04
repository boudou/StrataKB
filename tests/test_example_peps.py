"""Tests for the Python PEPs example against real PEP files.

The PEPs are not committed: these tests run only when the clone from
examples/README.md exists under examples/peps/source (CI fetches it before the
tests) and skip with the clone commands otherwise. The dry run needs no
database; the apply test needs the database and patches the embeddings. All
chunks get the same dummy embedding, so the search check reads only the
full-text leg of the hybrid SQL.
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from stratakb.knowledge.presets import (
    create_preset_layer,
    dry_run_parse,
    ingest_preset,
    load_preset,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
PEPS_PRESET = EXAMPLES_DIR / "peps" / "preset.yaml"
PEPS_SOURCE = EXAMPLES_DIR / "peps" / "source" / "peps"
EIGHT_PEPS = [
    "pep-0008.rst",
    "pep-0020.rst",
    "pep-0257.rst",
    "pep-0484.rst",
    "pep-0572.rst",
    "pep-0634.rst",
    "pep-0695.rst",
    "pep-0723.rst",
]
CLONE_HINT = (
    "PEPs not fetched; run: "
    "git clone --depth 1 --filter=blob:none --sparse https://github.com/python/peps.git "
    "examples/peps/source && "
    "git -C examples/peps/source sparse-checkout set --no-cone "
    + " ".join(f"/peps/{name}" for name in EIGHT_PEPS)
)


@pytest.fixture
def db_url():
    from stratakb.config import get_settings
    return get_settings().database_url


def test_checkout_dry_run_parses_everything():
    if not PEPS_SOURCE.is_dir():
        pytest.skip(CLONE_HINT)

    result = dry_run_parse(load_preset(PEPS_PRESET))

    assert result.total_files > 0
    assert result.parsed_files == result.total_files
    assert result.failed_files == 0
    assert result.errors == []


async def test_eight_peps_apply_and_find_pep8(db_url, tmp_path):
    if not all((PEPS_SOURCE / name).is_file() for name in EIGHT_PEPS):
        pytest.skip(CLONE_HINT)

    from stratakb.config import get_settings
    from stratakb.core.database import get_pool
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer
    from stratakb.rag.retriever import _build_hybrid_sql

    # Exactly these eight files, even when the clone holds every PEP
    docs = tmp_path / "source" / "peps"
    docs.mkdir(parents=True)
    for name in EIGHT_PEPS:
        shutil.copy2(PEPS_SOURCE / name, docs / name)

    preset = load_preset(PEPS_PRESET)
    preset.name = f"test_peps_{uuid4().hex[:10]}"
    dims = get_settings().embedding_dimensions
    vector = [0.01] * dims

    try:
        dry = dry_run_parse(preset, tmp_path)
        assert dry.total_files == 8
        assert dry.parsed_files == 8
        assert dry.failed_files == 0

        created = await create_preset_layer(db_url, preset)
        assert created["status"] == "created"

        with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
            mock.side_effect = lambda texts, **_: [vector for _ in texts]
            result = await ingest_preset(db_url, "fake-key", preset, tmp_path)

        assert result["total_docs"] == 8
        assert result["total_errors"] == 0
        assert result["groups"]["peps"]["status"] == "completed"

        pool = await get_pool(db_url)
        rows = await pool.fetch(
            "SELECT DISTINCT source_file FROM strata_vectors.chunks WHERE layer_name = $1",
            preset.name,
        )
        assert sorted(r["source_file"] for r in rows) == EIGHT_PEPS

        sql, params = _build_hybrid_sql(
            "[" + ",".join(["0.01"] * dims) + "]", "maximum line length", [preset.name], top_k=3
        )
        async with pool.acquire() as conn:
            hits = await conn.fetch(sql, *params)
            ranked = sorted((r for r in hits if r["source"] == "fulltext"), key=lambda r: r["rank"])
            files = [
                await conn.fetchval(
                    "SELECT source_file FROM strata_vectors.chunks WHERE id = $1", r["id"]
                )
                for r in ranked[:3]
            ]
        assert "pep-0008.rst" in files
    finally:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=preset.name)
