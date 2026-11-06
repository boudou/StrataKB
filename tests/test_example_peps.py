"""Tests for the Python PEPs example against real PEP files.

The PEPs are not committed: these tests run only when the clone from
examples/README.md exists under examples/peps/source (CI fetches it before the
tests) and skip with the clone commands otherwise. The dry run needs no
database; the apply test needs the database and patches the embeddings. All
chunks get the same dummy embedding, so the search check reads only the
full-text leg of the hybrid SQL. The list of PEPs lives in
examples/peps/peps.txt; a check that needs no clone keeps the copies in the
READMEs and the CI workflow equal to it.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from stratakb.knowledge.presets import (
    create_preset_layer,
    dry_run_parse,
    ingest_preset,
    load_preset,
)
from tests.helpers import (
    CLONE_HINT,
    EXAMPLE_PEPS,
    PEPS_PRESET,
    PEPS_SOURCE,
    REPO_DIR,
    fulltext_hits,
)

# Every other place that spells out the list
PEPS_LIST_COPIES = ["README.md", "examples/README.md", ".github/workflows/ci.yml"]


@pytest.mark.parametrize("copy", PEPS_LIST_COPIES)
def test_pep_list_copies_match_peps_txt(copy):
    text = (REPO_DIR / copy).read_text(encoding="utf-8")
    found = re.findall(r"/peps/(pep-\d+\.rst)", text)
    assert sorted(found) == sorted(EXAMPLE_PEPS), (
        f"{copy} fetches other PEPs than examples/peps/peps.txt"
    )
    assert found == EXAMPLE_PEPS, (
        f"{copy} lists the PEPs in another order than examples/peps/peps.txt"
    )


def test_clone_hint_has_the_git_bash_note():
    assert "MSYS_NO_PATHCONV=1" in CLONE_HINT


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


async def test_example_peps_apply_and_find_pep8(db_url, tmp_path):
    if not all((PEPS_SOURCE / name).is_file() for name in EXAMPLE_PEPS):
        pytest.skip(CLONE_HINT)

    from stratakb.config import get_settings
    from stratakb.core.database import get_pool
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer

    # Exactly the listed files, even when the clone holds every PEP
    docs = tmp_path / "source" / "peps"
    docs.mkdir(parents=True)
    for name in EXAMPLE_PEPS:
        shutil.copy2(PEPS_SOURCE / name, docs / name)

    preset = load_preset(PEPS_PRESET)
    preset.name = f"test_peps_{uuid4().hex[:10]}"
    dims = get_settings().embedding_dimensions
    vector = [0.01] * dims

    try:
        dry = dry_run_parse(preset, tmp_path)
        assert dry.total_files == len(EXAMPLE_PEPS)
        assert dry.parsed_files == len(EXAMPLE_PEPS)
        assert dry.failed_files == 0

        created = await create_preset_layer(db_url, preset)
        assert created["status"] == "created"

        with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
            mock.side_effect = lambda texts, **_: [vector for _ in texts]
            result = await ingest_preset(db_url, "fake-key", preset, tmp_path)

        assert result["total_docs"] == len(EXAMPLE_PEPS)
        assert result["total_errors"] == 0
        assert result["groups"]["peps"]["status"] == "completed"

        pool = await get_pool(db_url)
        rows = await pool.fetch(
            "SELECT DISTINCT source_file FROM strata_vectors.chunks WHERE layer_name = $1",
            preset.name,
        )
        assert sorted(r["source_file"] for r in rows) == sorted(EXAMPLE_PEPS)

        hits = await fulltext_hits(
            pool, preset.name, "[" + ",".join(["0.01"] * dims) + "]", "maximum line length"
        )
        assert "pep-0008.rst" in [r["source_file"] for r in hits[:3]]
    finally:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=preset.name)
