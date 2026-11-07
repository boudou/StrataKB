"""Tests for the archived label a citation carries, and the preset key that sets the patterns.

A file matching its layer's archived patterns is labelled in the citation, worked
out on every ask. The tool tests ingest through the real pipeline against the
database with embeddings mocked, and call strata_ask in-process with the search
patched.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pydantic_core
import pytest

from stratakb.knowledge.layers import (
    LayerNotFoundError,
    LayerValidationError,
    create_layer,
    delete_layer,
    get_layer,
    update_layer,
)
from stratakb.knowledge.presets import check_preset_layer, create_preset_layer, load_preset
from stratakb.rag.passages import build_passages, cite, estimate_tokens, passage_cost
from tests.helpers import capture_tools, ingest, openai_settings, write_file
from tests.test_replaced_by import _ask_about_layer, _citations, _payload

FIXTURES = Path(__file__).parent / "fixtures"
BODY = "# Failover\n\nSwitch by hand.\n"
BASE_URL = "http://127.0.0.1:11434/v1"


@pytest.fixture
def settings():
    from stratakb.config import Settings, get_settings

    return Settings(
        _env_file=None,
        database_url=get_settings().database_url,
        learning_enabled=False,
        reranking_enabled=False,
        strata_user="test-user",
        strata_role="viewer",
    )


@pytest.fixture
def server(settings):
    """A StrataKB server on the test database, its settings patched in everywhere."""
    from stratakb.mcp_server import build_server

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
        patch(
            "stratakb.rag.query_rewriter.rewrite_query",
            AsyncMock(side_effect=lambda question, **_: [question]),
        ),
    ):
        yield build_server(settings)


@pytest.fixture
async def make_layer(db_url):
    """Make fresh layers, deleted afterwards."""
    names = []

    async def make(**kwargs) -> str:
        name = f"test_archived_{uuid4().hex[:10]}"
        names.append(name)
        await create_layer(db_url, name=name, display_name="Test Archived", **kwargs)
        return name

    yield make
    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


@pytest.fixture
async def preset_name(db_url):
    """A fresh name for a layer a preset makes, deleted afterwards."""
    name = f"test_archived_preset_{uuid4().hex[:10]}"
    yield name
    with contextlib.suppress(LayerNotFoundError):
        await delete_layer(db_url, name=name)


def _write_preset(tmp_path: Path, name: str = "archived_preset", line: str | None = None) -> Path:
    """A preset file, with a line for the archived_patterns key when given."""
    key = f"archived_patterns: {line}\n" if line is not None else ""
    path = tmp_path / "preset.yaml"
    path.write_text(
        f"name: {name}\ndisplay_name: Archived Preset\n{key}"
        "sources:\n  - {name: docs, path: docs}\n",
        encoding="utf-8",
    )
    return path


# ── The label ────────────────────────────────────────────


async def test_matching_file_is_labelled_without_reingest(server, db_url, make_layer, tmp_path):
    layer = await make_layer()
    write_file(tmp_path / "archive" / "old.md", BODY)
    write_file(tmp_path / "current.md", BODY)
    await ingest(db_url, layer, tmp_path)

    before = await _ask_about_layer(server, db_url, layer)
    citations = {p["citation"]["file"]: p["citation"] for p in before["passages"]}
    assert "archived" not in citations["archive/old.md"]

    await update_layer(db_url, layer, archived_patterns=["archive/"])
    labelled = await _ask_about_layer(server, db_url, layer)
    citations = {p["citation"]["file"]: p["citation"] for p in labelled["passages"]}
    assert citations["archive/old.md"]["archived"] is True
    assert "archived" not in citations["current.md"]

    await update_layer(db_url, layer, archived_patterns=[])
    cleared = await _ask_about_layer(server, db_url, layer)
    assert all("archived" not in p["citation"] for p in cleared["passages"])

    # The same stored spans all along: nothing was ingested in between
    refs = [sorted(p["ref"] for p in r["passages"]) for r in (before, labelled, cleared)]
    assert refs[0] == refs[1] == refs[2]


async def test_last_match_wins(server, db_url, make_layer, tmp_path):
    layer = await make_layer(archived_patterns=["archive/", "!archive/keep.md"])
    write_file(tmp_path / "archive" / "keep.md", BODY)
    write_file(tmp_path / "archive" / "old.md", BODY)
    await ingest(db_url, layer, tmp_path)

    citations = await _citations(server, db_url, layer)

    assert "archived" not in citations["archive/keep.md"]
    assert citations["archive/old.md"]["archived"] is True


async def test_other_layers_patterns_do_not_label(server, db_url, make_layer, tmp_path):
    marked = await make_layer(archived_patterns=["*.md"])
    plain = await make_layer()
    write_file(tmp_path / "one" / "a.md", BODY)
    write_file(tmp_path / "two" / "a.md", BODY)
    await ingest(db_url, marked, tmp_path / "one")
    await ingest(db_url, plain, tmp_path / "two")

    assert (await _citations(server, db_url, marked))["a.md"]["archived"] is True
    assert "archived" not in (await _citations(server, db_url, plain))["a.md"]


async def test_label_and_replaced_by_both_shown(server, db_url, make_layer, tmp_path):
    layer = await make_layer(archived_patterns=["old.md"])
    write_file(tmp_path / "old.md", BODY)
    write_file(tmp_path / "new.md", f"---\nsupersedes: old.md\n---\n{BODY}")
    await ingest(db_url, layer, tmp_path)

    citation = (await _citations(server, db_url, layer))["old.md"]

    assert citation["archived"] is True
    assert citation["replaced_by"] == "new.md"


async def test_exact_clause_passage_is_labelled(server, db_url, make_layer, tmp_path):
    layer = await make_layer(
        chunk_config={"chunk_size": 200, "chunk_overlap": 40}, archived_patterns=["contract.md"]
    )
    shutil.copy(FIXTURES / "contract.md", tmp_path / "contract.md")
    await ingest(db_url, layer, tmp_path)

    with patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[])):
        response = _payload(
            await server.call_tool(
                "strata_ask", {"question": "what does clause 2.2 say", "layers": [layer]}
            )
        )

    citation = response["passages"][0]["citation"]
    assert citation["clause_path"] == "2 Termination > 2.2 Termination for cause"
    assert citation["archived"] is True


async def test_extractive_answer_is_labelled(server, db_url, make_layer, tmp_path):
    layer = await make_layer(answer_mode="extractive", archived_patterns=["old.md"])
    write_file(tmp_path / "old.md", "# Failover\n\nSwitch by hand to the other site.\n")
    write_file(tmp_path / "new.md", "# Failover\n\nSwitch with the script to the other site.\n")
    await ingest(db_url, layer, tmp_path)

    def score_pairs(_question, texts, _model):
        return [0.9 for _ in texts]

    with patch("stratakb.rag.reranker.score_pairs", side_effect=score_pairs):
        response = await _ask_about_layer(server, db_url, layer)

    assert response["mode"] == "extractive"
    citations = {item["citation"]["file"]: item["citation"] for item in response["answer"]}
    assert citations["old.md"]["archived"] is True
    assert "archived" not in citations["new.md"]


async def test_patterns_are_not_in_the_response(server, db_url, make_layer, tmp_path):
    layer = await make_layer(archived_patterns=["*-2024.md"])
    write_file(tmp_path / "plan-2024.md", BODY)
    await ingest(db_url, layer, tmp_path)

    response = await _ask_about_layer(server, db_url, layer)

    assert response["passages"][0]["citation"]["archived"] is True
    text = json.dumps(response)
    assert "*-2024.md" not in text
    assert "archived_patterns" not in text


# The last four fail without the validator: "[z-a].md" and 1 do not compile, and
# the others would label a.md through the usable part of the list
@pytest.mark.parametrize(
    "stored",
    [
        ["*a*a*a*b"],
        ["["],
        {"a": 1},
        ["[z-a].md"],
        [1],
        ["a.md", "*a*a*a*b"],
        {"a.md": 1},
    ],
)
async def test_unusable_stored_patterns_leave_the_citation_whole(
    server, db_url, make_layer, tmp_path, stored
):
    from stratakb.core.database import get_pool

    layer = await make_layer()
    write_file(tmp_path / "a.md", BODY)
    await ingest(db_url, layer, tmp_path)
    pool = await get_pool(db_url)
    await pool.execute(
        "UPDATE strata_core.knowledge_layers SET archived_patterns = $1::jsonb WHERE name = $2",
        json.dumps(stored),
        layer,
    )

    citation = (await _citations(server, db_url, layer))["a.md"]

    assert citation["lines"] is not None
    assert "archived" not in citation


def test_unusable_patterns_log_the_layer_and_no_pattern():
    from uuid import UUID

    from structlog.testing import capture_logs

    from stratakb.knowledge.documents import _archived_spec

    layer_id = UUID(int=1)
    with capture_logs() as logs:
        assert _archived_spec('["*a*a*a*b"]', layer_id) is None

    assert logs == [
        {"event": "archived_patterns_unusable", "log_level": "warning", "layer_id": str(layer_id)}
    ]


@pytest.mark.parametrize("length", range(4))
def test_filled_citation_with_label_fits_passage_cost(length):
    def tokens(response: dict) -> int:
        return estimate_tokens(pydantic_core.to_json(response, indent=2).decode())

    # Six-digit spans, across two three-digit pages
    boundary = 150_000
    document = {"relative_path": "a.pdf", "page_map": [[0, 100], [boundary, 101]],
                "text": "x\n" * 100_000, "doc_date": "2026-03-04",
                "doc_date_source": "front_matter", "archived": True}
    chunk = {"chunk_id": "c", "content": "x" * length, "layer": "l", "doc_type": "pdf",
             "metadata": {}, "source_file": "a.pdf", "document_id": 123_456,
             "char_start": boundary - 10, "char_end": boundary + 10}
    passage = build_passages([chunk], {123_456: document})[0]
    assert passage["citation"]["page_end"] == 101
    assert passage["citation"]["archived"] is True

    response = {"status": "found", "mode": "grounded"}
    filled = tokens({**response, "passages": [passage]}) - tokens({**response, "passages": []})

    assert filled <= passage_cost(chunk)


def test_chunk_without_offsets_has_no_label():
    chunk = {"chunk_id": "c", "content": "x", "layer": "l", "doc_type": "markdown",
             "metadata": {}, "source_file": "a.md", "document_id": 1,
             "char_start": None, "char_end": None}
    document = {"relative_path": "a.md", "page_map": None, "text": "x", "archived": True}

    assert "archived" not in cite(chunk, document)


async def test_contract_names_the_label(server):
    rule = "A citation with archived true is a file its layer's patterns mark as archived."
    assert rule in server.instructions
    assert len(server.instructions) <= 4000
    prompt = await server.get_prompt("answer_from_documents", {"question": "q"})
    assert rule in prompt.messages[0].content.text


# ── The preset key ───────────────────────────────────────


def test_preset_key_absent_is_none(tmp_path):
    assert load_preset(_write_preset(tmp_path)).archived_patterns is None


def test_preset_key_empty_is_none(tmp_path):
    assert load_preset(_write_preset(tmp_path, line="")).archived_patterns is None


def test_preset_empty_list_is_kept(tmp_path):
    assert load_preset(_write_preset(tmp_path, line="[]")).archived_patterns == []


def test_preset_quoted_patterns_load(tmp_path):
    path = _write_preset(tmp_path, line="['archive/', '*-2024.md', '!keep.md']")

    assert load_preset(path).archived_patterns == ["archive/", "*-2024.md", "!keep.md"]


@pytest.mark.parametrize(
    "line", ["archive/", "false", "0", "{a: 1}", "[1]", "['[']", "['*a*a*a*b']", "['']"]
)
def test_preset_refused_values(tmp_path, line):
    with pytest.raises(LayerValidationError):
        load_preset(_write_preset(tmp_path, line=line))


@pytest.mark.parametrize("line", ["[*-2024.md]", "[!keep.md]"])
def test_unquoted_pattern_is_a_validation_error(tmp_path, line):
    with pytest.raises(LayerValidationError, match="quote a pattern"):
        load_preset(_write_preset(tmp_path, line=line))


async def test_preset_creates_a_layer_holding_them(db_url, tmp_path, preset_name):
    path = _write_preset(tmp_path, preset_name, "['archive/', '!keep.md']")

    result = await create_preset_layer(db_url, load_preset(path))

    assert result["status"] == "created"
    stored = await get_layer(db_url, name=preset_name)
    assert stored["archived_patterns"] == ["archive/", "!keep.md"]


async def test_absent_key_leaves_stored_patterns(db_url, tmp_path, preset_name):
    await create_preset_layer(
        db_url, load_preset(_write_preset(tmp_path, preset_name, "['archive/']"))
    )

    result = await create_preset_layer(db_url, load_preset(_write_preset(tmp_path, preset_name)))

    assert result["status"] == "exists"
    assert result["changes"] == []
    assert (await get_layer(db_url, name=preset_name))["archived_patterns"] == ["archive/"]


async def test_changed_patterns_reported(db_url, tmp_path, preset_name):
    await create_preset_layer(
        db_url, load_preset(_write_preset(tmp_path, preset_name, "['archive/']"))
    )

    changed = await create_preset_layer(
        db_url, load_preset(_write_preset(tmp_path, preset_name, "['old/']"))
    )
    assert changed["status"] == "updated"
    assert changed["changes"] == ["archived_patterns"]
    assert (await get_layer(db_url, name=preset_name))["archived_patterns"] == ["old/"]

    cleared = await create_preset_layer(
        db_url, load_preset(_write_preset(tmp_path, preset_name, "[]"))
    )
    assert cleared["changes"] == ["archived_patterns"]
    assert (await get_layer(db_url, name=preset_name))["archived_patterns"] == []


async def test_dry_run_reports_the_change_by_name(db_url, tmp_path, preset_name):
    await create_preset_layer(
        db_url, load_preset(_write_preset(tmp_path, preset_name, "['archive/']"))
    )

    check = await check_preset_layer(
        db_url, load_preset(_write_preset(tmp_path, preset_name, "['*-2024.md']"))
    )

    assert check == {"status": "would_update", "changes": ["archived_patterns"]}
    assert (await get_layer(db_url, name=preset_name))["archived_patterns"] == ["archive/"]


# ── The audit entry from strata_apply_preset ─────────────


async def _apply(tmp_path: Path, line: str | None, layer_result: dict, ingest_mock=None):
    """Apply a preset with its layer step's result given; return the result and the entries."""
    from stratakb.knowledge.tools import register_knowledge_tools

    fn = capture_tools(register_knowledge_tools, "postgresql://test", "")["strata_apply_preset"]
    audit = AsyncMock()
    ingest_preset = ingest_mock or AsyncMock(return_value={"total_docs": 0, "total_chunks": 0})
    result = None
    with (
        patch(
            "stratakb.config.get_settings",
            return_value=openai_settings(embedding_base_url=BASE_URL),
        ),
        patch(
            "stratakb.knowledge.presets.create_preset_layer",
            AsyncMock(return_value=layer_result),
        ),
        patch("stratakb.knowledge.presets.ingest_preset", ingest_preset),
        patch("stratakb.knowledge.tools.write_audit_log", audit),
        contextlib.suppress(RuntimeError),
    ):
        result = await fn(preset_path=str(_write_preset(tmp_path, line=line)))
    entries = [
        call.kwargs for call in audit.await_args_list
        if call.kwargs["action"] == "set_archived_patterns"
    ]
    return result, entries


@pytest.mark.parametrize(
    ("line", "layer_result", "patterns"),
    [
        ("['old/']", {"status": "updated", "changes": ["archived_patterns"]}, ["old/"]),
        ("[]", {"status": "updated", "changes": ["archived_patterns"]}, []),
        ("['old/']", {"status": "created", "changes": []}, ["old/"]),
    ],
    ids=["changed", "cleared", "created_with_patterns"],
)
async def test_audit_entry_holds_only_layer_and_patterns(tmp_path, line, layer_result, patterns):
    _, entries = await _apply(tmp_path, line, layer_result)

    assert len(entries) == 1
    assert entries[0]["tool_name"] == "strata_apply_preset"
    assert entries[0]["parameters"] == {"layer": "archived_preset", "patterns": patterns}


@pytest.mark.parametrize(
    ("line", "layer_result"),
    [
        (None, {"status": "created", "changes": []}),
        ("[]", {"status": "created", "changes": []}),
        ("['old/']", {"status": "exists", "changes": []}),
        (None, {"status": "updated", "changes": ["description"]}),
    ],
    ids=["created_without", "created_empty", "unchanged", "other_setting_changed"],
)
async def test_no_audit_entry_without_a_change(tmp_path, line, layer_result):
    _, entries = await _apply(tmp_path, line, layer_result)

    assert entries == []


async def test_audit_entry_is_written_when_the_ingest_raises(tmp_path):
    failing = AsyncMock(side_effect=RuntimeError("ingest failed"))

    _, entries = await _apply(
        tmp_path, "['old/']", {"status": "updated", "changes": ["archived_patterns"]}, failing
    )

    failing.assert_awaited_once()
    assert len(entries) == 1


async def test_tool_reports_an_unquoted_pattern_as_an_error(tmp_path):
    result, entries = await _apply(
        tmp_path, "[*-2024.md]", {"status": "created", "changes": []}
    )

    assert result["status"] == "error"
    assert "quote a pattern" in result["error"]
    assert entries == []
