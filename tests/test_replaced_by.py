"""Tests for the replaced_by label a citation carries.

A document that declares supersedes labels the file it names, in the citation.
The tool tests ingest through the real pipeline against the database with
embeddings mocked, and call strata_ask in-process with the search patched.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pydantic_core
import pytest

from stratakb.knowledge.policy import MAX_RELATIVE_PATH
from stratakb.rag.extractive import Sentence, fit_to_answer_budget
from stratakb.rag.passages import build_passages, cite, estimate_tokens, passage_cost
from tests.helpers import ingest, write_file

FIXTURES = Path(__file__).parent / "fixtures"
OLD_TEXT = "# Failover\n\nSwitch by hand.\n"
NEW_BODY = "# Failover\n\nSwitch with the script.\n"
NO_LABEL_KEYS = {"file", "page", "lines", "clause_path", "layer", "date", "date_source"}


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
    from stratakb.knowledge.layers import LayerNotFoundError, create_layer, delete_layer

    names = []

    async def make(**kwargs) -> str:
        name = f"test_replaced_{uuid4().hex[:10]}"
        names.append(name)
        await create_layer(db_url, name=name, display_name="Test Replaced", **kwargs)
        return name

    yield make
    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


@pytest.fixture
async def layer(make_layer):
    return await make_layer()


def _declaring(target: str, body: str = NEW_BODY) -> str:
    return f"---\nsupersedes: {target}\n---\n{body}"


async def _ask_about_layer(server, db_url: str, layer: str) -> dict:
    """strata_ask with the search patched to return every chunk of the layer."""
    from stratakb.core.database import get_pool
    from stratakb.rag.retriever import SearchResult

    pool = await get_pool(db_url)
    rows = await pool.fetch(
        """
        SELECT id, document_id, char_start, char_end, content, source_file
        FROM strata_vectors.chunks WHERE layer_name = $1 ORDER BY id
        """,
        layer,
    )
    results = [
        SearchResult(
            chunk_id=str(row["id"]),
            content=row["content"],
            score=1.0,
            layer_name=layer,
            source_file=row["source_file"],
            source_section="section",
            doc_type="markdown",
            metadata={},
            similarity=0.9,
            document_id=row["document_id"],
            char_start=row["char_start"],
            char_end=row["char_end"],
        )
        for row in rows
    ]
    with patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=results)):
        return _payload(await server.call_tool("strata_ask", {"question": "q", "layers": [layer]}))


def _payload(result) -> dict:
    if isinstance(result, tuple):
        result = result[0]
    return result if isinstance(result, dict) else json.loads(result[0].text)


async def _citations(server, db_url: str, layer: str) -> dict[str, dict]:
    """Each cited file's citation, by file."""
    response = await _ask_about_layer(server, db_url, layer)
    assert response["status"] != "error"
    return {p["citation"]["file"]: p["citation"] for p in response["passages"]}


async def _labels(server, db_url: str, layer: str) -> dict[str, str | None]:
    """Each cited file's replaced_by, None without the key."""
    citations = await _citations(server, db_url, layer)
    return {file: citation.get("replaced_by") for file, citation in citations.items()}


async def _execute(db_url: str, sql: str, *args) -> str:
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.execute(sql, *args)


async def test_replaced_file_names_the_file_that_says_so(server, db_url, layer, tmp_path):
    write_file(tmp_path / "failover-2024.md", OLD_TEXT)
    write_file(tmp_path / "failover-2026.md", _declaring("failover-2024.md"))
    await ingest(db_url, layer, tmp_path)

    citations = await _citations(server, db_url, layer)

    assert citations["failover-2024.md"]["replaced_by"] == "failover-2026.md"
    assert "replaced_by" not in citations["failover-2026.md"]


async def test_nested_path_resolves_from_the_declaring_folder(server, db_url, layer, tmp_path):
    write_file(tmp_path / "old.md", OLD_TEXT)
    write_file(tmp_path / "docs" / "new.md", _declaring("../old.md"))
    await ingest(db_url, layer, tmp_path)

    labels = await _labels(server, db_url, layer)

    assert labels == {"old.md": "docs/new.md", "docs/new.md": None}


async def test_missing_target_gives_no_label(server, db_url, layer, tmp_path):
    write_file(tmp_path / "old.md", OLD_TEXT)
    write_file(tmp_path / "new.md", _declaring("gone.md"))
    await ingest(db_url, layer, tmp_path)

    citations = await _citations(server, db_url, layer)

    assert set(citations) == {"old.md", "new.md"}
    for citation in citations.values():
        assert set(citation) == NO_LABEL_KEYS


async def test_nothing_declared_adds_no_key(server, db_url, layer, tmp_path):
    write_file(tmp_path / "plain.md", OLD_TEXT)
    await ingest(db_url, layer, tmp_path)

    citations = await _citations(server, db_url, layer)

    assert set(citations["plain.md"]) == NO_LABEL_KEYS


async def test_key_removed_in_a_new_version_gives_no_label(server, db_url, layer, tmp_path):
    write_file(tmp_path / "old.md", OLD_TEXT)
    write_file(tmp_path / "new.md", _declaring("old.md"))
    await ingest(db_url, layer, tmp_path)
    assert (await _labels(server, db_url, layer))["old.md"] == "new.md"

    # Version 1 stays, with its value and its file_id, as a receipt can keep it
    await _execute(
        db_url,
        """
        INSERT INTO strata_vectors.documents
            (layer_id, file_id, source_root, relative_path, version, text, sha256, supersedes)
        SELECT layer_id, file_id, source_root, relative_path, version + 1, text, sha256, NULL
        FROM strata_vectors.documents
        WHERE supersedes = 'old.md'
          AND layer_id = (SELECT id FROM strata_core.knowledge_layers WHERE name = $1)
        """,
        layer,
    )

    assert (await _labels(server, db_url, layer))["old.md"] is None


async def test_removed_file_gives_no_label(server, db_url, layer, tmp_path):
    write_file(tmp_path / "old.md", OLD_TEXT)
    declaring = write_file(tmp_path / "new.md", _declaring("old.md"))
    await ingest(db_url, layer, tmp_path)
    assert (await _labels(server, db_url, layer))["old.md"] == "new.md"

    declaring.unlink()
    await ingest(db_url, layer, tmp_path)

    assert await _labels(server, db_url, layer) == {"old.md": None}


async def test_row_without_a_file_gives_no_label(server, db_url, layer, tmp_path):
    write_file(tmp_path / "old.md", OLD_TEXT)
    write_file(tmp_path / "new.md", _declaring("old.md"))
    await ingest(db_url, layer, tmp_path)

    await _execute(
        db_url,
        """
        UPDATE strata_vectors.documents SET file_id = NULL
        WHERE supersedes = 'old.md'
          AND layer_id = (SELECT id FROM strata_core.knowledge_layers WHERE name = $1)
        """,
        layer,
    )

    assert (await _labels(server, db_url, layer))["old.md"] is None


async def test_other_layer_gives_no_label(server, db_url, make_layer, tmp_path):
    first, second = await make_layer(), await make_layer()
    write_file(tmp_path / "old.md", OLD_TEXT)
    await ingest(db_url, first, tmp_path)
    write_file(tmp_path / "new.md", _declaring("old.md"))
    await ingest(db_url, second, tmp_path)

    assert (await _labels(server, db_url, first))["old.md"] is None
    assert (await _labels(server, db_url, second))["old.md"] == "new.md"


async def test_other_root_gives_no_label(server, db_url, layer, tmp_path):
    write_file(tmp_path / "one" / "old.md", OLD_TEXT)
    write_file(tmp_path / "two" / "new.md", _declaring("old.md"))
    await ingest(db_url, layer, tmp_path / "one")
    await ingest(db_url, layer, tmp_path / "two")

    labels = await _labels(server, db_url, layer)

    assert labels == {"old.md": None, "new.md": None}


async def test_first_replacing_file_by_path_is_named(server, db_url, layer, tmp_path):
    write_file(tmp_path / "old.md", OLD_TEXT)
    write_file(tmp_path / "b.md", _declaring("old.md", "# B\n\nThe second replacement.\n"))
    write_file(tmp_path / "a.md", _declaring("old.md", "# A\n\nThe first replacement.\n"))
    await ingest(db_url, layer, tmp_path)

    assert (await _labels(server, db_url, layer))["old.md"] == "a.md"


async def test_chain_names_only_the_next_file(server, db_url, layer, tmp_path):
    write_file(tmp_path / "a.md", OLD_TEXT)
    write_file(tmp_path / "b.md", _declaring("a.md", "# B\n\nThe second runbook.\n"))
    write_file(tmp_path / "c.md", _declaring("b.md", "# C\n\nThe third runbook.\n"))
    await ingest(db_url, layer, tmp_path)

    labels = await _labels(server, db_url, layer)

    assert labels == {"a.md": "b.md", "b.md": "c.md", "c.md": None}


async def test_key_stored_before_takes_effect_when_the_file_changes(
    server, db_url, layer, tmp_path
):
    write_file(tmp_path / "old.md", OLD_TEXT)
    declaring = write_file(tmp_path / "new.md", _declaring("old.md"))
    await ingest(db_url, layer, tmp_path)
    # As stored before the column existed
    await _execute(
        db_url,
        """
        UPDATE strata_vectors.documents SET supersedes = NULL
        WHERE layer_id = (SELECT id FROM strata_core.knowledge_layers WHERE name = $1)
        """,
        layer,
    )

    await ingest(db_url, layer, tmp_path)
    assert (await _labels(server, db_url, layer))["old.md"] is None

    declaring.write_text(_declaring("old.md", "# Failover\n\nSwitch with the tool.\n"), "utf-8")
    await ingest(db_url, layer, tmp_path)
    assert (await _labels(server, db_url, layer))["old.md"] == "new.md"


async def test_exact_clause_passage_is_labelled(server, db_url, make_layer, tmp_path):
    layer = await make_layer(chunk_config={"chunk_size": 200, "chunk_overlap": 40})
    shutil.copy(FIXTURES / "contract.md", tmp_path / "contract.md")
    write_file(tmp_path / "contract-2026.md", _declaring("contract.md"))
    await ingest(db_url, layer, tmp_path)

    with patch("stratakb.rag.retriever.search_chunks", AsyncMock(return_value=[])):
        response = _payload(
            await server.call_tool(
                "strata_ask", {"question": "what does clause 2.2 say", "layers": [layer]}
            )
        )

    citation = response["passages"][0]["citation"]
    assert citation["file"] == "contract.md"
    assert citation["clause_path"] == "2 Termination > 2.2 Termination for cause"
    assert citation["replaced_by"] == "contract-2026.md"


async def test_extractive_answer_is_labelled(server, db_url, make_layer, tmp_path):
    layer = await make_layer(answer_mode="extractive")
    write_file(tmp_path / "old.md", "# Failover\n\nSwitch by hand to the other site.\n")
    write_file(tmp_path / "new.md", _declaring("old.md"))
    await ingest(db_url, layer, tmp_path)

    def score_pairs(_question, texts, _model):
        return [0.9 if "by hand" in text else 0.05 for text in texts]

    with patch("stratakb.rag.reranker.score_pairs", side_effect=score_pairs):
        response = await _ask_about_layer(server, db_url, layer)

    assert response["mode"] == "extractive"
    citation = response["answer"][0]["citation"]
    assert citation["file"] == "old.md"
    assert citation["replaced_by"] == "new.md"


def test_chunk_without_offsets_has_no_label():
    chunk = {"chunk_id": "c", "content": "x", "layer": "l", "doc_type": "markdown",
             "metadata": {}, "source_file": "a.md", "document_id": 1,
             "char_start": None, "char_end": None}
    document = {"relative_path": "a.md", "page_map": None, "text": "x", "replaced_by": "b.md"}

    assert "replaced_by" not in cite(chunk, document)


def test_grounded_label_is_not_counted_and_overshoots_by_its_length():
    def tokens(passage: dict) -> int:
        return estimate_tokens(pydantic_core.to_json({"passages": [passage]}, indent=2).decode())

    label = "x" * MAX_RELATIVE_PATH
    chunk = {"chunk_id": "c", "content": "x", "layer": "l", "doc_type": "markdown",
             "metadata": {}, "source_file": "a.md", "document_id": 1,
             "char_start": 0, "char_end": 1}
    document = {"relative_path": "a.md", "page_map": None, "text": "x"}
    cost = passage_cost(chunk)

    plain = build_passages([chunk], {1: document})[0]
    labelled = build_passages([chunk], {1: {**document, "replaced_by": label}})[0]

    assert labelled["citation"]["replaced_by"] == label
    assert passage_cost(chunk) == cost
    assert 0 < tokens(labelled) - tokens(plain) <= estimate_tokens(label) + 8


def test_extractive_budget_counts_the_label():
    low = replace(Sentence(1, 0, 400, "x" * 400, 0, False), score=0.6)
    high = replace(Sentence(1, 500, 900, "y" * 400, 0, False), score=0.9)
    sentences = [low, high]
    chunks = [{"document_id": 1, "char_start": 0, "char_end": 1000}]
    document = {"relative_path": "a.md", "page_map": None, "text": "x" * 1000,
                "text_origin": "text"}
    labelled = {**document, "replaced_by": "x" * MAX_RELATIVE_PATH}
    # The smallest budget that keeps both sentences of the document with no label
    budget = next(
        n for n in range(1, 2000)
        if len(fit_to_answer_budget(sentences, chunks, {1: document}, n)) == 2
    )

    assert fit_to_answer_budget(sentences, chunks, {1: document}, budget) == sentences
    assert fit_to_answer_budget(sentences, chunks, {1: labelled}, budget) == [high]


async def test_contract_names_the_label(server):
    assert "replaced_by" in server.instructions
    assert len(server.instructions) <= 4000
    prompt = await server.get_prompt("answer_from_documents", {"question": "q"})
    assert "replaced_by" in prompt.messages[0].content.text
