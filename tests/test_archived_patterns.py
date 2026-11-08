"""Tests for a layer's archived files patterns and strata_set_archived_patterns.

The validator and matching tests need no database; the others do.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest
from pathspec import GitIgnoreSpec

from stratakb.core import migrations
from stratakb.core.audit import query_audit_log
from stratakb.core.auth import PermissionDeniedError
from stratakb.knowledge.layers import (
    LayerValidationError,
    _validate_archived_patterns,
    create_layer,
    get_layer_status,
)
from tests.helpers import call_tool, owner_execute, recorded_versions, unique_layer_name

pytestmark = pytest.mark.usefixtures("clear_checked")

PATTERNS = ["archive/", "*-2024.md", "!keep-2024.md"]
REFUSED = [
    "!",
    "\\",
    "foo\\",
    "#comment",
    "/",
    "[",
    "[z-a].md",
    "",
    "  ",
    "a\nb",
    "a\x00b",
    "*a*a*a*b",
    "****",
    "***/***/x",
    "/***/***/***/x",
]


@pytest.fixture
def as_admin():
    with patch("stratakb.core.auth.get_current_user", return_value=("u", "admin")):
        yield


@pytest.fixture
async def layer(db_url, cleanup):
    name = unique_layer_name()
    cleanup.append(name)
    await create_layer(db_url, name=name, display_name="Archived files test")
    return name


async def _audit_entries(db_url: str, layer_name: str) -> list[dict]:
    """The parameters of this layer's strata_set_archived_patterns entries, newest first."""
    entries = await query_audit_log(db_url, tool_name="strata_set_archived_patterns", limit=1000)
    parameters = [
        json.loads(e["parameters"]) if isinstance(e["parameters"], str) else e["parameters"]
        for e in entries
    ]
    return [p for p in parameters if p["layer"] == layer_name]


@pytest.mark.parametrize(
    "pattern",
    ["archive/", "*-2024.md", "!keep.md", "\\#x", "**", "archive/**/*-2024.md", "**/a/**/a/**/b"],
)
def test_validator_accepts(pattern):
    assert _validate_archived_patterns([pattern]) == [pattern]


@pytest.mark.parametrize("pattern", REFUSED)
def test_validator_refuses(pattern):
    with pytest.raises(LayerValidationError, match="archived files pattern"):
        _validate_archived_patterns([pattern])


@pytest.mark.parametrize("value", ["archive/", None, 3])
def test_validator_refuses_what_is_not_a_list(value):
    with pytest.raises(LayerValidationError, match="must be a list"):
        _validate_archived_patterns(value)


def test_validator_refuses_a_non_string_item():
    with pytest.raises(LayerValidationError, match="non-blank string"):
        _validate_archived_patterns([1])


def test_patterns_match_as_documented():
    spec = GitIgnoreSpec.from_lines(PATTERNS)
    for path in ("archive/x.md", "a/archive/x.md", "d/a-2024.md"):
        assert spec.match_file(path), path
    assert not spec.match_file("keep-2024.md")


async def test_tool_sets_and_clears(server, db_url, layer, as_admin):
    result = await call_tool(
        server, "strata_set_archived_patterns", layer_name=layer, patterns=PATTERNS
    )
    assert result["status"] == "updated"
    assert result["layer"]["archived_patterns"] == PATTERNS

    with patch("stratakb.knowledge.tools.get_current_user", return_value=("u", "admin")):
        status = await call_tool(server, "strata_get_layer_status", name=layer)
    assert status["layer"]["archived_patterns"] == PATTERNS

    result = await call_tool(server, "strata_set_archived_patterns", layer_name=layer, patterns=[])
    assert result["layer"]["archived_patterns"] == []
    assert (await get_layer_status(db_url, name=layer))["archived_patterns"] == []


@pytest.mark.parametrize("pattern", REFUSED)
async def test_tool_refuses_bad_pattern_and_keeps_old(server, db_url, layer, as_admin, pattern):
    await call_tool(server, "strata_set_archived_patterns", layer_name=layer, patterns=PATTERNS)
    audited = await _audit_entries(db_url, layer)

    result = await call_tool(
        server, "strata_set_archived_patterns", layer_name=layer, patterns=["ok/", pattern]
    )

    assert result["status"] == "error"
    assert "archived files pattern" in result["error"]
    assert (await get_layer_status(db_url, name=layer))["archived_patterns"] == PATTERNS
    assert await _audit_entries(db_url, layer) == audited


async def test_tool_unknown_layer(server, db_url, as_admin):
    name = unique_layer_name()
    result = await call_tool(server, "strata_set_archived_patterns", layer_name=name, patterns=[])
    assert result["status"] == "error"
    assert await _audit_entries(db_url, name) == []


@pytest.mark.parametrize("role", ["viewer", "operator"])
async def test_tool_needs_admin(server, db_url, layer, role):
    with (
        patch("stratakb.core.auth.get_current_user", return_value=("u", role)),
        pytest.raises(PermissionDeniedError),
    ):
        await call_tool(server, "strata_set_archived_patterns", layer_name=layer, patterns=PATTERNS)
    assert (await get_layer_status(db_url, name=layer))["archived_patterns"] == []


async def test_tool_audits_patterns_only(server, db_url, layer, as_admin):
    await call_tool(server, "strata_set_archived_patterns", layer_name=layer, patterns=PATTERNS)
    await call_tool(server, "strata_set_archived_patterns", layer_name=layer, patterns=[])

    assert await _audit_entries(db_url, layer) == [
        {"layer": layer, "patterns": []},
        {"layer": layer, "patterns": PATTERNS},
    ]
    entries = await query_audit_log(db_url, tool_name="strata_set_archived_patterns", limit=1)
    assert entries[0]["action"] == "set_archived_patterns"
    assert entries[0]["result_summary"] is None


class TestFieldForCaller:
    LAYER = {"name": "x", "origin": "manual", "archived_patterns": PATTERNS}

    def _patched(self, role: str):
        return (
            patch("stratakb.core.auth.get_current_user", return_value=("u", role)),
            patch("stratakb.knowledge.tools.get_current_user", return_value=("u", role)),
            patch(
                "stratakb.knowledge.tools.list_layers",
                new_callable=AsyncMock,
                return_value=[dict(self.LAYER)],
            ),
            patch(
                "stratakb.knowledge.tools.get_layer_status",
                new_callable=AsyncMock,
                return_value=dict(self.LAYER),
            ),
        )

    @pytest.mark.parametrize("role", ["viewer", "operator"])
    async def test_below_admin_does_not_see_patterns(self, server, role):
        auth, tools, listed_layers, layer_status = self._patched(role)
        with auth, tools, listed_layers, layer_status:
            listed = await call_tool(server, "strata_list_knowledge_layers")
            status = await call_tool(server, "strata_get_layer_status", name="x")

        for layer in (listed["layers"][0], status["layer"]):
            assert layer == {"name": "x", "origin": "manual"}

    async def test_admin_sees_patterns(self, server):
        auth, tools, listed_layers, layer_status = self._patched("admin")
        with auth, tools, listed_layers, layer_status:
            listed = await call_tool(server, "strata_list_knowledge_layers")
            status = await call_tool(server, "strata_get_layer_status", name="x")

        for layer in (listed["layers"][0], status["layer"]):
            assert layer == self.LAYER


async def test_create_layer_stores_patterns(db_url, cleanup):
    name = unique_layer_name()
    cleanup.append(name)
    created = await create_layer(db_url, name=name, display_name="x", archived_patterns=PATTERNS)
    assert created["archived_patterns"] == PATTERNS
    assert (await get_layer_status(db_url, name=name))["archived_patterns"] == PATTERNS


async def test_create_layer_refuses_bad_patterns(db_url, cleanup):
    name = unique_layer_name()
    cleanup.append(name)
    with pytest.raises(LayerValidationError, match="archived files pattern"):
        await create_layer(db_url, name=name, display_name="x", archived_patterns=["!"])


async def test_applies_missing_16_existing_layers_have_no_patterns(db_url, restore_schema):
    await owner_execute(
        db_url,
        """
        DELETE FROM strata_core.knowledge_layers WHERE name = 'test_mig_archived';
        ALTER TABLE strata_core.knowledge_layers DROP COLUMN archived_patterns;
        DELETE FROM strata_core.schema_migrations WHERE version = 16;
        INSERT INTO strata_core.knowledge_layers (name, display_name)
        VALUES ('test_mig_archived', 'Older layer');
        """,
    )
    try:
        await migrations.ensure_schema(db_url)

        conn = await asyncpg.connect(db_url)
        try:
            patterns = await conn.fetchval(
                "SELECT archived_patterns::text FROM strata_core.knowledge_layers "
                "WHERE name = 'test_mig_archived'"
            )
        finally:
            await conn.close()
        assert patterns == "[]"
        assert await recorded_versions(db_url) == {m.version for m in migrations.MIGRATIONS}
    finally:
        await owner_execute(
            db_url,
            "DELETE FROM strata_core.knowledge_layers WHERE name = 'test_mig_archived'",
        )
