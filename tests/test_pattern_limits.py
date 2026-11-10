"""Limits on the text of layer patterns, and the messages about them.

Every write path refuses, with a LayerValidationError, a routing, structure
or archived files pattern the database cannot store (a lone surrogate, a
NUL), one longer than MAX_PATTERN_LENGTH, and one Python warns it will read
differently later. The question-time warning about a stored structure
pattern names its layer, and a preset's YAML error hints at quoting only
when a value starts with * or !.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from structlog.testing import capture_logs

from stratakb.knowledge.layers import (
    MAX_PATTERN_LENGTH,
    LayerValidationError,
    _validate_archived_patterns,
    _validate_routing_patterns,
    _validate_structure_patterns,
    create_layer,
)
from stratakb.knowledge.presets import load_preset
from stratakb.rag.clause_lookup import question_regexes

# The refused patterns warn when compiled before they are refused
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

VALIDATORS = [
    pytest.param(_validate_archived_patterns, "", id="archived"),
    pytest.param(_validate_routing_patterns, "", id="routing"),
    pytest.param(_validate_structure_patterns, "^", id="structure"),
]


# ── Text the database cannot store ───────────────────────


@pytest.mark.parametrize(("validate", "prefix"), VALIDATORS)
def test_lone_surrogate_is_refused(validate, prefix):
    with pytest.raises(LayerValidationError, match="lone surrogate") as e:
        validate([prefix + "a\ud800"])
    # The message reaches a tool reply as JSON text
    str(e.value).encode("utf-8")


@pytest.mark.parametrize(
    ("validate", "prefix"),
    [p for p in VALIDATORS if p.id != "archived"],
)
def test_nul_is_refused(validate, prefix):
    with pytest.raises(LayerValidationError, match="NUL"):
        validate([prefix + "a\x00"])


# ── Length ───────────────────────────────────────────────


@pytest.mark.parametrize(("validate", "prefix"), VALIDATORS)
def test_pattern_over_the_cap_is_refused(validate, prefix):
    pattern = prefix + "a" * (MAX_PATTERN_LENGTH + 1 - len(prefix))
    with pytest.raises(LayerValidationError, match="longer than 500 characters") as e:
        validate([pattern])
    # Only the start of the pattern is shown
    assert len(str(e.value)) < 200


@pytest.mark.parametrize(("validate", "prefix"), VALIDATORS)
def test_pattern_at_the_cap_is_accepted(validate, prefix):
    pattern = prefix + "a" * (MAX_PATTERN_LENGTH - len(prefix))
    assert validate([pattern]) == [pattern]


def test_cap_is_500_characters():
    assert MAX_PATTERN_LENGTH == 500


# ── Patterns Python reads differently later ──────────────


@pytest.mark.parametrize("pattern", ["[[:alpha:]]", "docs/[[:alpha:]]*.md"])
def test_posix_class_is_refused_as_archived_pattern(pattern):
    with pytest.raises(LayerValidationError, match=r"\.gitignore has no class"):
        _validate_archived_patterns([pattern])
    # A second check in the same process still refuses it (re caches compiles)
    with pytest.raises(LayerValidationError, match=r"\.gitignore has no class"):
        _validate_archived_patterns([pattern])


@pytest.mark.parametrize("pattern", ["[[:alpha:]]", "[a&&b]"])
def test_future_warning_is_refused_as_routing_pattern(pattern):
    for _ in range(2):
        with pytest.raises(LayerValidationError, match="escape it with a backslash"):
            _validate_routing_patterns([pattern])


def test_future_warning_is_refused_as_structure_pattern():
    with pytest.raises(LayerValidationError, match="later Python"):
        _validate_structure_patterns(["^[[:alpha:]]"])


def test_escaped_bracket_is_accepted():
    assert _validate_routing_patterns([r"[\[:alpha:]]"]) == [r"[\[:alpha:]]"]
    assert _validate_archived_patterns(["docs/*.md"]) == ["docs/*.md"]


async def test_create_layer_refuses_before_connecting():
    with pytest.raises(LayerValidationError, match="lone surrogate"):
        await create_layer(
            "postgresql://unused",
            name="surrogate_layer",
            display_name="x",
            routing_patterns=["\ud800"],
        )


# ── The question-time warning names the layer ────────────


def _unsafe() -> str:
    # Unique per test: _question_regex is cached, and a cached entry warns no more
    return rf"^(?P<label>(a{uuid4().hex}+)+)"


def test_unsafe_structure_pattern_warning_names_the_layer():
    with capture_logs() as logs:
        assert question_regexes({"layer_x": [_unsafe()]}) == []

    warnings = [e for e in logs if e["event"] == "unsafe_structure_pattern"]
    assert [e["layer"] for e in warnings] == ["layer_x"]


def test_unsafe_pattern_in_two_layers_is_warned_for_each():
    pattern = _unsafe()
    with capture_logs() as logs:
        question_regexes({"layer_a": [pattern], "layer_b": [pattern]})

    warnings = [e for e in logs if e["event"] == "unsafe_structure_pattern"]
    assert sorted(e["layer"] for e in warnings) == ["layer_a", "layer_b"]


def test_safe_pattern_in_two_layers_is_returned_once():
    pattern = rf"^(?P<label>AT{uuid4().hex[:6]}-\d+)"
    compiled = question_regexes({"layer_a": [pattern, 7], "layer_b": [pattern, ["x"]]})

    assert len(compiled) == 1
    assert isinstance(compiled[0], re.Pattern)


async def test_exact_clauses_hands_compiled_patterns_to_the_lookup():
    from stratakb.rag.tools import _exact_clauses

    lookup = AsyncMock(return_value="matches")
    with (
        patch(
            "stratakb.knowledge.layers.get_structure_patterns",
            AsyncMock(return_value={"a": [r"^(?P<label>AT-\d+)"]}),
        ),
        patch("stratakb.rag.tools.lookup_clauses", lookup),
    ):
        assert await _exact_clauses("postgresql://unused", "show AT-44", ["a"], 100) == "matches"

    patterns = lookup.await_args.args[-1]
    assert len(patterns) == 1
    assert isinstance(patterns[0], re.Pattern)
    assert patterns[0].search("show AT-44")


# ── The YAML hint ────────────────────────────────────────


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "preset.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_tab_indent_gets_no_quoting_hint(tmp_path):
    path = _write(
        tmp_path,
        "name: tab_preset\ndisplay_name: Tab\nsources:\n\t- {name: docs, path: docs}\n",
    )
    with pytest.raises(LayerValidationError, match="not valid YAML") as e:
        load_preset(path)
    assert "quote a pattern" not in str(e.value)


def test_block_list_star_gets_the_quoting_hint(tmp_path):
    path = _write(
        tmp_path,
        "name: star_preset\ndisplay_name: Star\narchived_patterns:\n  - *.md\n"
        "sources:\n  - {name: docs, path: docs}\n",
    )
    with pytest.raises(LayerValidationError, match="quote a pattern"):
        load_preset(path)


@pytest.mark.parametrize("preset", ["sepa", "iso20022", "peps"])
def test_example_presets_still_load(preset):
    assert load_preset(EXAMPLES / preset / "preset.yaml").name
