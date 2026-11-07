"""Tests for stratakb.core.terminal_text: one rule for unprintable characters."""

from __future__ import annotations

import json

import pytest

from stratakb.core.terminal_text import (
    one_line,
    printable_only,
    terminal_safe_json,
    unprintable,
    visible,
)

# C0, DEL, C1 (one-byte CSI), bidi overrides and isolates, a line separator,
# a zero-width space and a tag character above U+FFFF
HOSTILE = "\x1b\r\x08\x7f\x85\x9b​‪‮⁦⁩ \U000e0001"


@pytest.mark.parametrize("char", list(HOSTILE))
def test_unprintable(char):
    assert unprintable(char)


@pytest.mark.parametrize("char", ["a", " ", "\n", "\t", "é", "ταυ", "…", "\U0001f600"])
def test_printable_or_layout(char):
    assert not any(unprintable(c) for c in char)


def test_visible_escapes_and_keeps_layout():
    expected = "a\\x1b[2J\tb\nc\\x9b\\u202e\\U000e0001"
    assert visible("a\x1b[2J\tb\nc\x9b‮\U000e0001") == expected
    assert visible(42) == "42"


def test_one_line_also_escapes_layout():
    assert one_line("a\x1b[2J\tb\nc\r‮") == "a\\x1b[2J\\tb\\nc\\x0d\\u202e"
    assert one_line("back\\slash ταυ") == "back\\slash ταυ"
    assert one_line(42) == "42"


def test_printable_only_drops():
    assert printable_only("gpt\x1b[31m‮-4\n\t") == "gpt[31m-4"


@pytest.mark.parametrize("indent", [None, 2, "\t"])
def test_json_round_trips_with_no_raw_unprintable(indent):
    value = {"text": f"keep ταυ …{HOSTILE}\n\t\U0001f600", "n": [1, None, True]}
    out = terminal_safe_json(value, indent=indent)
    assert not any(unprintable(c) for c in out)
    assert json.loads(out) == value
    # Non-ASCII text stays readable
    assert "ταυ" in out and "\U0001f600" in out
    for escape in ("\\u009b", "\\u007f", "\\u202e", "\\udb40\\udc01"):
        assert escape in out
