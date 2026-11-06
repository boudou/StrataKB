"""Tests for the table of confirmation needs behind stratakb add, and the parts
of the command built from it. No database is needed."""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import patch

from stratakb import cli
from stratakb.knowledge import add_folder
from stratakb.knowledge.add_folder import Options, Plan


def test_every_need_has_a_flag_that_sets_its_options_field():
    fields = {field.name for field in dataclasses.fields(Options)}
    for need in cli._NEEDS:
        assert need.option in fields
        assert getattr(Options(), need.option) is False
        args = cli._parser().parse_args(["add", "docs", need.flag])
        assert getattr(args, need.option) is True
        others = [other.option for other in cli._NEEDS if other is not need]
        assert not any(getattr(args, option) for option in others)


def test_needs_are_asked_in_table_order():
    assert [need.name for need in cli._NEEDS] == ["broad_path", "large_run"]


def test_plan_counts_files_added_or_updated():
    assert Plan(root="r").added_or_updated == 0
    plan = Plan(root="r", planned={"added": 2, "updated": 3, "unchanged": 7, "removed": 1})
    assert plan.added_or_updated == 5


def test_usage_error_refusal_hint_is_the_error_line(capsys):
    with patch.object(add_folder, "refused", wraps=add_folder.refused) as refused:
        assert cli.main(["add", "--json"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["reason"] == "usage_error"
    reason, hint = refused.call_args.args
    assert reason == "usage_error"
    assert hint.startswith("stratakb add: error: ")
    assert "\n" not in hint
    # stderr carries the usage too, ending with the same line
    assert captured.err.rstrip("\n").endswith(hint)
    assert captured.err.startswith("usage: ")
