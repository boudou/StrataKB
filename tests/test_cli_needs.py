"""Tests for the table of confirmation needs behind stratakb add, and the parts
of the command built from it. No database is needed."""

from __future__ import annotations

import json
from unittest.mock import patch

from stratakb import cli
from stratakb.knowledge import add_folder
from stratakb.knowledge.add_folder import Options, Plan


def test_every_need_has_a_flag_that_confirms_only_it():
    for need in cli._NEEDS:
        args = cli._parser().parse_args(["add", "docs", need.flag])
        assert args.confirmed == [need.name]
    assert cli._parser().parse_args(["add", "docs"]).confirmed is None
    assert Options().confirmed == frozenset()


def _both_needs() -> Plan:
    return Plan(root="r", needs={"large_run", "broad_path"}, broad_why="a drive root")


def test_confirmations_ask_broad_path_then_large_run():
    with (
        patch.object(cli, "_stdin_is_tty", return_value=True),
        patch.object(cli, "_confirm", return_value=True) as confirm,
    ):
        confirmed, refusal = cli._confirmations(_both_needs(), Options())
    assert [call.args[0] for call in confirm.call_args_list] == [
        "r is a broad path (a drive root). Add it anyway?",
        "This run adds or updates 0 files (0.0 MB) and removes 0. Continue?",
    ]
    assert confirmed == {"broad_path", "large_run"}
    assert refusal is None


def test_confirmations_refuse_at_the_first_unconfirmed_need():
    with patch.object(cli, "_stdin_is_tty", return_value=False):
        confirmed, refusal = cli._confirmations(_both_needs(), Options())
        assert confirmed == set()
        assert refusal is not None and refusal[0] == "broad_path"

        confirmed, refusal = cli._confirmations(
            _both_needs(), Options(confirmed=frozenset({"broad_path"}))
        )
        assert confirmed == {"broad_path"}
        assert refusal is not None and refusal[0] == "large_run"


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
