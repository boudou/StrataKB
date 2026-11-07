"""Tests for how stratakb add writes its report: the text and JSON forms, and the
escaping of what a report line shows. No database is needed."""

from __future__ import annotations

import argparse
import json

from stratakb import cli
from stratakb.core.terminal_text import unprintable
from stratakb.knowledge import add_folder
from stratakb.knowledge.add_folder import REPORT_KEYS
from tests.helpers import ROOT, add_report


def test_json_keys_match_the_interface(capsys):
    assert cli._finish(add_report(), argparse.Namespace(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert tuple(data) == REPORT_KEYS
    assert data["counts"]["added"] == 3


def test_json_keeps_non_ascii_paths():
    path = "Ünïcode/ταυτότητα.md"
    report = add_report(files=[{"path": path, "outcome": "added", "reason": None}])
    text = cli.render_json(report)
    assert "ταυτότητα.md" in text
    assert tuple(json.loads(text)) == REPORT_KEYS


def test_json_escapes_c1_del_and_bidi_in_paths(capsys):
    # A one-byte CSI, DEL and a right-to-left override: json.dumps leaves them raw
    path = "evil\x9b2J\x7f‮dm.md"
    report = add_report(files=[{"path": path, "outcome": "added", "reason": None}])
    assert cli._finish(report, argparse.Namespace(json=True)) == 0
    out = capsys.readouterr().out
    for raw in ("\x9b", "\x7f", "‮"):
        assert raw not in out
    for escape in ("\\u009b", "\\u007f", "\\u202e"):
        assert escape in out
    assert json.loads(out)["files"][0]["path"] == path


def test_refusal_json_shape():
    report = add_folder.refused("broad_path", "Pass --allow-broad.", root=ROOT, layer="docs")
    data = json.loads(cli.render_json(report))
    assert tuple(data) == REPORT_KEYS
    assert data["counts"] == dict.fromkeys(add_folder.OUTCOME_KEYS, 0)
    assert data["files"] == [] and data["excluded"] == []
    assert data["created"] is False
    assert data["reason"] == "broad_path"


def test_text_groups_by_reason_with_and_more():
    files = [{"path": f"src/m{i}.py", "outcome": "skipped", "reason": "code"} for i in range(7)]
    files.append({"path": ".env.local", "outcome": "skipped", "reason": "secret"})
    files.append({"path": "bad.pdf", "outcome": "failed", "reason": "parse_error"})
    counts = dict.fromkeys(add_folder.OUTCOME_KEYS, 0)
    counts.update(skipped=8, failed=1)
    report = add_report(
        counts=counts,
        files=files,
        excluded=[{"path": ".github/", "reason": "builtin_skip"}],
        notes=["parse_error: not a PDF"],
        exit_code=1,
    )
    text = cli.render_text(report)
    assert "  8 skipped" in text
    assert "  1 failed" in text
    assert "added" not in text  # zero counts are left out
    assert "  code (7): src/m0.py, src/m1.py, src/m2.py, src/m3.py, src/m4.py and 2 more" in text
    assert "  secret (1): .env.local" in text
    assert "  builtin_skip (1): .github/" in text
    assert "  bad.pdf (parse_error)" in text
    assert "  parse_error: not a PDF" in text
    assert text.rstrip().endswith("what do the docs in docs say about …?")


def test_text_quiet_rerun():
    counts = dict.fromkeys(add_folder.OUTCOME_KEYS, 0)
    counts["unchanged"] = 4
    text = cli.render_text(add_report(created=False, counts=counts, files=[]))
    assert text.startswith("Synced layer docs in the database at db.example:5433")
    assert "  4 unchanged" in text


def _flagged_report():
    counts = dict.fromkeys(add_folder.OUTCOME_KEYS, 0)
    counts.update(flagged=2)
    files = [
        {"path": "codes.csv", "outcome": "flagged", "reason": "truncated", "count": 1234},
        {"path": "scan.pdf", "outcome": "flagged", "reason": "probably_scanned", "count": None},
    ]
    return add_report(
        counts=counts, files=files, excluded=[{"path": ".github/", "reason": "builtin_skip"}]
    )


def test_text_lists_flagged_files_after_excluded():
    lines = cli.render_text(_flagged_report()).split("\n")
    start = lines.index("Flagged:")
    assert lines[start - 2:start + 3] == [
        "Excluded:",
        "  builtin_skip (1): .github/",
        "Flagged:",
        "  codes.csv (truncated, 1,234 left out)",
        "  scan.pdf (probably_scanned)",
    ]


def test_text_has_no_flagged_section_without_flagged_files():
    assert "Flagged" not in cli.render_text(add_report())


def test_json_carries_the_flagged_rows():
    report = _flagged_report()
    assert json.loads(cli.render_json(report))["files"] == report.files


# A name that would recolour, break, overwrite and reverse a report line
HOSTILE = "a\x1b[31m\n\r‮b"
HOSTILE_SHOWN = "a\\x1b[31m\\n\\x0d\\u202eb"


def _assert_clean(text: str) -> None:
    assert not any(unprintable(c) for c in text)


def test_text_escapes_file_names_and_keeps_one_line_per_file():
    counts = dict.fromkeys(add_folder.OUTCOME_KEYS, 0)
    counts.update(skipped=1, flagged=1, failed=1)
    files = [
        {"path": f"skipped{HOSTILE}", "outcome": "skipped", "reason": "code"},
        {"path": f"flagged{HOSTILE}", "outcome": "flagged", "reason": "truncated", "count": 5},
        {"path": f"failed{HOSTILE}", "outcome": "failed", "reason": "parse_error"},
    ]
    excluded = [{"path": f"excluded{HOSTILE}", "reason": "builtin_skip"}]
    text = cli.render_text(add_report(counts=counts, files=files, excluded=excluded))
    _assert_clean(text)
    lines = text.split("\n")
    assert lines[lines.index("Skipped:") + 1] == f"  code (1): skipped{HOSTILE_SHOWN}"
    assert lines[lines.index("Excluded:") + 1] == f"  builtin_skip (1): excluded{HOSTILE_SHOWN}"
    assert lines[lines.index("Flagged:") + 1] == f"  flagged{HOSTILE_SHOWN} (truncated, 5 left out)"
    assert lines[lines.index("Failed:") + 1] == f"  failed{HOSTILE_SHOWN} (parse_error)"
    assert lines[lines.index("Failed:") + 2].startswith("Next: ")


def test_text_escapes_every_other_field():
    files = [{"path": "a.txt", "outcome": "failed", "reason": f"why{HOSTILE}"}]
    text = cli.render_text(
        add_report(
            layer=f"layer{HOSTILE}",
            root=f"root{HOSTILE}",
            db_target=f"db{HOSTILE}",
            mode=f"mode{HOSTILE}",
            files=files,
            excluded=[{"path": "b.txt", "reason": f"rule{HOSTILE}"}],
            notes=[f"note{HOSTILE}"],
        )
    )
    _assert_clean(text)
    lines = text.split("\n")
    assert lines[0] == (
        f"Created layer layer{HOSTILE_SHOWN} in the database at db{HOSTILE_SHOWN}"
    )
    assert lines[1] == f"Folder: root{HOSTILE_SHOWN} (mode{HOSTILE_SHOWN} mode)"
    assert f"  a.txt (why{HOSTILE_SHOWN})" in lines
    assert f"  rule{HOSTILE_SHOWN} (1): b.txt" in lines
    assert f"  note{HOSTILE_SHOWN}" in lines
    assert lines[-1].endswith(f"docs in layer{HOSTILE_SHOWN} say about …?")


def _hostile_refusal():
    return add_folder.refused("name_taken", f"hint{HOSTILE}", root=ROOT, layer=f"layer{HOSTILE}")


def test_refusal_text_is_escaped_on_one_line(capsys):
    assert cli._finish(_hostile_refusal(), argparse.Namespace(json=False)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"Not added (name_taken), layer layer{HOSTILE_SHOWN}: hint{HOSTILE_SHOWN}\n"
    )


def test_refusal_without_a_hint_is_escaped():
    report = add_folder.refused("name_taken", "", root=ROOT, layer=f"layer{HOSTILE}")
    text = cli.render_text(report)
    assert text == f"Not added (name_taken), layer layer{HOSTILE_SHOWN}"


def test_refusal_hint_is_escaped_on_stderr_with_json(capsys):
    assert cli._finish(_hostile_refusal(), argparse.Namespace(json=True)) == 2
    captured = capsys.readouterr()
    assert captured.err == f"hint{HOSTILE_SHOWN}\n"
    # The JSON stays a valid form of the same values
    assert json.loads(captured.out)["layer"] == f"layer{HOSTILE}"


def test_text_for_ordinary_names_is_unchanged():
    counts = dict.fromkeys(add_folder.OUTCOME_KEYS, 0)
    counts.update(added=1, skipped=1, flagged=1, failed=1)
    files = [
        {"path": "notes/a.md", "outcome": "added", "reason": None},
        {"path": "src/my file (1).py", "outcome": "skipped", "reason": "code"},
        {"path": "συμβόλαια/σύμβαση.csv", "outcome": "flagged", "reason": "truncated", "count": 7},
        {"path": "back\\slash…\U0001f600.pdf", "outcome": "failed", "reason": "parse_error"},
    ]
    report = add_report(
        counts=counts,
        files=files,
        excluded=[{"path": ".github/", "reason": "builtin_skip"}],
        notes=["parse_error: not a PDF"],
    )
    assert cli.render_text(report) == "\n".join(
        [
            "Created layer docs in the database at db.example:5433",
            f"Folder: {ROOT} (git mode)",
            "  1 added",
            "  1 skipped",
            "  1 flagged",
            "  1 failed",
            "Skipped:",
            "  code (1): src/my file (1).py",
            "Excluded:",
            "  builtin_skip (1): .github/",
            "Flagged:",
            "  συμβόλαια/σύμβαση.csv (truncated, 7 left out)",
            "Failed:",
            "  back\\slash…\U0001f600.pdf (parse_error)",
            "Notes:",
            "  parse_error: not a PDF",
            "Next: ask in your MCP client: what do the docs in docs say about …?",
        ]
    )
    refusal = add_folder.refused("broad_path", "Pass --allow-broad.", root=ROOT, layer="docs")
    assert cli.render_text(refusal) == "Not added (broad_path), layer docs: Pass --allow-broad."
