"""The pull request rules check in scripts/check_repo_rules.py.

Paths are built by concatenation so this file holds none itself, and the
private terms are made up.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_repo_rules.py"
_spec = importlib.util.spec_from_file_location("check_repo_rules", SCRIPT)
rules = importlib.util.module_from_spec(_spec)
sys.modules["check_repo_rules"] = rules
_spec.loader.exec_module(rules)

DRIVE = "C" + ":"
TERMS = rules.private_terms("zorbix, Quillon Ltd\nfrobnitz")


def _diff(file: str, *added: str, start: int = 1) -> str:
    lines = [f"diff --git a/{file} b/{file}", f"--- a/{file}", f"+++ b/{file}",
             f"@@ -0,0 +{start},{len(added)} @@"]
    return "\n".join(lines + [f"+{line}" for line in added]) + "\n"


def test_clean_message_passes():
    message = "Keep the best full-text hits\n\nCloses #12\n"
    assert rules.check_message("commit", message, TERMS) == []


def test_human_co_author_passes():
    message = "Fix\n\nCo-authored-by: Jane Doe <jane@example.org>\n"
    assert rules.check_message("commit", message, TERMS) == []


def test_dependency_bot_co_author_passes():
    message = (
        "Bump x\n\n"
        "Co-authored-by: dependabot[bot] <1+dependabot[bot]@users.noreply.github.com>\n"
    )
    assert rules.check_message("commit", message, TERMS) == []


def test_bot_and_noreply_co_authors_are_flagged():
    message = (
        "Fix\n\n"
        "Co-authored-by: helper[bot] <12+helper[bot]@users.noreply.github.com>\n"
        "Co-authored-by: Some Tool <noreply@example.com>\n"
        "Co-authored-by: Writer AI <writer@example.com>\n"
    )
    findings = rules.check_message("commit", message, TERMS)
    assert len(findings) == 3
    assert all("attribution trailer" in f.message for f in findings)


def test_generated_line_is_flagged():
    message = "Fix\n\n-- Generated with Some Tool\n"
    findings = rules.check_message("pr body", message, TERMS)
    assert [f.message for f in findings] == ["generated-by line: -- Generated with Some Tool"]


def test_generated_inside_a_sentence_passes():
    message = "Store the generated with-clause in the query\n"
    assert rules.check_message("commit", message, TERMS) == []


def test_private_term_is_reported_without_the_term():
    findings = rules.check_message("pr title", "Port the ZORBIX parser", TERMS)
    assert len(findings) == 1
    assert "zorbix" not in findings[0].annotation().lower()


def test_private_term_matches_whole_words_only():
    assert rules.check_message("commit", "zorbixes and quillon", TERMS) == []
    assert rules.check_message("commit", "quillon ltd", TERMS)


def test_unset_terms_check_nothing():
    assert rules.private_terms(None) == []
    assert rules.private_terms(" , \n") == []


def test_added_lines_track_file_and_line_numbers():
    diff = _diff("a.py", "one", "two", start=10) + _diff("b.md", "three", start=4)
    assert list(rules.added_lines(diff)) == [
        ("a.py", 10, "one"), ("a.py", 11, "two"), ("b.md", 4, "three"),
    ]


def test_local_paths_in_added_lines_are_flagged():
    diff = _diff(
        "docs/x.md",
        "see " + DRIVE + "\\" + "Users\\jane\\notes.txt",
        "or " + DRIVE + "/" + "users/jane/notes",
        "or /" + "home/jane/notes",
        "or /" + "Users/jane/notes",
    )
    findings = rules.check_diff(diff, [])
    assert [f.line for f in findings] == [1, 2, 3, 4]
    assert all(f.file == "docs/x.md" for f in findings)


def test_non_local_paths_pass():
    diff = _diff(
        "docs/x.md",
        "postgresql://strata:pw@127.0.0.1:5433/strata",
        "https://example.org/a:b/c",
        "cache in ~/.cache/huggingface and /home/runner/work",
        "key: value",
        'ROOT = "' + DRIVE + '/work/docs"  # a made-up path in a test',
        DRIVE + "\\" + "Program Files\\App",
    )
    assert rules.check_diff(diff, []) == []


def test_private_terms_in_diff_report_location_only():
    diff = _diff("src/port.py", "x = 1", "# from the Zorbix code", start=7)
    findings = rules.check_diff(diff, TERMS)
    assert [(f.file, f.line) for f in findings] == [("src/port.py", 8)]
    assert "zorbix" not in findings[0].annotation().lower()


def test_private_file_name_is_never_printed():
    diff = _diff("src/frobnitz/port.py", "x = 1", "# from the Zorbix code")
    findings = rules.check_diff(diff, TERMS)
    assert len(findings) == 2
    for finding in findings:
        text = finding.annotation().lower()
        assert "frobnitz" not in text
        assert "zorbix" not in text
        assert finding.file is None


def test_deleted_files_are_ignored():
    diff = "--- a/old.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-zorbix\n"
    assert rules.check_diff(diff, TERMS) == []
