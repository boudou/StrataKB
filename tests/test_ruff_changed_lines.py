"""Tests for scripts/ruff_changed_lines.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ruff_changed_lines", Path(__file__).parent.parent / "scripts" / "ruff_changed_lines.py"
)
rcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcl)

DIFF = """\
diff --git a/src/a.py b/src/a.py
index 1..2 100644
--- a/src/a.py
+++ b/src/a.py
@@ -3,0 +4,2 @@ def f():
+    x = 1
+    y = 2
@@ -10 +12 @@ def g():
-    old
+    new
@@ -20,3 +21,0 @@ def h():
-    gone
diff --git a/src/new.py b/src/new.py
new file mode 100644
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,3 @@
+import os
+import sys
+print(os, sys)
"""


def _finding(path: str, row: int, code: str = "F401") -> dict:
    return {"filename": path, "location": {"row": row, "column": 1}, "code": code, "message": "m"}


def test_changed_lines_reads_added_and_changed_rows():
    lines = rcl.changed_lines(DIFF)
    assert lines == {"src/a.py": {4, 5, 12}, "src/new.py": {1, 2, 3}}


def test_a_pure_deletion_marks_no_line():
    diff = "+++ b/x.py\n@@ -5,2 +4,0 @@\n-a\n-b\n"
    assert rcl.changed_lines(diff) == {"x.py": set()}


def test_only_findings_on_changed_lines_are_kept():
    changed = {"src/a.py": {4, 5, 12}}
    findings = [
        _finding("/repo/src/a.py", 4),
        _finding("/repo/src/a.py", 7),  # an older finding on an untouched line
        _finding("/repo/src/b.py", 4),  # a file the diff did not touch
    ]
    kept = rcl.findings_on_changed_lines(findings, changed, "/repo")
    assert [(f["filename"], f["location"]["row"]) for f in kept] == [("src/a.py", 4)]


def test_windows_paths_are_matched():
    kept = rcl.findings_on_changed_lines(
        [_finding("C:\\repo\\src\\a.py", 12)], {"src/a.py": {12}}, "C:\\repo"
    )
    assert len(kept) == 1
