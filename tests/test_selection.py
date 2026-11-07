"""Tests for file selection: git mode, walk mode, ignore files, case folding. No database needed."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from stratakb.knowledge.selection import select_files

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


pytestmark = pytest.mark.usefixtures("isolated_git")


def _write(path: Path, text: str = "# doc\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _symlink(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links can't be created here")


def _paths(result) -> list[str]:
    return [e.relative_path for e in result.entries]


def _excluded_paths(result) -> set[str]:
    return {path for path, _ in result.excluded}


@needs_git
def test_git_mode_selection(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _write(root / "tracked.md")
    _write(root / "untracked.md")
    _write(root / "ignored.md")
    _write(root / ".gitignore", "ignored.md\n")
    _write(root / ".strataignore", "private/\n")
    _write(root / "private" / "p.md")
    _write(root / "build" / "x.md")
    (root / "sub").mkdir()
    _git(root, "init", "-q", "nested")
    _write(root / "nested" / "n.md")
    _git(root, "add", "tracked.md", ".gitignore")
    _git(root, "add", "-f", "build/x.md")
    _git(
        root, "update-index", "--add", "--cacheinfo",
        "160000,1234567890123456789012345678901234567890,sub",
    )

    result = select_files(root)

    assert result.mode == "git"
    assert _paths(result) == ["tracked.md", "untracked.md"]
    assert ("build/", "builtin_skip") in result.excluded
    assert ("private/", "ignored_by_strataignore") in result.excluded
    assert [p for p, _ in result.excluded].count("build/") == 1
    assert ("build/x.md", "builtin_skip") not in result.excluded
    for gone in ("ignored.md", "sub", "sub/", "nested", "nested/", "nested/n.md"):
        assert gone not in _paths(result)
        assert gone not in _excluded_paths(result)
    assert result.nested_gitignores == 0
    assert result.notes == []
    assert result.entries[0].path == root.resolve() / "tracked.md"
    assert result.entries[0].size == (root / "tracked.md").stat().st_size
    assert result.entries[0].sha256 is None


@needs_git
def test_git_mode_symlink(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    target = _write(root / "real.md")
    _symlink(target, root / "link.md")

    result = select_files(root)

    assert result.mode == "git"
    assert ("link.md", "symlink") in result.excluded
    assert _paths(result) == ["real.md"]


@needs_git
def test_git_mode_ignores_fsmonitor_hook(tmp_path):
    """A folder's own .git/config must not make selection run a command."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _write(root / "doc.md")
    _git(root, "add", "doc.md")
    _git(root, "config", "core.fsmonitor", "echo PWNED > marker.txt; false")

    result = select_files(root)

    assert result.mode == "git"
    assert _paths(result) == ["doc.md"]
    assert not (root / "marker.txt").exists()


def test_walk_mode_gitignore_and_nested(tmp_path):
    _write(tmp_path / "keep.md")
    _write(tmp_path / "drop.md")
    _write(tmp_path / ".gitignore", "drop.md\n")
    _write(tmp_path / "a" / ".gitignore", "x.md\n")
    _write(tmp_path / "a" / "x.md")
    _write(tmp_path / "a" / "b" / ".gitignore", "*\n")

    result = select_files(tmp_path)

    assert result.mode == "walk"
    assert ("drop.md", "ignored_by_gitignore") in result.excluded
    # Nested .gitignore files are counted, not applied
    assert "a/x.md" in _paths(result)
    assert result.nested_gitignores == 2
    assert (
        "2 nested .gitignore files were not applied; "
        "run inside git or add the rules to .strataignore"
    ) in result.notes
    assert _paths(result) == ["a/x.md", "keep.md"]


def test_walk_mode_skip_list(tmp_path):
    _write(tmp_path / "doc.md")
    _write(tmp_path / "node_modules" / "pkg" / "readme.md")
    _write(tmp_path / ".cache" / "c.md")
    _write(tmp_path / "docs" / "__pycache__" / "x.md")

    result = select_files(tmp_path)

    assert _paths(result) == ["doc.md"]
    assert ("node_modules/", "builtin_skip") in result.excluded
    assert (".cache/", "builtin_skip") in result.excluded
    assert ("docs/__pycache__/", "builtin_skip") in result.excluded
    assert not any(p.startswith(("node_modules/", ".cache/")) and p.endswith(".md")
                   for p in _excluded_paths(result))


def test_walk_mode_symlinked_folder(tmp_path):
    root = tmp_path / "root"
    _write(root / "doc.md")
    outside = tmp_path / "outside"
    _write(outside / "o.md")
    _symlink(outside, root / "linked")

    result = select_files(root)

    assert ("linked/", "symlink") in result.excluded
    assert _paths(result) == ["doc.md"]


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are a Windows feature")
def test_walk_mode_junction_is_a_link(tmp_path):
    import _winapi

    root = tmp_path / "root"
    _write(root / "doc.md")
    outside = tmp_path / "outside"
    _write(outside / "o.md")
    _winapi.CreateJunction(str(outside), str(root / "joined"))

    result = select_files(root)

    assert ("joined/", "symlink") in result.excluded
    assert _paths(result) == ["doc.md"]


def test_strataignore_excludes_folder(tmp_path):
    _write(tmp_path / "docs" / "a.md")
    _write(tmp_path / "keep.md")
    _write(tmp_path / ".strataignore", "docs/\n")

    result = select_files(tmp_path)

    assert ("docs/", "ignored_by_strataignore") in result.excluded
    assert "docs/a.md" not in _excluded_paths(result)
    assert _paths(result) == ["keep.md"]


def test_strataignore_negation_reincludes_dot_folder(tmp_path):
    _write(tmp_path / ".github" / "workflows" / "ci.yml", "on: push\n")
    _write(tmp_path / ".strataignore", "!.github/\n")

    result = select_files(tmp_path)

    assert ".github/workflows/ci.yml" in _paths(result)
    assert (".github/", "builtin_skip") not in result.excluded
    # The ignore file itself starts with "." and stays skipped
    assert (".strataignore", "builtin_skip") in result.excluded


@needs_git
def test_strataignore_negation_in_git_mode(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _write(root / ".github" / "workflows" / "ci.yml", "on: push\n")
    _write(root / ".strataignore", "!.github/\n")

    result = select_files(root)

    assert result.mode == "git"
    assert ".github/workflows/ci.yml" in _paths(result)


@pytest.mark.parametrize("platform, folds", [("win32", True), ("darwin", True), ("linux", False)])
def test_case_folding_by_platform(tmp_path, monkeypatch, platform, folds):
    _write(tmp_path / "Node_Modules" / "x.md")
    _write(tmp_path / "docs" / "d.md")
    _write(tmp_path / "Notes" / "Guide.md")
    _write(tmp_path / ".strataignore", "Docs/\n")

    with monkeypatch.context() as m:
        m.setattr(sys, "platform", platform)
        result = select_files(tmp_path)

    assert "Notes/Guide.md" in _paths(result)
    if folds:
        assert ("Node_Modules/", "builtin_skip") in result.excluded
        assert ("docs/", "ignored_by_strataignore") in result.excluded
        assert _paths(result) == ["Notes/Guide.md"]
    else:
        assert "Node_Modules/x.md" in _paths(result)
        assert "docs/d.md" in _paths(result)


def test_git_fallback_note(tmp_path):
    _write(tmp_path / "a.md")
    stderr = b"fatal: not a git repository (or any of the parent directories): .git\n"

    def fake_run(args, **_):
        return subprocess.CompletedProcess(args, 128, b"", stderr)

    with patch("stratakb.knowledge.selection.subprocess.run", side_effect=fake_run):
        result = select_files(tmp_path)

    assert result.mode == "walk"
    assert any("fatal: not a git repository" in note for note in result.notes)
    assert _paths(result) == ["a.md"]


def test_git_missing_note(tmp_path):
    _write(tmp_path / "a.md")

    with patch("stratakb.knowledge.selection.subprocess.run", side_effect=FileNotFoundError):
        result = select_files(tmp_path)

    assert result.mode == "walk"
    assert result.notes == ["not using git: git was not found"]


def test_not_a_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        select_files(_write(tmp_path / "a.md"))
