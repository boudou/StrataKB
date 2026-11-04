"""File selection: which files of a folder are candidates for ingestion.

Inside a git work tree the list comes from git (tracked files plus untracked
files that are not ignored; submodules are dropped). Elsewhere the folder is
walked and only the root .gitignore applies. Both modes then apply the same
rules: symbolic links are excluded, a built-in skip list of folder names
(dot folders, node_modules, venv, ...) is excluded, and the root
.strataignore applies. Folders are decided top-down, as git does, and an
excluded folder is reported once, as "dir/", with none of the files below it.

No database access, no logging: the caller reports the result.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pathspec import GitIgnoreSpec

BUILTIN_SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    "site-packages",
    "bower_components",
    "vendor",
})

_GIT_TIMEOUT_SECONDS = 60
_SUBMODULE_MODE = "160000"


@dataclass(frozen=True)
class FileEntry:
    """One selected file: its POSIX path relative to the root, size and path."""

    relative_path: str
    size: int
    path: Path
    sha256: str | None = None


@dataclass
class SelectionResult:
    """The selected files, the excluded paths with a reason, and notes."""

    entries: list[FileEntry] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)
    nested_gitignores: int = 0
    mode: Literal["git", "walk"] = "walk"
    notes: list[str] = field(default_factory=list)


class _GitUnavailableError(Exception):
    """git can't list the folder; the message is the note for the result."""


def _fold_enabled() -> bool:
    # Read at call time so tests can patch sys.platform
    return sys.platform in ("win32", "darwin")


def _read_spec(path: Path, fold: bool = False) -> GitIgnoreSpec:
    """Build a spec from an ignore file; a missing file gives an empty spec."""
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except FileNotFoundError:
        lines = []
    if fold:
        lines = [line.casefold() for line in lines]
    return GitIgnoreSpec.from_lines(lines)


def _git(root: Path, *args: str) -> bytes:
    """Run one git command in root and return its stdout, or raise _GitUnavailableError.

    The folder's own .git/config is untrusted: core.fsmonitor there names a command
    that reading the index would run, so it is switched off on every call.
    """
    try:
        proc = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        raise _GitUnavailableError("not using git: git was not found") from None
    except subprocess.TimeoutExpired:
        raise _GitUnavailableError("not using git: git timed out") from None
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        first = next((line.strip() for line in stderr.splitlines() if line.strip()), "")
        reason = first or f"git exited with {proc.returncode}"
        raise _GitUnavailableError(f"not using git: {reason}")
    return proc.stdout or b""


def _split_z(output: bytes) -> list[str]:
    return [p.decode("utf-8", errors="surrogateescape") for p in output.split(b"\0") if p]


def _git_paths(root: Path) -> list[str]:
    """List the candidate paths through git, or raise _GitUnavailableError."""
    inside = _git(root, "rev-parse", "--is-inside-work-tree").decode(
        "utf-8", errors="replace"
    ).strip()
    if inside != "true":
        raise _GitUnavailableError("not using git: not inside a work tree")
    listed = _split_z(_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"))
    submodules = set()
    for line in _split_z(_git(root, "ls-files", "-z", "-s")):
        # "<mode> <object> <stage>\t<path>"
        meta, _, path = line.partition("\t")
        if meta.split(" ", 1)[0] == _SUBMODULE_MODE:
            submodules.add(path)
    # Unmerged index stages list a path more than once: keep the first
    unique = dict.fromkeys(p for p in listed if p not in submodules and not p.endswith("/"))
    return list(unique)


class _Rules:
    """The exclusion rules of one selection, shared by both modes."""

    def __init__(self, root: Path, *, walk: bool) -> None:
        self.root = root
        self.fold = _fold_enabled()
        self.strataignore = _read_spec(root / ".strataignore", fold=self.fold)
        self.gitignore = _read_spec(root / ".gitignore") if walk else None

    def _folded(self, value: str) -> str:
        return value.casefold() if self.fold else value

    def decide(self, rel: str, *, is_dir: bool) -> str | None:
        """Return the exclusion reason for one path, or None when it is kept.

        rel is the POSIX path relative to the root, without a trailing slash.
        """
        path = self.root / rel
        if path.is_symlink() or path.is_junction():
            return "symlink"
        # pathspec matches a folder pattern only against a key ending in "/"
        key = f"{rel}/" if is_dir else rel
        if self.gitignore is not None and self.gitignore.check_file(key).include is True:
            return "ignored_by_gitignore"
        include = self.strataignore.check_file(self._folded(key)).include
        if include is True:
            return "ignored_by_strataignore"
        if include is False:
            # Re-included by a "!" pattern: overrides the built-in skip list
            return None
        name = rel.rsplit("/", 1)[-1]
        if name.startswith(".") or self._folded(name) in BUILTIN_SKIP_DIRS:
            return "builtin_skip"
        return None


def _add_entry(result: SelectionResult, root: Path, rel: str) -> None:
    path = root / rel
    try:
        if not path.is_file():
            return
        size = path.stat().st_size
    except OSError:
        # Vanished while the folder was being listed
        return
    result.entries.append(FileEntry(relative_path=rel, size=size, path=path))


def _select_git(root: Path, paths: list[str], result: SelectionResult) -> None:
    rules = _Rules(root, walk=False)
    folders: dict[str, bool] = {}  # folder -> kept
    for rel in paths:
        parts = rel.split("/")
        kept = True
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            if prefix not in folders:
                reason = rules.decide(prefix, is_dir=True)
                folders[prefix] = reason is None
                if reason is not None:
                    result.excluded.append((f"{prefix}/", reason))
            if not folders[prefix]:
                kept = False
                break
        if not kept:
            continue
        reason = rules.decide(rel, is_dir=False)
        if reason is not None:
            result.excluded.append((rel, reason))
            continue
        _add_entry(result, root, rel)


def _select_walk(root: Path, result: SelectionResult) -> None:
    rules = _Rules(root, walk=True)

    def relative(path: str) -> str:
        try:
            return Path(path).relative_to(root).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def on_error(error: OSError) -> None:
        result.notes.append(f"could not list {relative(error.filename)}/: {error.strerror}")

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=on_error):
        base = Path(dirpath).relative_to(root).as_posix()
        prefix = "" if base == "." else f"{base}/"
        kept_dirs = []
        for name in dirnames:
            rel = f"{prefix}{name}"
            reason = rules.decide(rel, is_dir=True)
            if reason is None:
                kept_dirs.append(name)
            else:
                result.excluded.append((f"{rel}/", reason))
        dirnames[:] = kept_dirs
        for name in filenames:
            rel = f"{prefix}{name}"
            if name == ".gitignore" and prefix:
                result.nested_gitignores += 1
            reason = rules.decide(rel, is_dir=False)
            if reason is not None:
                result.excluded.append((rel, reason))
                continue
            _add_entry(result, root, rel)

    n = result.nested_gitignores
    if n > 0:
        result.notes.append(
            f"{n} nested .gitignore files were not applied; "
            "run inside git or add the rules to .strataignore"
        )


def select_files(root: Path) -> SelectionResult:
    """Select the candidate files of a folder, through git when it is a work tree.

    Raises NotADirectoryError when root is not a folder, and PermissionError
    when the root itself can't be listed.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    # List the root first, so an unreadable root raises instead of giving nothing
    with os.scandir(root):
        pass

    result = SelectionResult()
    try:
        paths = _git_paths(root)
    except _GitUnavailableError as e:
        result.mode = "walk"
        result.notes.append(str(e))
        _select_walk(root, result)
    else:
        result.mode = "git"
        _select_git(root, paths, result)

    result.entries.sort(key=lambda e: e.relative_path)
    result.excluded.sort(key=lambda item: item[0])
    return result
