"""Add a folder: turn one folder into one layer and keep it in sync.

plan() decides everything and writes nothing: the root, which layer (create,
adopt, sync, or a refusal with a reason code), which files go in and why the
others don't, and what needs confirming. apply() then creates or adopts the
layer, ingests the file list and writes one audit entry, and returns a
Report holding every key the command prints.

The service never prints, never reads stdin and logs nothing itself, so any
front end (the `stratakb add` command, a later push) can drive it. plan()
never loads the embedding model.
"""

from __future__ import annotations

import asyncio
import dataclasses
import getpass
import hashlib
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args
from uuid import UUID, uuid4

import asyncpg

from stratakb.core.audit import write_audit_log
from stratakb.core.urls import url_host_port
from stratakb.knowledge.ingestion import IngestionResult, Outcome, ingest_documents, tracked_files
from stratakb.knowledge.layers import (
    MAX_OWNER_LENGTH,
    NAME_PATTERN,
    LayerError,
    LayerNotFoundError,
    LayerValidationError,
    create_layer,
    find_layer_by_root,
    find_overlapping_roots,
    get_layer,
    is_under,
    layer_holds_folder,
    root_key,
    set_description,
    set_root_path,
)
from stratakb.knowledge.policy import Verdict, evaluate
from stratakb.knowledge.selection import (
    FileEntry,
    SelectionResult,
    _fold_enabled,
    select_files,
)

# A run above these asks for confirmation (large_run)
LARGE_RUN_FILES = 2000
LARGE_RUN_BYTES = 500_000_000
# ... and so does removing at least this many files, when that is more than half
LARGE_REMOVAL_MIN = 10

OUTCOME_KEYS = get_args(Outcome)

# The keys of the JSON report, in order: a stable interface
REPORT_KEYS = (
    "run_id",
    "layer",
    "created",
    "root",
    "mode",
    "counts",
    "skipped_by_reason",
    "excluded_by_reason",
    "files",
    "excluded",
    "notes",
    "reason",
    "exit_code",
)

_README_NAMES = ("readme.md", "readme.rst", "readme.txt", "readme")
_README_MAX_BYTES = 64 * 1024
_DESCRIPTION_MAX = 300
_DISPLAY_NAME_MAX = 200
_NAME_MAX = 100
_HASH_CHUNK = 1024 * 1024
_MAX_ERROR_NOTES = 3

_WINDOWS_SYSTEM_VARS = (
    "SystemRoot",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramData",
    "APPDATA",
    "LOCALAPPDATA",
)
_POSIX_SYSTEM_DIRS = (
    "/bin", "/boot", "/dev", "/etc", "/lib", "/opt", "/proc", "/sbin", "/sys",
    "/usr", "/var", "/System", "/Library", "/Applications",
)

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RULE_LINE = re.compile(r"^[=\-~^*+#_`]+$")


@dataclass
class Options:
    """What the caller asked for: the command's flags.

    confirmed holds the needs (as in Plan.needs) a flag already confirmed.
    """

    name: str | None = None
    description: str | None = None
    max_file_size_mb: float = 50.0
    confirmed: frozenset[str] = frozenset()


@dataclass
class Plan:
    """What apply() would do, or why it can't.

    action is None when refusal (reason, hint) is set. description is the
    description to write: a new layer's (given or derived), or on adopt and
    sync the --description value, None to keep the current one. selection is
    set on every plan apply() accepts; only a plan refused before selection
    lacks it.
    """

    root: str
    layer: str | None = None
    action: Literal["create", "adopt", "sync"] | None = None
    display_name: str = ""
    description: str | None = None
    selection: SelectionResult | None = None
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    entries: list[FileEntry] = field(default_factory=list)
    planned: dict[str, int] = field(default_factory=dict)
    work_files: int = 0
    work_bytes: int = 0
    removals: dict[str, str] = field(default_factory=dict)
    tracked: int = 0
    needs: set[str] = field(default_factory=set)
    refusal: tuple[str, str] | None = None
    broad_why: str | None = None
    layer_id: UUID | None = None

    @property
    def added_or_updated(self) -> int:
        """Files this run adds or updates: the ones it embeds."""
        return self.planned.get("added", 0) + self.planned.get("updated", 0)


@dataclass
class Report:
    """The outcome of one run. REPORT_KEYS are printed; hint and db_target are not."""

    run_id: str
    layer: str | None
    created: bool
    root: str
    mode: str | None
    counts: dict[str, int]
    skipped_by_reason: dict[str, int]
    excluded_by_reason: dict[str, int]
    files: list[dict]
    excluded: list[dict]
    notes: list[str]
    reason: str | None
    exit_code: int
    hint: str | None = None
    db_target: str = ""

    def as_dict(self) -> dict:
        """The printed keys, in REPORT_KEYS order."""
        return {key: getattr(self, key) for key in REPORT_KEYS}


def _zero_counts() -> dict[str, int]:
    return dict.fromkeys(OUTCOME_KEYS, 0)


def refused(
    reason: str,
    hint: str,
    *,
    root: str,
    layer: str | None = None,
    mode: str | None = None,
    db_target: str = "",
    exit_code: int = 2,
) -> Report:
    """The report of a refused run: nothing written, zero counts."""
    return Report(
        run_id=uuid4().hex,
        layer=layer,
        created=False,
        root=root,
        mode=mode,
        counts=_zero_counts(),
        skipped_by_reason={},
        excluded_by_reason={},
        files=[],
        excluded=[],
        notes=[],
        reason=reason,
        exit_code=exit_code,
        hint=hint,
        db_target=db_target,
    )


def resolve_owner(settings: Any) -> str:
    """STRATA_USER when it was set; otherwise the login name, marked unverified."""
    if "strata_user" in settings.model_fields_set:
        return settings.strata_user
    try:
        return "os:" + getpass.getuser()
    except Exception:
        # getuser() raises without a login name, as in some containers
        return "os:unknown"


def _system_folders() -> list[str]:
    if sys.platform == "win32":
        folders = []
        for var in _WINDOWS_SYSTEM_VARS:
            value = os.environ.get(var)
            if value:
                folders.append(value.replace("\\", "/").rstrip("/"))
        return folders
    return list(_POSIX_SYSTEM_DIRS)


def broad_reason(root: str) -> str | None:
    """Why root is too broad to add without --allow-broad, or None."""
    fold = _fold_enabled()
    stripped = root.replace("\\", "/").rstrip("/")
    if stripped == "" or re.fullmatch(r"[A-Za-z]:", stripped):
        return "a drive root"
    key = root_key(root, fold)
    home_key = root_key(Path.home().resolve().as_posix(), fold)
    if key == home_key:
        return "your home folder"
    if is_under(home_key, key):
        return "a folder above your home folder"
    for folder in _system_folders():
        if is_under(key, root_key(folder, fold)):
            return f"inside {folder}"
    return None


def is_broad(root: str) -> bool:
    """True for a drive root, the home folder, an ancestor of it, or a system folder."""
    return broad_reason(root) is not None


def slugify(folder_name: str, root: str | None = None) -> str:
    """A layer name from a folder name; a hash of root when nothing usable is left."""
    slug = re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")
    if slug[:1].isdigit():
        slug = "d_" + slug
    if len(slug) < 3:
        slug = slug.ljust(3, "0")
    if len(slug) > _NAME_MAX:
        slug = slug[:_NAME_MAX].rstrip("_")
    if not NAME_PATTERN.match(slug):
        digest = hashlib.sha256((root or folder_name).encode("utf-8")).hexdigest()
        slug = "layer_" + digest[:8]
    return slug


def _is_decoration(line: str) -> bool:
    """A badge, image or HTML line."""
    return line.startswith(("[![", "![", "<"))


def _read_readme(root: str, allowed: Collection[str] | None) -> str | None:
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return None
    by_fold: dict[str, str] = {}
    for name in names:
        by_fold.setdefault(name.casefold(), name)
    for candidate in _README_NAMES:
        real = by_fold.get(candidate)
        if real is None or (allowed is not None and real not in allowed):
            continue
        path = Path(root) / real
        try:
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as f:
                data = f.read(_README_MAX_BYTES)
        except OSError:
            continue
        return data.decode("utf-8", errors="replace")
    return None


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut <= 0:
        cut = limit - 1
    return text[:cut].rstrip() + "…"


def derive_description(root: str, allowed: Collection[str] | None = None) -> str:
    """A layer description from the folder's README, or "Documents from <folder>".

    allowed, when given, holds the relative paths that may be read: a README
    the user excluded never feeds the description.
    """
    fallback = f"Documents from {Path(root).name or root}"
    text = _read_readme(root, allowed)
    if text is None:
        return fallback
    lines = text.splitlines()

    title = None
    start = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _is_decoration(stripped):
            continue
        candidate = stripped.lstrip("#= ").strip()
        if candidate and not _RULE_LINE.match(stripped):
            title = candidate
            start = i + 1
            break
    if title is None:
        return fallback

    # Paragraphs after the title: runs of consecutive non-empty lines
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)

    body = ""
    for paragraph in paragraphs:
        first = paragraph[0]
        if _is_decoration(first) or first.startswith("#") or _RULE_LINE.match(first):
            continue
        body = " ".join(_LINK.sub(r"\1", " ".join(paragraph)).split())
        if body:
            break

    result = f"{title}: {body}" if body else title
    return _shorten(result, _DESCRIPTION_MAX)


def _sha256_file(path: Path) -> str | None:
    """SHA-256 of the whole file, as ingestion computes it; None when it can't be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _hash_entries(entries: list[FileEntry]) -> tuple[list[FileEntry], list[str]]:
    """Hash each entry; return the hashed entries and the paths that became links."""
    hashed: list[FileEntry] = []
    links: list[str] = []
    for entry in entries:
        # Never follow a file turned into a link after selection
        try:
            is_link = entry.path.is_symlink()
        except OSError:
            is_link = False
        if is_link:
            links.append(entry.relative_path)
            continue
        hashed.append(dataclasses.replace(entry, sha256=_sha256_file(entry.path)))
    return hashed, links


def max_file_size_error(size_mb: float) -> str | None:
    """Why size_mb is not a usable --max-file-size, or None when it is."""
    # float() also accepts inf and nan; neither is a size
    if math.isfinite(size_mb) and size_mb > 0:
        return None
    return "must be a finite number above 0"


def _refuse(plan: Plan, reason: str, hint: str) -> None:
    plan.refusal = (reason, hint)
    plan.action = None


def _removal_reason(path: str, plan: Plan, selection: SelectionResult, root: Path) -> str:
    verdict = plan.verdicts.get(path)
    if verdict is not None and not verdict.ingest and verdict.reason:
        return verdict.reason
    for excluded, reason in selection.excluded:
        if excluded == path or (excluded.endswith("/") and is_under(path, excluded)):
            return reason
    if selection.mode == "git" and (root / path).exists():
        return "ignored_by_gitignore"
    return "deleted"


# The steps of plan(): each returns None and refuses by setting result.refusal
# (through _refuse); plan() stops at the first refusal.


def _check_input(result: Plan, path: Path, options: Options) -> None:
    """The usage errors; a plan refused here has no selection."""
    if not path.exists():
        _refuse(result, "path_not_found", f"{path} does not exist.")
    elif not path.is_dir():
        _refuse(result, "not_a_folder", f"{path} is a file; give the folder that holds it.")
    elif options.name is not None and not NAME_PATTERN.match(options.name):
        _refuse(
            result,
            "usage_error",
            f"Invalid layer name '{options.name}'. Must be 3-100 chars, "
            "lowercase alphanumeric with underscores, starting with a letter.",
        )
    # Here as well as in the command's argument parser: Options can come from
    # any front end, and a parser's check protects only its own command line
    elif (size_error := max_file_size_error(options.max_file_size_mb)) is not None:
        _refuse(result, "usage_error", f"--max-file-size {size_error}.")


async def _select(result: Plan, max_bytes: int) -> None:
    """Select the files and judge each by policy; refuse when the folder can't be listed."""
    root = result.root
    try:
        selection = await asyncio.to_thread(select_files, Path(root))
    except NotADirectoryError:
        _refuse(result, "not_a_folder", f"{root} is not a folder.")
        return
    except PermissionError:
        _refuse(result, "path_unreadable", f"{root} can't be listed (permission denied).")
        return
    result.selection = selection
    result.verdicts = {e.relative_path: evaluate(e, max_bytes) for e in selection.entries}


async def _resolve_layer(database_url: str, result: Plan, options: Options) -> None:
    """Set the layer and the action, or refuse. The first matching row wins.

    1. A layer holds this root: sync it (root_taken when --name names another).
    2. A layer's root is above or below this one: overlapping_root.
    3. No layer has the name: create. A preset's layer: preset_layer. A layer
       without a root holding only files from this folder: adopt. Any other:
       name_taken.
    """
    root = result.root
    name = options.name or slugify(Path(root).name, root)
    result.layer = name

    same = await find_layer_by_root(database_url, root)
    if same is not None:
        if options.name is None or options.name == same["name"]:
            result.layer = same["name"]
            result.layer_id = same["id"]
            result.action = "sync"
        else:
            _refuse(
                result,
                "root_taken",
                f"The layer '{same['name']}' already holds {root}. "
                f"Run without --name, or with --name {same['name']}.",
            )
        return

    # Folded at run time: where the filesystem ignores case, so does the overlap
    overlapping = await find_overlapping_roots(database_url, root, fold=_fold_enabled())
    if overlapping:
        other = overlapping[0]
        _refuse(
            result,
            "overlapping_root",
            f"The layer '{other['name']}' holds {other['root_path']}, which overlaps {root}.",
        )
        return

    try:
        existing = await get_layer(database_url, name=name)
    except LayerNotFoundError:
        result.action = "create"
        return
    if (existing.get("origin") or "").startswith("preset:"):
        _refuse(
            result,
            "preset_layer",
            f"The layer '{name}' comes from a preset; "
            "pass --name to add the folder as another layer.",
        )
    elif existing["root_path"] is None and await layer_holds_folder(database_url, name, root):
        result.layer_id = existing["id"]
        result.action = "adopt"
    else:
        _refuse(
            result,
            "name_taken",
            f"A layer named '{name}' already exists with other content. "
            "Pass --name with another name, or delete that layer.",
        )


async def _compare_with_tracked(result: Plan, tracked: dict[str, asyncpg.Record]) -> None:
    """Hash the files policy lets in and count what the run adds, updates and removes."""
    selection = result.selection
    assert selection is not None  # _select set it, or plan() stopped
    candidates = [e for e in selection.entries if result.verdicts[e.relative_path].ingest]
    result.tracked = len(tracked)
    # Only files that passed policy are ever opened
    entries, links = await asyncio.to_thread(_hash_entries, candidates)
    # Before the removal reasons, which read the exclusions
    selection.excluded.extend((rel, "symlink") for rel in links)
    result.entries = entries

    added = updated = unchanged = 0
    work_bytes = 0
    for entry in entries:
        previous = tracked.get(entry.relative_path)
        if previous is None:
            added += 1
        elif entry.sha256 is None or entry.sha256 != previous["content_hash"]:
            updated += 1
        else:
            unchanged += 1
            continue
        work_bytes += entry.size
    ingest_set = {e.relative_path for e in entries}
    root_path = Path(result.root)
    result.removals = {
        rel: _removal_reason(rel, result, selection, root_path)
        for rel in sorted(tracked) if rel not in ingest_set
    }
    removed = len(result.removals)
    result.planned = {
        "added": added, "updated": updated, "unchanged": unchanged, "removed": removed,
    }
    result.work_files = added + updated + removed
    result.work_bytes = work_bytes
    if (
        result.work_files > LARGE_RUN_FILES
        or work_bytes > LARGE_RUN_BYTES
        or (removed >= LARGE_REMOVAL_MIN and removed * 2 > result.tracked)
    ):
        result.needs.add("large_run")


def _finish(result: Plan, options: Options) -> None:
    """Refuse a new layer without files; set the display name and description."""
    root = result.root
    if result.action == "create" and not result.entries:
        _refuse(
            result,
            "nothing_to_add",
            f"No file in {root} can be added: every file is excluded or skipped.",
        )
        return
    if result.action == "create":
        result.display_name = Path(root).name[:_DISPLAY_NAME_MAX] or slugify("", root)
        result.description = (
            options.description
            if options.description is not None
            else derive_description(root, {e.relative_path for e in result.entries})
        )
    else:
        result.description = options.description


async def plan(database_url: str, path: str | os.PathLike, options: Options) -> Plan:
    """Decide what adding the folder at path would do. Writes nothing."""
    p = Path(path).expanduser()
    result = Plan(root=(p.resolve() if p.is_dir() else p.absolute()).as_posix())

    _check_input(result, p, options)
    if result.refusal is not None:
        return result

    result.broad_why = broad_reason(result.root)
    if result.broad_why is not None:
        result.needs.add("broad_path")

    await _select(result, int(options.max_file_size_mb * 1_000_000))
    if result.refusal is not None:
        return result

    await _resolve_layer(database_url, result, options)
    if result.refusal is not None:
        return result

    tracked = (
        await tracked_files(database_url, result.layer_id, result.root)
        if result.layer_id is not None
        else {}
    )
    await _compare_with_tracked(result, tracked)
    _finish(result, options)
    return result


async def _write_layer(database_url: str, plan: Plan, owner: str) -> None:
    """Create the layer, or adopt the folder into it; write a new description."""
    layer = plan.layer
    assert layer is not None  # apply() refused a plan without one
    if plan.action == "create":
        await create_layer(
            database_url,
            name=layer,
            display_name=plan.display_name,
            description=plan.description or "",
            origin="folder",
            owner=owner,
            root_path=plan.root,
        )
        return
    if plan.action == "adopt":
        await set_root_path(database_url, layer, plan.root)
    if plan.description is not None:
        await set_description(database_url, layer, plan.description)


def _file_rows(plan: Plan, result: IngestionResult) -> list[dict]:
    """One row per path: ingestion outcomes, then untracked policy skips."""
    rows: dict[str, tuple[str, str | None, int | None]] = {}
    for outcome in result.files:
        reason = outcome.reason
        if outcome.outcome == "removed" and reason == "not_selected":
            reason = plan.removals.get(outcome.relative_path, reason)
        rows[outcome.relative_path] = (outcome.outcome, reason, outcome.count)
    for rel, verdict in plan.verdicts.items():
        if not verdict.ingest and rel not in rows:
            rows[rel] = ("skipped", verdict.reason, None)
    return [
        {"path": rel, "outcome": outcome, "reason": reason, "count": count}
        for rel, (outcome, reason, count) in sorted(rows.items())
    ]


def _error_notes(files: list[dict], result: IngestionResult) -> list[str]:
    """A note for each of the first few distinct errors of failed files."""
    notes: list[str] = []
    reason_by_path = {row["path"]: row["reason"] for row in files if row["outcome"] == "failed"}
    seen_errors: set[str] = set()
    for error in result.errors:
        path = error.get("file")
        text = error.get("error", "")
        if path not in reason_by_path or text in seen_errors:
            continue
        seen_errors.add(text)
        notes.append(f"{reason_by_path[path]}: {text}")
        if len(seen_errors) >= _MAX_ERROR_NOTES:
            break
    return notes


async def apply(
    database_url: str,
    settings: Any,
    plan: Plan,
    *,
    confirmed: set[str],
    progress: Callable[[int, int], None] | None = None,
) -> Report:
    """Carry out a plan: create or adopt the layer, ingest, audit, report.

    Raises ValueError for a refused plan or an unconfirmed need, and
    LayerValidationError for an owner longer than its column, before any write.
    LayerExistsError and RootTakenError propagate when another process won
    the name or the folder after plan(); LayerError when the layer is gone.
    """
    if (
        plan.refusal is not None
        or plan.action is None
        or plan.layer is None
        or plan.selection is None
    ):
        raise ValueError(f"The plan was refused: {plan.refusal}")
    missing = plan.needs - set(confirmed)
    if missing:
        raise ValueError(f"Not confirmed: {', '.join(sorted(missing))}")
    owner = resolve_owner(settings)
    if len(owner) > MAX_OWNER_LENGTH:
        raise LayerValidationError(f"owner must be at most {MAX_OWNER_LENGTH} characters.")
    layer = plan.layer
    selection = plan.selection

    await _write_layer(database_url, plan, owner)
    result = await ingest_documents(
        database_url,
        settings.openai_api_key,
        layer_name=layer,
        source_path=plan.root,
        files=plan.entries,
        progress=progress,
    )
    if result.status == "failed" and not result.files and result.errors:
        raise LayerError(result.errors[0].get("error", "ingestion failed"))

    files = _file_rows(plan, result)
    excluded = [{"path": rel, "reason": reason} for rel, reason in selection.excluded]
    counts = _zero_counts()
    for row in files:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    skipped_by_reason = dict(
        Counter(row["reason"] for row in files if row["outcome"] == "skipped")
    )
    excluded_by_reason = dict(Counter(reason for _, reason in selection.excluded))
    notes = list(selection.notes) + _error_notes(files, result)

    run_id = uuid4().hex
    created = plan.action == "create"
    changes = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
    await write_audit_log(
        database_url,
        user_id=owner,
        action="ingest",
        tool_name="cli:add",
        parameters={
            "source": "cli",
            "resource_type": "knowledge_layer",
            "run_id": run_id,
            "root": plan.root,
            "created": created,
            "counts": dict(counts),
        },
        result_summary=f"{layer}: {changes}" if changes else f"{layer}: no changes",
    )

    return Report(
        run_id=run_id,
        layer=layer,
        created=created,
        root=plan.root,
        mode=selection.mode,
        counts=counts,
        skipped_by_reason=skipped_by_reason,
        excluded_by_reason=excluded_by_reason,
        files=files,
        excluded=excluded,
        notes=notes,
        reason=None,
        exit_code=1 if counts["failed"] > 0 else 0,
        db_target=url_host_port(database_url),
    )
