"""The stratakb command.

    stratakb                  start the MCP server on stdio (as before)
    stratakb serve            the same
    stratakb add <path> ...   turn a folder into a layer, or sync it
    stratakb db up            start a local database in Docker, creating it on first use
    stratakb db down          stop it; its data stays

Each subcommand is a subparser and a handler; a new one adds both and
nothing else. The command owns the terminal (prompts, progress, the report
and the exit code); add_folder makes every decision. stdout carries only
the report; prompts, progress and hints go to stderr. `add` never imports
the server.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from stratakb.core.local_db import DownOutcome, UpState
    from stratakb.knowledge.add_folder import Plan, Report

logger = structlog.get_logger()

# Exit codes
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_DATABASE = 3
EXIT_INTERRUPTED = 130

# A group of skipped or excluded files shows this many names
_GROUP_NAMES = 5

_NEED_FLAGS = {"broad_path": "--allow-broad", "large_run": "--yes"}


class _UsageError(Exception):
    """A bad command line; argparse's message is the text."""


class _ArgumentParser(argparse.ArgumentParser):
    """Raises instead of exiting, so a usage error can be reported like a refusal."""

    def error(self, message: str):  # type: ignore[override]
        raise _UsageError(f"{self.format_usage()}{self.prog}: error: {message}")


def _parser() -> _ArgumentParser:
    from stratakb.config import LOCAL_DB_PORT

    parser = _ArgumentParser(
        prog="stratakb",
        description="StrataKB: a layered knowledge base for MCP clients. "
        "With no command, starts the MCP server on stdio.",
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    commands.add_parser("serve", help="start the MCP server on stdio (the default)")

    add = commands.add_parser(
        "add",
        help="turn a folder into a layer, or sync the layer that holds it",
        description="Turn a folder into a knowledge layer, or sync the layer "
        "that already holds it. Running it again adds new files, re-embeds "
        "changed ones and removes deleted ones.",
    )
    add.add_argument("path", help="the folder to add")
    add.add_argument("--name", help="layer name (default: from the folder name)")
    add.add_argument("--description", help="layer description (default: from the README)")
    add.add_argument(
        "--max-file-size",
        type=_size_mb,
        default=50.0,
        metavar="MB",
        help="skip files larger than this many megabytes (default: 50)",
    )
    add.add_argument(
        "--allow-broad",
        action="store_true",
        help="allow a drive root, the home folder or a system folder",
    )
    add.add_argument(
        "--yes", action="store_true", help="go ahead with a large run without asking"
    )
    add.add_argument("--json", action="store_true", help="print the report as JSON")

    db = commands.add_parser(
        "db",
        help="start or stop a local database in Docker",
        description="Start or stop a local PostgreSQL database in a Docker "
        "container, bound to 127.0.0.1. Its data stays in a Docker volume.",
    )
    db_commands = db.add_subparsers(dest="db_command", metavar="<command>", required=True)
    up = db_commands.add_parser("up", help="start the local database, creating it on first use")
    up.add_argument(
        "--port",
        type=_port,
        default=LOCAL_DB_PORT,
        help="host port on 127.0.0.1 when the container is created (default: %(default)s)",
    )
    db_commands.add_parser("down", help="stop the local database; its data stays")
    return parser


def _size_mb(text: str) -> float:
    try:
        size = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text}") from None
    # float() also accepts inf and nan; neither is a size
    if not (math.isfinite(size) and size > 0):
        raise argparse.ArgumentTypeError(f"not a finite number above 0: {text}")
    return size


def _port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text}") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"not a port number (1-65535): {text}")
    return port


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stratakb command and return its exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(argv)
    except _UsageError as e:
        return _usage_error(str(e), argv)

    if args.command == "db":
        return _run_db(args)
    if args.command == "add":
        return _run_add(args)
    return _serve()


def _serve() -> int:
    # Lazy: the server module imports every tool module, which db and add do not need
    from stratakb.mcp_server import main as serve_main

    serve_main()
    return EXIT_OK


def _usage_error(message: str, argv: list[str]) -> int:
    _err(message.rstrip("\n"))
    if argv[:1] == ["add"] and "--json" in argv:
        from stratakb.knowledge.add_folder import refused

        report = refused("usage_error", message.strip().splitlines()[-1], root="")
        _out(render_json(report))
    return EXIT_REFUSED


# ── Terminal helpers ─────────────────────────────────────


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _err(text: str) -> None:
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def _isatty(stream: Any) -> bool:
    try:
        return stream is not None and stream.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _stdin_is_tty() -> bool:
    if not _isatty(sys.stdin):
        return False
    if sys.platform != "win32":
        return True
    # On Windows the NUL device passes isatty(); only a console has a console mode
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except Exception:
        return True


def _confirm(question: str) -> bool:
    """Ask on stderr and read one line from stdin; only y or yes confirms."""
    sys.stderr.write(f"{question} [y/N] ")
    sys.stderr.flush()
    answer = sys.stdin.readline()
    if not answer.endswith("\n"):
        # End of input: finish the prompt line
        sys.stderr.write("\n")
    return answer.strip().lower() in ("y", "yes")


def _use_utf8_output() -> None:
    """A piped stream on Windows uses the ANSI code page; paths may not fit it."""
    for stream in (sys.stdout, sys.stderr):
        if _isatty(stream):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")


class _Progress:
    """One updating stderr line: embedding 37/112."""

    def __init__(self) -> None:
        self.shown = False

    def __call__(self, done: int, total: int) -> None:
        sys.stderr.write(f"\rembedding {done}/{total}")
        sys.stderr.flush()
        self.shown = True

    def finish(self) -> None:
        if self.shown:
            sys.stderr.write("\n")
            sys.stderr.flush()


# ── db ───────────────────────────────────────────────────

_UP_TEXT: dict[UpState, str] = {
    "created": "created",
    "started": "started",
    "running": "already running",
}


def _connection_errors() -> tuple[type[BaseException], ...]:
    """The errors that mean the database can't be reached."""
    import asyncpg

    return (
        OSError,
        asyncpg.PostgresConnectionError,
        asyncpg.InvalidAuthorizationSpecificationError,
        asyncpg.InvalidCatalogNameError,
        asyncpg.InterfaceError,
    )


def _run_db(args: argparse.Namespace) -> int:
    from stratakb.config import config_files, get_settings, user_config_file
    from stratakb.core import local_db
    from stratakb.core.logging import configure_logging
    from stratakb.core.migrations import MigrationError

    configure_logging("WARNING")
    _use_utf8_output()

    try:
        if args.db_command == "down":
            _out(_down_text(asyncio.run(local_db.down())))
            return EXIT_OK
        result = asyncio.run(local_db.up(args.port, notify=_err))
    except KeyboardInterrupt:
        _err("Interrupted.")
        return EXIT_INTERRUPTED
    except (local_db.LocalDbError, MigrationError) as e:
        _err(str(e))
        return EXIT_DATABASE
    except _connection_errors() as e:
        _err(f"Can't reach the database in {local_db.CONTAINER} ({type(e).__name__}).")
        return EXIT_DATABASE
    except Exception as e:
        _err(
            f"stratakb db {args.db_command} failed ({type(e).__name__}): "
            f"{local_db.first_line(str(e))}"
        )
        return EXIT_DATABASE

    state = _UP_TEXT[result.state]
    _out(
        f"Database ready at 127.0.0.1:{result.port} "
        f"(container {local_db.CONTAINER}, {state})."
    )
    if local_db.points_here(get_settings().database_url, result.port):
        _out("Nothing to configure: DATABASE_URL already points here.")
        return EXIT_OK
    _out(f"DATABASE_URL={result.url}")
    if any(key.upper() == "DATABASE_URL" for key in os.environ):
        _out(
            "Your environment sets DATABASE_URL to another database; "
            "set it to the line above to use this one."
        )
    else:
        files = config_files()
        target = files[-1] if files else user_config_file()
        _out(f"Put this line in {target} to use this database.")
    return EXIT_OK


def _down_text(outcome: DownOutcome) -> str:
    from stratakb.core.local_db import CONTAINER, VOLUME

    if outcome == "stopped":
        return (
            f"Stopped {CONTAINER}. Its data stays in the Docker volume {VOLUME}; "
            "stratakb db up starts it again."
        )
    if outcome == "not_running":
        return f"{CONTAINER} is not running."
    return f"There is no {CONTAINER} container; nothing to stop."


# ── add ──────────────────────────────────────────────────


def _run_add(args: argparse.Namespace) -> int:
    from stratakb.config import config_file_missing, dotenv_warning, get_settings
    from stratakb.core.logging import configure_logging

    configure_logging("WARNING")
    _use_utf8_output()

    root = Path(args.path).expanduser().absolute().as_posix()
    if config_file_missing():
        from stratakb.knowledge.add_folder import refused

        return _finish(
            refused(
                "config_not_found",
                "STRATAKB_CONFIG names a file that does not exist. "
                "Fix the path or unset STRATAKB_CONFIG.",
                root=root,
            ),
            args,
        )
    if hint := dotenv_warning():
        _err(hint)
    settings = get_settings()
    if settings.strata_local_only:
        # Before add_folder imports huggingface_hub, which reads these once
        from stratakb.core.egress import (
            embedding_provider,
            keep_hugging_face_offline,
            refusal_line,
        )

        keep_hugging_face_offline()
        # Only the embedding side: add never rewrites queries, so an Anthropic
        # key or a remote rewrite endpoint does not stop it
        if provider := embedding_provider(settings):
            from stratakb.knowledge.add_folder import refused

            return _finish(refused("local_only", refusal_line(provider), root=root), args)

    try:
        report = asyncio.run(_add(args, settings))
    except KeyboardInterrupt:
        _err("Interrupted. Files stored so far stay; the next run carries on from there.")
        return EXIT_INTERRUPTED
    return _finish(report, args)


def _finish(report: Report, args: argparse.Namespace) -> int:
    """Print the report and return its exit code."""
    if args.json:
        if report.reason is not None and report.hint:
            _err(report.hint)
        _out(render_json(report))
    elif report.reason is not None:
        _err(render_text(report))
    else:
        _out(render_text(report))
    return report.exit_code


async def _add(args: argparse.Namespace, settings: Any) -> Report:
    """Plan, confirm and apply one add; every failure becomes a refusal report."""
    from stratakb.core.database import close_pool
    from stratakb.core.migrations import MigrationError, ensure_schema
    from stratakb.knowledge import add_folder
    from stratakb.knowledge.layers import (
        LayerExistsError,
        LayerValidationError,
        RootTakenError,
    )

    url = settings.database_url
    target = add_folder.db_target(url)
    root = Path(args.path).expanduser().absolute().as_posix()
    layer: str | None = None

    def refuse(reason: str, hint: str, exit_code: int = EXIT_REFUSED) -> Report:
        return add_folder.refused(
            reason, hint, root=root, layer=layer, db_target=target, exit_code=exit_code
        )

    options = add_folder.Options(
        name=args.name,
        description=args.description,
        max_file_size_mb=args.max_file_size,
        allow_broad=args.allow_broad,
        yes=args.yes,
    )
    progress = _Progress() if _isatty(sys.stderr) else None
    try:
        await ensure_schema(url)
        plan = await add_folder.plan(url, args.path, options)
        root = plan.root
        layer = plan.layer
        if plan.refusal is not None:
            reason, hint = plan.refusal
            report = refuse(reason, hint)
            report.mode = plan.selection.mode if plan.selection is not None else None
            return report

        confirmed, refusal = _confirmations(plan, options)
        if refusal is not None:
            return refuse(*refusal)

        _model_notice(plan, settings)
        try:
            return await add_folder.apply(
                url, settings, plan, confirmed=confirmed, progress=progress
            )
        finally:
            if progress is not None:
                progress.finish()
    except _connection_errors() as e:
        return refuse("db_unreachable", _unreachable_message(target, e), EXIT_DATABASE)
    except MigrationError as e:
        return refuse("schema", str(e), EXIT_DATABASE)
    except LayerValidationError as e:
        return refuse("usage_error", str(e))
    except LayerExistsError as e:
        return refuse("name_taken", str(e))
    except RootTakenError as e:
        return refuse("root_taken", str(e))
    except Exception as e:
        # LayerNotFoundError (the layer was deleted meanwhile) lands here too
        logger.error("add_failed", error_type=type(e).__name__)
        return refuse("internal_error", f"{type(e).__name__}: {e}", EXIT_FAILED)
    finally:
        await close_pool()


def _confirmations(plan: Plan, options: Any) -> tuple[set[str], tuple[str, str] | None]:
    """Confirm each need: by its flag, else by a prompt at a terminal, else refuse.

    Returns the confirmed needs and, when one is not confirmed, the refusal.
    """
    flags = {"broad_path": options.allow_broad, "large_run": options.yes}
    confirmed: set[str] = set()
    for need in ("broad_path", "large_run"):
        if need not in plan.needs:
            continue
        if flags[need]:
            confirmed.add(need)
            continue
        if not _stdin_is_tty():
            hint = f"{_need_text(plan, need)} Pass {_NEED_FLAGS[need]} to go ahead."
            return confirmed, (need, hint)
        if not _confirm(_question(plan, need)):
            return confirmed, (need, "Not confirmed; nothing was written.")
        confirmed.add(need)
    return confirmed, None


def _need_text(plan: Plan, need: str) -> str:
    if need == "broad_path":
        return f"{plan.root} is a broad path ({plan.broad_why})."
    files = plan.planned.get("added", 0) + plan.planned.get("updated", 0)
    return (
        f"This run adds or updates {files} files ({plan.work_bytes / 1_000_000:.1f} MB) "
        f"and removes {plan.planned.get('removed', 0)}."
    )


def _question(plan: Plan, need: str) -> str:
    if need == "broad_path":
        return f"{_need_text(plan, need)} Add it anyway?"
    return f"{_need_text(plan, need)} Continue?"


def _model_notice(plan: Plan, settings: Any) -> None:
    """Say so before a first run downloads the local embedding model."""
    if settings.embedding_backend != "local" or settings.strata_local_only:
        return
    if not plan.planned.get("added", 0) + plan.planned.get("updated", 0):
        return
    from stratakb.knowledge.embeddings import is_model_cached, resolve_embedding_model

    model = resolve_embedding_model("local", settings.embedding_model)
    if not is_model_cached(model):
        _err(f"Downloading the embedding model {model} (first run only)…")


def _unreachable_message(target: str, error: BaseException) -> str:
    from stratakb.config import config_files
    from stratakb.core.local_db import START_HINT

    files = ", ".join(str(path) for path in config_files()) or "none"
    return (
        f"Can't reach the database at {target} ({type(error).__name__}). "
        f"Config files read: {files}. Set DATABASE_URL or STRATAKB_CONFIG. {START_HINT}"
    )


# ── Rendering ────────────────────────────────────────────


def _group_lines(pairs: list[tuple[str, str | None]]) -> list[str]:
    """One line per reason: up to five names, then "and N more"."""
    groups: dict[str, list[str]] = {}
    for path, reason in pairs:
        groups.setdefault(reason or "unknown", []).append(path)
    lines = []
    for reason, paths in groups.items():
        shown = ", ".join(paths[:_GROUP_NAMES])
        more = len(paths) - _GROUP_NAMES
        if more > 0:
            shown += f" and {more} more"
        lines.append(f"  {reason} ({len(paths)}): {shown}")
    return lines


def render_text(report: Report) -> str:
    """The report for a person to read."""
    if report.reason is not None:
        text = f"Not added ({report.reason})"
        if report.layer:
            text += f", layer {report.layer}"
        return f"{text}: {report.hint}" if report.hint else text

    verb = "Created" if report.created else "Synced"
    lines = [f"{verb} layer {report.layer} in the database at {report.db_target}"]
    lines.append(f"Folder: {report.root}" + (f" ({report.mode} mode)" if report.mode else ""))
    counts = [f"  {n} {outcome}" for outcome, n in report.counts.items() if n]
    lines.extend(counts or ["  no files"])

    skipped = [(f["path"], f["reason"]) for f in report.files if f["outcome"] == "skipped"]
    if skipped:
        lines.append("Skipped:")
        lines.extend(_group_lines(skipped))
    excluded = [(e["path"], e["reason"]) for e in report.excluded]
    if excluded:
        lines.append("Excluded:")
        lines.extend(_group_lines(excluded))
    failed = [f for f in report.files if f["outcome"] == "failed"]
    if failed:
        lines.append("Failed:")
        lines.extend(f"  {f['path']} ({f['reason']})" for f in failed)
    if report.notes:
        lines.append("Notes:")
        lines.extend(f"  {note}" for note in report.notes)
    lines.append(
        f"Next: ask in your MCP client: what do the docs in {report.layer} say about …?"
    )
    return "\n".join(lines)


def render_json(report: Report) -> str:
    """The report as one JSON object with exactly the REPORT_KEYS."""
    return json.dumps(report.as_dict(), ensure_ascii=False, indent=2)

