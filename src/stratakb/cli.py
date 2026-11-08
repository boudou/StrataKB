"""The stratakb command.

    stratakb                  start the MCP server on stdio (as before)
    stratakb serve            the same
    stratakb add <path> ...   turn a folder into a layer, or sync it
    stratakb db up            start a local database in Docker, creating it on first use
    stratakb db down          stop it; its data stays
    stratakb receipt <id>     open a receipt: its quotes in context

Each subcommand is a subparser and a handler; a new one adds both and
nothing else. The command owns the terminal (prompts, progress, the report
and the exit code); add_folder makes every decision. stdout carries only
the report; prompts, progress and hints go to stderr. `add` and `receipt`
never import the server.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import structlog

from stratakb.core.terminal_text import one_line, terminal_safe_json, visible

if TYPE_CHECKING:
    from stratakb.core.local_db import DownOutcome, UpState
    from stratakb.knowledge.add_folder import Options, Plan, Report

logger = structlog.get_logger()

# Exit codes
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_DATABASE = 3
EXIT_INTERRUPTED = 130

# A group of skipped or excluded files shows this many names
_GROUP_NAMES = 5


class _UsageError(Exception):
    """A bad command line: the usage and argparse's error line are the text."""

    def __init__(self, usage: str, line: str) -> None:
        super().__init__(f"{usage}{line}")
        self.line = line


class _ArgumentParser(argparse.ArgumentParser):
    """Raises instead of exiting, so a usage error can be reported like a refusal."""

    def error(self, message: str):  # type: ignore[override]
        raise _UsageError(self.format_usage(), f"{self.prog}: error: {message}")


# ── Confirmation needs of add ────────────────────────────


def _broad_path_text(plan: Plan) -> str:
    return f"{plan.root} is a broad path ({plan.broad_why})."


def _large_run_text(plan: Plan) -> str:
    return (
        f"This run adds or updates {plan.added_or_updated} files "
        f"({plan.work_bytes / 1_000_000:.1f} MB) "
        f"and removes {plan.planned.get('removed', 0)}."
    )


class _Need(NamedTuple):
    """A confirmation need of add_folder.plan: how the command confirms it."""

    name: str  # as in Plan.needs and Options.confirmed
    flag: str  # the command-line flag that confirms it
    help: str
    text: Callable[[Plan], str]  # what needs confirming, in one sentence
    question: str  # the prompt after the text


# In the order they are asked. A row only says how to confirm a need:
# add_folder.plan() must also detect it and add its name to Plan.needs.
_NEEDS = (
    _Need(
        name="broad_path",
        flag="--allow-broad",
        help="allow a drive root, the home folder or a system folder",
        text=_broad_path_text,
        question="Add it anyway?",
    ),
    _Need(
        name="large_run",
        flag="--yes",
        help="go ahead with a large run without asking",
        text=_large_run_text,
        question="Continue?",
    ),
)


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
    for need in _NEEDS:
        add.add_argument(
            need.flag, dest="confirmed", action="append_const", const=need.name, help=need.help
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

    receipt = commands.add_parser(
        "receipt",
        help="open a receipt: its quotes in context, and whether the source changed",
        description="Open a receipt as the strata_get_receipt tool does: each "
        "quote in its surrounding document text, marked current, changed or "
        "gone. STRATA_USER and STRATA_ROLE say who you are.",
    )
    receipt.add_argument("receipt_id", metavar="id", help="the receipt id (rcpt_…)")
    receipt.add_argument(
        "--json", action="store_true", help="print the receipt as the tool returns it"
    )
    return parser


def _size_mb(text: str) -> float:
    try:
        size = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text}") from None
    from stratakb.knowledge.add_folder import max_file_size_error

    # Here as well as in add_folder.plan(): a bad value is a usage error of the
    # command line, reported before the command needs a database
    err = max_file_size_error(size)
    if err is not None:
        raise argparse.ArgumentTypeError(f"{err}: {text}")
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
        return _usage_error(e, argv)

    if args.command == "db":
        return _run_db(args)
    if args.command == "add":
        return _run_add(args)
    if args.command == "receipt":
        return _run_receipt(args)
    return _serve()


def _serve() -> int:
    # Lazy: the server module imports every tool module, which db and add do not need
    from stratakb.mcp_server import main as serve_main

    serve_main()
    return EXIT_OK


def _usage_error(error: _UsageError, argv: list[str]) -> int:
    _err(str(error).rstrip("\n"))
    if argv[:1] == ["add"] and "--json" in argv:
        from stratakb.knowledge.add_folder import refused

        report = refused("usage_error", error.line.strip(), root="")
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
        report = asyncio.run(_add(args, settings, root))
    except KeyboardInterrupt:
        _err("Interrupted. Files stored so far stay; the next run carries on from there.")
        return EXIT_INTERRUPTED
    return _finish(report, args)


def _finish(report: Report, args: argparse.Namespace) -> int:
    """Print the report and return its exit code."""
    if args.json:
        if report.reason is not None and report.hint:
            _err(one_line(report.hint))
        _out(render_json(report))
    elif report.reason is not None:
        _err(render_text(report))
    else:
        _out(render_text(report))
    return report.exit_code


async def _add(args: argparse.Namespace, settings: Any, root: str) -> Report:
    """Plan, confirm and apply one add; every failure becomes a refusal report.

    root is the folder as shown until the plan resolves it.
    """
    from stratakb.core.database import close_pool
    from stratakb.core.migrations import MigrationError, ensure_schema
    from stratakb.core.urls import url_host_port
    from stratakb.knowledge import add_folder
    from stratakb.knowledge.layers import (
        LayerExistsError,
        LayerValidationError,
        RootTakenError,
    )

    url = settings.database_url
    target = url_host_port(url)
    layer: str | None = None

    def refuse(
        reason: str, hint: str, exit_code: int = EXIT_REFUSED, mode: str | None = None
    ) -> Report:
        return add_folder.refused(
            reason,
            hint,
            root=root,
            layer=layer,
            mode=mode,
            db_target=target,
            exit_code=exit_code,
        )

    options = add_folder.Options(
        name=args.name,
        description=args.description,
        max_file_size_mb=args.max_file_size,
        confirmed=frozenset(args.confirmed or ()),
    )
    progress = _Progress() if _isatty(sys.stderr) else None
    try:
        # stderr: stdout carries only the report
        await ensure_schema(
            url,
            on_setup=lambda db: _err(f"Setting up StrataKB in the empty database at {db}."),
        )
        plan = await add_folder.plan(url, args.path, options)
        root = plan.root
        layer = plan.layer
        if plan.refusal is not None:
            reason, hint = plan.refusal
            mode = plan.selection.mode if plan.selection is not None else None
            return refuse(reason, hint, mode=mode)

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
        # LayerError from apply() (the layer was deleted meanwhile) lands here too
        logger.error("add_failed", error_type=type(e).__name__)
        return refuse("internal_error", f"{type(e).__name__}: {e}", EXIT_FAILED)
    finally:
        await close_pool()


def _confirmations(plan: Plan, options: Options) -> tuple[set[str], tuple[str, str] | None]:
    """Confirm each need: by its flag, else by a prompt at a terminal, else refuse.

    Returns the confirmed needs and, when one is not confirmed, the refusal.
    """
    confirmed: set[str] = set()
    for need in _NEEDS:
        if need.name not in plan.needs:
            continue
        if need.name in options.confirmed:
            confirmed.add(need.name)
            continue
        if not _stdin_is_tty():
            hint = f"{need.text(plan)} Pass {need.flag} to go ahead."
            return confirmed, (need.name, hint)
        if not _confirm(f"{need.text(plan)} {need.question}"):
            return confirmed, (need.name, "Not confirmed; nothing was written.")
        confirmed.add(need.name)
    return confirmed, None


def _model_notice(plan: Plan, settings: Any) -> None:
    """Say so before a first run downloads the local embedding model."""
    if settings.embedding_backend != "local" or settings.strata_local_only:
        return
    if not plan.added_or_updated:
        return
    from stratakb.knowledge.embeddings import (
        EMBEDDING_MODEL_KIND,
        is_model_cached,
        resolve_embedding_model,
    )

    model = resolve_embedding_model("local", settings.embedding_model)
    if not is_model_cached(model, EMBEDDING_MODEL_KIND):
        _err(f"Downloading the embedding model {model} (first run only)…")


def _unreachable_message(target: str, error: BaseException) -> str:
    from stratakb.config import config_files
    from stratakb.core.local_db import START_HINT

    files = ", ".join(str(path) for path in config_files()) or "none"
    return (
        f"Can't reach the database at {target} ({type(error).__name__}). "
        f"Config files read: {files}. Set DATABASE_URL or STRATAKB_CONFIG. {START_HINT}"
    )


# ── receipt ──────────────────────────────────────────────


def _run_receipt(args: argparse.Namespace) -> int:
    """Open a receipt with the tool's access check; its audit entry is cli:receipt."""
    from stratakb.config import config_file_missing, dotenv_warning, get_settings
    from stratakb.core.auth import PermissionDeniedError, check_role, get_current_user
    from stratakb.core.logging import configure_logging

    # A refusal is one line; open_receipt's receipt_refused warning is in the audit log
    configure_logging("ERROR")
    _use_utf8_output()

    if config_file_missing():
        return _receipt_failed(
            "STRATAKB_CONFIG names a file that does not exist. "
            "Fix the path or unset STRATAKB_CONFIG.",
            args,
            EXIT_REFUSED,
        )
    if hint := dotenv_warning():
        _err(hint)
    user_id, role = get_current_user()
    # The tool's own role check (require_role("viewer"))
    if not check_role(role, "viewer"):
        message = str(PermissionDeniedError(user_id, role, "viewer", "cli:receipt"))
        return _receipt_failed(message, args, EXIT_FAILED)

    url = get_settings().database_url
    try:
        receipt = asyncio.run(_open_receipt(url, args.receipt_id, user_id, role))
    except KeyboardInterrupt:
        _err("Interrupted.")
        return EXIT_INTERRUPTED
    except _ReceiptOpenError as e:
        return _receipt_failed(str(e), args, e.exit_code)

    if args.json:
        _out(terminal_safe_json({"status": "ok", "receipt": receipt}, indent=2))
    else:
        _out(render_receipt(receipt))
    return EXIT_OK


class _ReceiptOpenError(Exception):
    """A receipt that could not be opened: the one-line message and the exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


async def _open_receipt(url: str, receipt_id: str, user_id: str, role: str) -> dict:
    """open_receipt, with every failure turned into a _ReceiptOpenError."""
    from stratakb.core.database import close_pool
    from stratakb.core.migrations import MigrationError
    from stratakb.core.urls import url_host_port
    from stratakb.rag.receipts import ReceiptAccessError, open_receipt

    try:
        return await open_receipt(url, receipt_id, user_id, role, tool_name="cli:receipt")
    except ReceiptAccessError as e:
        raise _ReceiptOpenError(str(e), EXIT_FAILED) from None
    except _connection_errors() as e:
        message = _unreachable_message(url_host_port(url), e)
        raise _ReceiptOpenError(message, EXIT_DATABASE) from None
    except MigrationError as e:
        raise _ReceiptOpenError(str(e), EXIT_DATABASE) from None
    except Exception as e:
        logger.error("receipt_open_failed", error_type=type(e).__name__)
        message = f"Opening the receipt failed ({type(e).__name__})."
        raise _ReceiptOpenError(message, EXIT_FAILED) from None
    finally:
        await close_pool()


def _receipt_failed(message: str, args: argparse.Namespace, exit_code: int) -> int:
    """One line on stderr; with --json, the tool's error form on stdout too."""
    _err(visible(message))
    if args.json:
        _out(terminal_safe_json({"status": "error", "error": message}))
    return exit_code


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


def _flagged_line(row: dict) -> str:
    """One flagged file: its reason, and how much was left out when the parser counted."""
    left_out = "" if row["count"] is None else f", {row['count']:,} left out"
    return f"  {row['path']} ({row['reason']}{left_out})"


def render_text(report: Report) -> str:
    """The report for a person to read.

    Each entry of lines is one report line, escaped once at the end
    (stratakb.core.terminal_text), so no field reaches the terminal with its
    control characters and a newline in a file name starts no new line.
    """
    if report.reason is not None:
        text = f"Not added ({report.reason})"
        if report.layer:
            text += f", layer {report.layer}"
        return one_line(f"{text}: {report.hint}" if report.hint else text)

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
    flagged = [f for f in report.files if f["outcome"] == "flagged"]
    if flagged:
        lines.append("Flagged:")
        lines.extend(_flagged_line(f) for f in flagged)
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
    return "\n".join(one_line(line) for line in lines)


def _verification_line(verification: dict[str, Any]) -> str:
    """The stored result of strata_verify, as --json carries it."""
    return (
        f"Verification: {verification['result']} "
        f"(quotes checked: {verification['quotes']}, facts checked: {verification['facts']}, "
        f"flags: {verification['flags']})"
    )


def render_receipt(receipt: dict) -> str:
    """A receipt for a person to read: each quote marked >>> <<< in its context.

    The whole text is escaped once at the end (stratakb.core.terminal_text),
    so no field reaches the terminal with its control characters.
    """
    from stratakb.rag.receipts import CONTEXT_CHARS

    lines = [f"Receipt {receipt['receipt_id']} ({receipt['kind']}, {receipt['mode']} mode)"]
    layers = ", ".join(str(layer) for layer in receipt["layers"]) or "none"
    lines.append(f"Issued {receipt['issued_at']} by {receipt['issued_by']}; layers: {layers}")
    if receipt.get("question"):
        lines.append(f"Question: {receipt['question']}")
    if receipt.get("draft"):
        lines.append(f"Draft: {receipt['draft']}")
    if receipt.get("verification"):
        lines.append(_verification_line(receipt["verification"]))
    if receipt.get("verification_note"):
        lines.append(receipt["verification_note"])
    for position, quote in enumerate(receipt["quotes"], start=1):
        where = [f"layer {quote['layer']}", f"v{quote['version']}"]
        if quote.get("clause_path"):
            where.append(f"clause {quote['clause_path']}")
        where.append(f"chars {quote['char_start']}-{quote['char_end']}")
        lines.append("")
        lines.append(f"[{position}] {quote['file']} ({', '.join(where)}): {quote['status']}")
        if quote.get("note"):
            lines.append(f"    {quote['note']}")
        # An ellipsis where the shown context stops short of the text around it
        before = "…" if len(quote["context_before"]) == CONTEXT_CHARS else ""
        after = "…" if len(quote["context_after"]) == CONTEXT_CHARS else ""
        shown = (
            f"{before}{quote['context_before']}>>> {quote['quote']} <<<"
            f"{quote['context_after']}{after}"
        )
        # Split on newlines only: a carriage return or form feed is shown escaped
        lines.extend(f"    {line}".rstrip(" \t") for line in shown.removesuffix("\n").split("\n"))
    if not receipt["quotes"]:
        lines.append("No quotes.")
    return visible("\n".join(lines))


def render_json(report: Report) -> str:
    """The report as one JSON object with exactly the REPORT_KEYS."""
    return terminal_safe_json(report.as_dict(), indent=2)

