"""Tests for the stratakb command: dispatch, confirmations, exit codes and rendering,
and the db up / db down subcommands.

add_folder.plan and add_folder.apply are patched, and so are the docker calls,
so no database and no Docker are needed.
"""

from __future__ import annotations

import io
import json
import os
import typing
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
import structlog

from stratakb import cli
from stratakb.config import Settings, user_config_file
from stratakb.core import local_db
from stratakb.core.migrations import MigrationError
from stratakb.knowledge import add_folder
from stratakb.knowledge.add_folder import REPORT_KEYS, Plan, Report
from stratakb.knowledge.layers import LayerExistsError, LayerValidationError, RootTakenError

DB_URL = "postgresql://strata:secret_pw@db.example:5433/strata"
ROOT = "C:/work/docs"

# The autouse fixture replaces cli._confirm; keep the real one for its own test
REAL_CONFIRM = cli._confirm

NEED_FLAGS = [("broad_path", "--allow-broad"), ("large_run", "--yes")]


def _settings(**overrides) -> Settings:
    values = {"database_url": DB_URL, "strata_user": "test", "embedding_backend": "local"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _plan(needs: set[str] | None = None, **overrides) -> Plan:
    values = {
        "root": ROOT,
        "layer": "docs",
        "action": "create",
        "planned": {"added": 3, "updated": 0, "unchanged": 0, "removed": 0},
        "work_files": 3,
        "work_bytes": 2_500_000,
        "needs": set(needs or ()),
    }
    values.update(overrides)
    if "broad_path" in values["needs"]:
        values.setdefault("broad_why", "your home folder")
    return Plan(**values)


def _report(**overrides) -> Report:
    counts = dict.fromkeys(add_folder.OUTCOME_KEYS, 0)
    counts["added"] = 3
    values = {
        "run_id": "r" * 32,
        "layer": "docs",
        "created": True,
        "root": ROOT,
        "mode": "git",
        "counts": counts,
        "skipped_by_reason": {},
        "excluded_by_reason": {},
        "files": [
            {"path": f"f{i}.md", "outcome": "added", "reason": None} for i in range(3)
        ],
        "excluded": [],
        "notes": [],
        "reason": None,
        "exit_code": 0,
        "db_target": "db.example:5433",
    }
    values.update(overrides)
    return Report(**values)


@pytest.fixture(autouse=True)
def cli_env(monkeypatch, tmp_path):
    """No config files, no ./.env, no DATABASE_URL in the environment, no logging
    changes, no real database or model."""
    monkeypatch.delenv("STRATAKB_CONFIG", raising=False)
    # The suite runs with DATABASE_URL set; db up reports on it
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    settings = _settings()
    # add configures logging (to stderr); put back the suite's configuration after
    saved_logging = structlog.get_config()
    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.core.migrations.ensure_schema", new_callable=AsyncMock) as schema,
        patch("stratakb.knowledge.embeddings.is_model_cached", return_value=True),
        patch.object(cli, "_stdin_is_tty", return_value=False),
        patch.object(cli, "_confirm", side_effect=AssertionError("prompted")) as confirm,
    ):
        yield {"settings": settings, "schema": schema, "confirm": confirm}
    structlog.configure(**saved_logging)


def _patch_service(plan: Plan, report: Report | None = None, apply_error=None):
    """Patch plan() and apply(); returns the context manager pair."""
    apply_mock = AsyncMock(return_value=report or _report(), side_effect=apply_error)
    return (
        patch.object(add_folder, "plan", AsyncMock(return_value=plan)),
        patch.object(add_folder, "apply", apply_mock),
    )


def _run(argv, plan: Plan | None = None, report=None, apply_error=None):
    plan_patch, apply_patch = _patch_service(plan or _plan(), report, apply_error)
    with plan_patch as plan_mock, apply_patch as apply_mock:
        code = cli.main(argv)
    return code, plan_mock, apply_mock


# ── Dispatch ─────────────────────────────────────────────


def test_no_subcommand_starts_the_server():
    serve = MagicMock()
    with patch("stratakb.mcp_server.main", serve):
        assert cli.main([]) == 0
    serve.assert_called_once_with()


def test_serve_starts_the_server():
    serve = MagicMock()
    with patch("stratakb.mcp_server.main", serve):
        assert cli.main(["serve"]) == 0
    serve.assert_called_once_with()


def test_usage_error_exits_2(capsys):
    assert cli.main(["add", ROOT, "--max-file-size", "lots"]) == 2
    assert "--max-file-size" in capsys.readouterr().err


@pytest.mark.parametrize("size", ["inf", "-inf", "nan", "0", "-2"])
def test_max_file_size_must_be_finite_and_positive(capsys, size):
    plan_mock = AsyncMock()
    with patch.object(add_folder, "plan", plan_mock):
        assert cli.main(["add", ROOT, "--max-file-size", size]) == 2
        assert "--max-file-size" in capsys.readouterr().err

        assert cli.main(["add", ROOT, "--max-file-size", size, "--json"]) == 2
        data = json.loads(capsys.readouterr().out)
    assert data["reason"] == "usage_error"
    assert data["exit_code"] == 2
    plan_mock.assert_not_called()


def test_usage_error_with_json_prints_a_refusal(capsys):
    assert cli.main(["add", "--json"]) == 2
    data = json.loads(capsys.readouterr().out)
    assert data["reason"] == "usage_error"
    assert data["exit_code"] == 2


def test_add_passes_the_flags_to_the_service():
    code, plan_mock, _ = _run(
        ["add", ROOT, "--name", "my_docs", "--description", "D", "--max-file-size", "2.5"]
    )
    assert code == 0
    _, path, options = plan_mock.call_args.args
    assert path == ROOT
    assert options.name == "my_docs"
    assert options.description == "D"
    assert options.max_file_size_mb == 2.5
    assert options.allow_broad is False and options.yes is False


# ── Exit codes ───────────────────────────────────────────


def test_exit_0_prints_the_report(capsys):
    code, _, apply_mock = _run(["add", ROOT])
    assert code == 0
    out = capsys.readouterr().out
    assert "Created layer docs" in out
    assert "3 added" in out
    assert apply_mock.call_args.kwargs["confirmed"] == set()


def test_exit_1_when_a_file_failed():
    report = _report(exit_code=1)
    code, _, _ = _run(["add", ROOT], report=report)
    assert code == 1


def test_exit_2_on_a_plan_refusal(capsys):
    plan = _plan(action=None, refusal=("name_taken", "A layer named 'docs' already exists."))
    code, _, apply_mock = _run(["add", ROOT, "--json"], plan=plan)
    assert code == 2
    apply_mock.assert_not_called()
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["reason"] == "name_taken"
    assert data["layer"] == "docs"
    assert data["created"] is False
    assert "already exists" in captured.err


def test_exit_3_when_the_database_is_unreachable(cli_env, capsys):
    cli_env["schema"].side_effect = OSError("connection refused")
    code, plan_mock, _ = _run(["add", ROOT])
    assert code == 3
    plan_mock.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "Can't reach the database at db.example:5433 (OSError). "
        "Config files read: none. Set DATABASE_URL or STRATAKB_CONFIG."
    ) in captured.err
    assert captured.err.rstrip().endswith("Run stratakb db up to start a local database.")
    assert "secret_pw" not in captured.err


def test_unreachable_json_hint_names_db_up(cli_env, capsys):
    cli_env["schema"].side_effect = OSError("connection refused")
    code, _, _ = _run(["add", ROOT, "--json"])
    assert code == 3
    captured = capsys.readouterr()
    # The JSON report carries no hint; --json prints it on stderr
    assert json.loads(captured.out)["reason"] == "db_unreachable"
    assert captured.err.rstrip().endswith(local_db.START_HINT)


@pytest.mark.parametrize(
    "error",
    [
        asyncpg.InvalidPasswordError("bad password"),
        asyncpg.InvalidAuthorizationSpecificationError("no such role"),
        asyncpg.InvalidCatalogNameError("no such database"),
        asyncpg.PostgresConnectionError("lost"),
        asyncpg.InterfaceError("closed"),
    ],
)
def test_connection_errors_exit_3(cli_env, capsys, error):
    cli_env["schema"].side_effect = error
    code, _, _ = _run(["add", ROOT, "--json"])
    assert code == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "db_unreachable"


def test_migration_error_exits_3(cli_env, capsys):
    cli_env["schema"].side_effect = MigrationError("Apply 05_add_folder.sql as the owner.")
    code, _, _ = _run(["add", ROOT, "--json"])
    assert code == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "schema"


@pytest.mark.parametrize(
    ("error", "reason", "code"),
    [
        (LayerExistsError("taken"), "name_taken", 2),
        (RootTakenError("held"), "root_taken", 2),
        (LayerValidationError("owner too long"), "usage_error", 2),
        (RuntimeError("boom"), "internal_error", 1),
    ],
)
def test_apply_errors_map_to_reasons(capsys, error, reason, code):
    exit_code, _, _ = _run(["add", ROOT, "--json"], apply_error=error)
    assert exit_code == code
    data = json.loads(capsys.readouterr().out)
    assert data["reason"] == reason
    assert data["exit_code"] == code


def test_exit_130_on_ctrl_c(capsys):
    code, _, _ = _run(["add", ROOT], apply_error=KeyboardInterrupt())
    assert code == 130
    assert "Interrupted" in capsys.readouterr().err


def test_config_not_found_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STRATAKB_CONFIG", str(tmp_path / "missing.env"))
    code, plan_mock, _ = _run(["add", ROOT, "--json"])
    assert code == 2
    plan_mock.assert_not_called()
    assert json.loads(capsys.readouterr().out)["reason"] == "config_not_found"


# ── Confirmations ────────────────────────────────────────


@pytest.mark.parametrize(("need", "flag"), NEED_FLAGS)
def test_need_without_a_tty_is_refused_and_names_the_flag(capsys, need, flag):
    code, _, apply_mock = _run(["add", ROOT], plan=_plan({need}))
    assert code == 2
    apply_mock.assert_not_called()
    err = capsys.readouterr().err
    assert f"Not added ({need})" in err
    assert flag in err


@pytest.mark.parametrize(("need", "flag"), NEED_FLAGS)
def test_flag_confirms_without_a_prompt(cli_env, need, flag):
    code, _, apply_mock = _run(["add", ROOT, flag], plan=_plan({need}))
    assert code == 0
    assert apply_mock.call_args.kwargs["confirmed"] == {need}
    cli_env["confirm"].assert_not_called()


def test_flag_skips_the_prompt_at_a_tty(cli_env):
    with patch.object(cli, "_stdin_is_tty", return_value=True):
        code, _, apply_mock = _run(
            ["add", ROOT, "--allow-broad", "--yes"], plan=_plan({"broad_path", "large_run"})
        )
    assert code == 0
    assert apply_mock.call_args.kwargs["confirmed"] == {"broad_path", "large_run"}
    cli_env["confirm"].assert_not_called()


def test_no_at_the_prompt_refuses(capsys):
    with (
        patch.object(cli, "_stdin_is_tty", return_value=True),
        patch.object(cli, "_confirm", return_value=False) as confirm,
    ):
        code, _, apply_mock = _run(["add", ROOT], plan=_plan({"large_run"}))
    assert code == 2
    apply_mock.assert_not_called()
    question = confirm.call_args.args[0]
    assert question == "This run adds or updates 3 files (2.5 MB) and removes 0. Continue?"
    assert "Not added (large_run)" in capsys.readouterr().err


def test_yes_at_the_prompt_asks_broad_path_first():
    with (
        patch.object(cli, "_stdin_is_tty", return_value=True),
        patch.object(cli, "_confirm", return_value=True) as confirm,
    ):
        code, _, apply_mock = _run(["add", ROOT], plan=_plan({"broad_path", "large_run"}))
    assert code == 0
    questions = [c.args[0] for c in confirm.call_args_list]
    assert questions[0] == f"{ROOT} is a broad path (your home folder). Add it anyway?"
    assert questions[1].endswith("Continue?")
    assert apply_mock.call_args.kwargs["confirmed"] == {"broad_path", "large_run"}


def test_yes_does_not_confirm_broad_path(capsys):
    code, _, apply_mock = _run(["add", ROOT, "--yes"], plan=_plan({"broad_path", "large_run"}))
    assert code == 2
    apply_mock.assert_not_called()
    assert "Not added (broad_path)" in capsys.readouterr().err


def test_confirm_answers(monkeypatch, capsys):
    for answer, expected in [("y\n", True), ("YES\n", True), ("n\n", False), ("\n", False)]:
        monkeypatch.setattr("sys.stdin", io.StringIO(answer))
        assert REAL_CONFIRM("Go?") is expected
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Go? [y/N]" in captured.err


# ── Model notice and offline flags ───────────────────────


def test_model_notice_when_not_cached(capsys):
    with patch("stratakb.knowledge.embeddings.is_model_cached", return_value=False):
        code, _, _ = _run(["add", ROOT])
    assert code == 0
    assert "Downloading the embedding model all-mpnet-base-v2" in capsys.readouterr().err


def test_no_model_notice_without_work(capsys):
    plan = _plan(planned={"added": 0, "updated": 0, "unchanged": 4, "removed": 1})
    with patch("stratakb.knowledge.embeddings.is_model_cached", return_value=False):
        _run(["add", ROOT], plan=plan)
    assert "Downloading" not in capsys.readouterr().err


def test_local_only_keeps_hugging_face_offline(cli_env, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    settings = _settings(strata_local_only=True)
    with patch("stratakb.config.get_settings", return_value=settings):
        code, _, _ = _run(["add", ROOT])
    assert code == 0
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


REMOTE_EMBEDDINGS = {"embedding_backend": "openai", "embedding_base_url": "http://gpu-box:11434/v1"}


@pytest.fixture
def hf_flags(monkeypatch):
    """Put back the offline flags that a local-only add sets."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")


def test_local_only_refuses_a_remote_embedding_url(cli_env, hf_flags, capsys):
    settings = _settings(strata_local_only=True, **REMOTE_EMBEDDINGS)
    with patch("stratakb.config.get_settings", return_value=settings):
        code, plan_mock, apply_mock = _run(["add", ROOT])
    assert code == 2
    plan_mock.assert_not_called()
    apply_mock.assert_not_called()
    cli_env["schema"].assert_not_awaited()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "EMBEDDING_BASE_URL" in captured.err
    assert "STRATA_LOCAL_ONLY" in captured.err
    assert "gpu-box" in captured.err


def test_local_only_refusal_json(hf_flags, capsys):
    settings = _settings(strata_local_only=True, **REMOTE_EMBEDDINGS)
    with patch("stratakb.config.get_settings", return_value=settings):
        code, _, apply_mock = _run(["add", ROOT, "--json"])
    assert code == 2
    apply_mock.assert_not_called()
    data = json.loads(capsys.readouterr().out)
    assert tuple(data) == REPORT_KEYS
    assert data["reason"] == "local_only"
    assert data["exit_code"] == 2
    assert data["layer"] is None
    assert not any(data["counts"].values())


def test_local_only_refuses_the_openai_default(hf_flags, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    settings = _settings(strata_local_only=True, embedding_backend="openai")
    with patch("stratakb.config.get_settings", return_value=settings):
        code, _, apply_mock = _run(["add", ROOT])
    assert code == 2
    apply_mock.assert_not_called()
    assert "EMBEDDING_BACKEND" in capsys.readouterr().err


@pytest.mark.parametrize(
    "rewrite",
    [
        {"anthropic_api_key": "sk-ant-test"},
        {
            "query_rewrite_provider": "openai_compatible",
            "query_rewrite_base_url": "http://gpu-box:11434/v1",
            "query_rewrite_model": "llama3.2",
        },
    ],
    ids=["anthropic_key", "remote_rewrite"],
)
def test_local_only_ignores_query_rewriting(hf_flags, rewrite):
    settings = _settings(strata_local_only=True, embedding_backend="local", **rewrite)
    with patch("stratakb.config.get_settings", return_value=settings):
        code, _, apply_mock = _run(["add", ROOT])
    assert code == 0
    apply_mock.assert_awaited_once()


def test_local_only_allows_a_loopback_embedding_url(hf_flags):
    settings = _settings(
        strata_local_only=True,
        embedding_backend="openai",
        embedding_base_url="http://127.0.0.1:11434/v1",
    )
    with patch("stratakb.config.get_settings", return_value=settings):
        code, _, apply_mock = _run(["add", ROOT])
    assert code == 0
    apply_mock.assert_awaited_once()


def test_remote_embedding_url_runs_when_local_only_is_off():
    settings = _settings(**REMOTE_EMBEDDINGS)
    with patch("stratakb.config.get_settings", return_value=settings):
        code, _, apply_mock = _run(["add", ROOT])
    assert code == 0
    apply_mock.assert_awaited_once()


def test_add_leaves_offline_flags_when_local_only_is_off(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    code, _, _ = _run(["add", ROOT])
    assert code == 0
    assert os.environ["HF_HUB_OFFLINE"] == "0"


# ── Rendering ────────────────────────────────────────────


def test_json_keys_match_the_interface(capsys):
    code, _, _ = _run(["add", ROOT, "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert tuple(data) == REPORT_KEYS
    assert data["counts"]["added"] == 3


def test_json_keeps_non_ascii_paths():
    report = _report(files=[{"path": "Ünïcode/ταυτότητα.md", "outcome": "added", "reason": None}])
    text = cli.render_json(report)
    assert "ταυτότητα.md" in text
    assert tuple(json.loads(text)) == REPORT_KEYS


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
    report = _report(
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
    text = cli.render_text(_report(created=False, counts=counts, files=[]))
    assert text.startswith("Synced layer docs in the database at db.example:5433")
    assert "  4 unchanged" in text


# ── db up / db down ──────────────────────────────────────

CONTAINER_URL = "postgresql://strata:strata_dev_password@127.0.0.1:5433/strata"
NO_SUCH = (1, "", "Error: No such container: stratakb-db")


def _fake_docker(status=NO_SUCH, port="127.0.0.1:5433", **answers):
    """An AsyncMock for local_db._docker that answers by the first argument.

    status is the answer to `container inspect`: a state string or a tuple.
    """
    defaults = {
        "info": (0, "27.0.1", ""),
        "container": status if isinstance(status, tuple) else (0, status, ""),
        "run": (0, "abc123", ""),
        "start": (0, "stratakb-db", ""),
        "unpause": (0, "stratakb-db", ""),
        "stop": (0, "stratakb-db", ""),
        "port": (0, port, ""),
        "rm": (0, "stratakb-db", ""),
    }
    defaults.update(answers)

    async def answer(*args, timeout=None):
        return defaults[args[0]]

    return AsyncMock(side_effect=answer)


def _db(argv, docker=None, wait_ready=None):
    docker = docker or _fake_docker()
    wait_ready = wait_ready or AsyncMock()
    with (
        patch.object(local_db, "_docker", docker),
        patch.object(local_db, "_wait_ready", wait_ready),
    ):
        code = cli.main(argv)
    return code, docker, wait_ready


def _verbs(docker) -> list[str]:
    return [c.args[0] for c in docker.call_args_list]


def test_db_up_without_docker_exits_3(capsys):
    with patch.object(local_db.shutil, "which", return_value=None):
        code = cli.main(["db", "up"])
    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert "docker is not on PATH" in lines[0]
    assert "https://docs.docker.com/get-docker/" in lines[0]


def test_db_up_docker_not_runnable_exits_3(capsys):
    with (
        patch.object(local_db.shutil, "which", return_value="C:/bin/docker.exe"),
        patch.object(
            local_db.asyncio,
            "create_subprocess_exec",
            AsyncMock(side_effect=PermissionError("blocked")),
        ),
    ):
        code = cli.main(["db", "up"])
    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "could not be run (PermissionError)" in captured.err


def test_db_up_daemon_down_exits_3(capsys):
    docker = _fake_docker(info=(1, "", "Cannot connect to the Docker daemon"))
    code, docker, wait_ready = _db(["db", "up"], docker)
    assert code == 3
    assert _verbs(docker) == ["info"]
    wait_ready.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "daemon is not running" in captured.err


def test_db_up_creates_the_container(cli_env, capsys):
    code, docker, wait_ready = _db(["db", "up"])
    assert code == 0
    run = next(c.args for c in docker.call_args_list if c.args[0] == "run")
    assert list(run) == [
        "run", "-d",
        "--name", "stratakb-db",
        "--restart", "unless-stopped",
        "-e", "POSTGRES_DB=strata",
        "-e", "POSTGRES_USER=strata",
        "-e", "POSTGRES_PASSWORD=strata_dev_password",
        "-p", "127.0.0.1:5433:5432",
        "-v", "stratakb-data:/var/lib/postgresql/data",
        "pgvector/pgvector:pg16",
    ]  # fmt: skip
    wait_ready.assert_awaited_once_with(CONTAINER_URL)
    cli_env["schema"].assert_awaited_once_with(CONTAINER_URL)
    captured = capsys.readouterr()
    assert "Database ready at 127.0.0.1:5433 (container stratakb-db, created)." in captured.out
    assert f"DATABASE_URL={CONTAINER_URL}" in captured.out
    assert f"Put this line in {user_config_file()} to use this database." in captured.out
    # Progress goes to stderr; the container id docker run prints is never echoed
    assert "Creating the container stratakb-db" in captured.err
    assert "abc123" not in captured.out + captured.err


def test_db_up_nothing_to_configure(capsys):
    default_url = Settings(_env_file=None).database_url
    with patch("stratakb.config.get_settings", return_value=_settings(database_url=default_url)):
        code, _, _ = _db(["db", "up"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Nothing to configure: DATABASE_URL already points here." in out
    assert "strata_dev_password" not in out


def test_db_up_environment_sets_database_url(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    code, _, _ = _db(["db", "up"])
    assert code == 0
    out = capsys.readouterr().out
    assert f"DATABASE_URL={CONTAINER_URL}" in out
    assert "Your environment sets DATABASE_URL to another database" in out


def test_db_up_already_running(cli_env, capsys):
    code, docker, wait_ready = _db(["db", "up"], _fake_docker("running"))
    assert code == 0
    assert not {"run", "start", "unpause"} & set(_verbs(docker))
    wait_ready.assert_awaited_once_with(CONTAINER_URL)
    cli_env["schema"].assert_awaited_once_with(CONTAINER_URL)
    assert "(container stratakb-db, already running)." in capsys.readouterr().out


def test_db_up_starts_a_stopped_container(capsys):
    code, docker, _ = _db(["db", "up"], _fake_docker("exited"))
    assert code == 0
    verbs = _verbs(docker)
    assert verbs.count("start") == 1
    assert "run" not in verbs
    start = next(c.args for c in docker.call_args_list if c.args[0] == "start")
    assert start == ("start", "stratakb-db")
    assert "(container stratakb-db, started)." in capsys.readouterr().out


def test_db_up_unpauses(capsys):
    code, docker, _ = _db(["db", "up"], _fake_docker("paused"))
    assert code == 0
    verbs = _verbs(docker)
    assert verbs.count("unpause") == 1
    assert "start" not in verbs
    assert "run" not in verbs


@pytest.mark.parametrize(
    "message",
    [
        "docker: Error response from daemon: ... port is already allocated.",
        "Bind for 127.0.0.1:5433 failed: address already in use",
        # Docker Desktop on Windows, when a host process holds the port
        "docker: Error response from daemon: Ports are not available: exposing port "
        "TCP 127.0.0.1:5433 -> 0.0.0.0:0: listen tcp 127.0.0.1:5433: bind: Only one usage "
        "of each socket address (protocol/network address/port) is normally permitted.",
    ],
)
def test_db_up_port_in_use(capsys, message):
    docker = _fake_docker(run=(125, "", message))
    code, docker, wait_ready = _db(["db", "up"], docker)
    assert code == 3
    wait_ready.assert_not_called()
    rm = [c.args for c in docker.call_args_list if c.args[0] == "rm"]
    assert rm == [("rm", "stratakb-db")]
    for call in docker.call_args_list:
        assert "volume" not in call.args
        assert not (call.args[0] == "rm" and "-v" in call.args)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Port 5433 on 127.0.0.1 is already in use" in captured.err
    assert "--port" in captured.err


def test_db_up_start_port_in_use(capsys):
    docker = _fake_docker(
        "exited", start=(1, "", "Bind for 127.0.0.1:5433 failed: address already in use")
    )
    code, docker, _ = _db(["db", "up"], docker)
    assert code == 3
    assert "rm" not in _verbs(docker)
    err = capsys.readouterr().err
    assert "Port 5433 on 127.0.0.1 is already in use" in err
    assert "docker rm stratakb-db" in err


def test_db_up_reports_the_existing_port(capsys):
    docker = _fake_docker("running", port="127.0.0.1:5433")
    code, _, wait_ready = _db(["db", "up", "--port", "5440"], docker)
    assert code == 0
    wait_ready.assert_awaited_once_with(CONTAINER_URL)
    captured = capsys.readouterr()
    assert "Database ready at 127.0.0.1:5433" in captured.out
    assert "already uses port 5433" in captured.err


def test_db_up_readiness_timeout(capsys):
    with (
        patch.object(local_db, "_docker", _fake_docker("running")),
        patch.object(local_db, "_READY_TIMEOUT_S", 0.2),
        patch.object(local_db.asyncpg, "connect", AsyncMock(side_effect=OSError("reset"))),
    ):
        code = cli.main(["db", "up"])
    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "did not accept connections within 0.2 s (OSError)" in captured.err


def test_db_up_schema_error_exits_3(cli_env, capsys):
    cli_env["schema"].side_effect = MigrationError("Apply 05_add_folder.sql as the owner.")
    code, _, _ = _db(["db", "up"], _fake_docker("running"))
    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Apply 05_add_folder.sql as the owner." in captured.err


def test_db_up_unexpected_error_exits_3(cli_env, capsys):
    cli_env["schema"].side_effect = asyncpg.PostgresError("relation x does not exist\nmore")
    code, _, _ = _db(["db", "up"], _fake_docker("running"))
    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    # Progress lines come first; the error is the last line, and only its first line
    last = captured.err.strip().splitlines()[-1]
    assert last == "stratakb db up failed (PostgresError): relation x does not exist"


def test_db_down_unexpected_error_exits_3(capsys):
    with patch.object(local_db, "down", AsyncMock(side_effect=RuntimeError("boom"))):
        code = cli.main(["db", "down"])
    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert captured.err.strip().splitlines() == ["stratakb db down failed (RuntimeError): boom"]


def test_up_text_covers_every_state():
    assert set(cli._UP_TEXT) == set(typing.get_args(local_db.UpState))


@pytest.mark.parametrize(
    ("status", "stops", "line"),
    [
        ("running", True, "Stopped stratakb-db. Its data stays in the Docker volume stratakb-data"),
        ("exited", False, "stratakb-db is not running."),
        (NO_SUCH, False, "There is no stratakb-db container; nothing to stop."),
    ],
)
def test_db_down(capsys, status, stops, line):
    code, docker, _ = _db(["db", "down"], _fake_docker(status))
    assert code == 0
    verbs = _verbs(docker)
    assert ("stop" in verbs) is stops
    assert "rm" not in verbs
    assert all("volume" not in c.args for c in docker.call_args_list)
    assert line in capsys.readouterr().out


def test_db_down_without_docker_exits_3(capsys):
    with patch.object(local_db.shutil, "which", return_value=None):
        code = cli.main(["db", "down"])
    assert code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "docker is not on PATH" in captured.err


def test_db_without_subcommand_exits_2(capsys):
    assert cli.main(["db"]) == 2
    assert "usage" in capsys.readouterr().err


def test_db_up_bad_port_exits_2(capsys):
    assert cli.main(["db", "up", "--port", "0"]) == 2
    assert "not a port number" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://strata:strata_dev_password@localhost:5433/strata", True),
        ("postgresql://strata:strata_dev_password@127.0.0.1:5433/strata", True),
        ("postgresql://strata:strata_dev_password@[::1]:5433/strata", True),
        ("postgresql://strata:strata_dev_password@localhost:5434/strata", False),
        ("postgresql://strata:other@localhost:5433/strata", False),
        ("postgresql://someone:strata_dev_password@localhost:5433/strata", False),
        ("postgresql://strata:strata_dev_password@localhost:5433/other", False),
        ("postgresql://strata:strata_dev_password@db.example:5433/strata", False),
        ("postgresql://strata:strata_dev_password@localhost:99999/strata", False),
    ],
)
def test_points_here(url, expected):
    assert local_db.points_here(url, 5433) is expected
