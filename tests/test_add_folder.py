"""Tests for the add-folder service: names, broad paths, descriptions, plan and apply.

The pure-function tests need no database; the plan/apply tests do, with
embeddings mocked.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from stratakb.knowledge import add_folder
from stratakb.knowledge.add_folder import (
    REPORT_KEYS,
    Options,
    broad_reason,
    db_target,
    derive_description,
    is_broad,
    resolve_owner,
    slugify,
)
from tests.helpers import FAKE_EMBEDDING


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch, tmp_path):
    """Keep the machine's git config and any repository above tmp_path out of the tests."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))


@pytest.fixture
def not_broad():
    """tmp_path can sit under a system folder (%LOCALAPPDATA%): treat it as normal."""
    with patch("stratakb.knowledge.add_folder.broad_reason", return_value=None):
        yield


def _write(path: Path, text: str = "# Doc\n\nSome text about the topic.\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _settings(user: str | None = None, *, key: str = "fake-key") -> SimpleNamespace:
    fields = {"strata_user"} if user is not None else set()
    return SimpleNamespace(
        strata_user=user or "developer", model_fields_set=fields, openai_api_key=key
    )


# --- slugify ---------------------------------------------------------------


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("api-specs", "api_specs"),
        ("2024", "d_2024"),
        ("a", "a00"),
        ("My Docs!", "my_docs"),
        ("Ünïcode", "n_code"),
    ],
)
def test_slugify(folder, expected):
    assert slugify(folder) == expected


def test_slugify_cuts_long_names_to_100():
    name = slugify("a" * 99 + "-" + "b" * 50)
    assert len(name) <= 100
    assert not name.endswith("_")
    assert slugify("x" * 150) == "x" * 100


@pytest.mark.parametrize("folder", ["---", ""])
def test_slugify_fallback(folder):
    name = slugify(folder, "C:/some/root")
    assert name.startswith("layer_") and len(name) == len("layer_") + 8
    # Stable for one root, different for another
    assert slugify(folder, "C:/some/root") == name
    assert slugify(folder, "C:/other") != name


# --- broad paths -------------------------------------------------------------


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    home = tmp_path / "home" / "me"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home.resolve().as_posix()


def test_broad_home_and_its_ancestors(fake_home):
    assert broad_reason(fake_home) == "your home folder"
    parent = fake_home.rsplit("/", 1)[0]
    assert broad_reason(parent) == "a folder above your home folder"
    assert is_broad(parent)


def test_broad_drive_roots(fake_home):
    assert broad_reason("C:/") == "a drive root"
    assert broad_reason("/") == "a drive root"


def test_broad_program_files(monkeypatch, fake_home):
    monkeypatch.setattr("sys.platform", "win32")
    for var in ("SystemRoot", "ProgramFiles(x86)", "ProgramData", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ProgramFiles", "D:\\Program Files")
    assert broad_reason("D:/Program Files") == "inside D:/Program Files"
    assert broad_reason("D:/program files/App/docs") == "inside D:/Program Files"
    assert broad_reason("D:/Program Files Extra") is None


def test_broad_posix_system_folder(monkeypatch, fake_home):
    monkeypatch.setattr("sys.platform", "linux")
    assert broad_reason("/usr/share") == "inside /usr"
    assert broad_reason("/usrdocs") is None


def test_broad_normal_folder(monkeypatch, fake_home):
    for var in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData",
                "APPDATA", "LOCALAPPDATA"):
        monkeypatch.delenv(var, raising=False)
    assert broad_reason(f"{fake_home}/repos/app/docs") is None
    assert not is_broad(f"{fake_home}/repos/app/docs")


# --- derive_description ------------------------------------------------------


def test_description_skips_badges(tmp_path):
    _write(
        tmp_path / "README.md",
        "# My Project\n\n[![CI](https://x/badge.svg)](https://x)\n\n"
        "A tool for [teams](https://example.com) that\nship docs.\n",
    )
    assert derive_description(str(tmp_path)) == "My Project: A tool for teams that ship docs."


def test_description_without_readme(tmp_path):
    folder = tmp_path / "handbook"
    folder.mkdir()
    assert derive_description(str(folder)) == "Documents from handbook"


def test_description_cut_at_300(tmp_path):
    _write(tmp_path / "readme.md", "# Title\n\n" + "word " * 80 + "\n")
    description = derive_description(str(tmp_path))
    assert len(description) <= 300
    assert description.endswith("…")
    assert description.startswith("Title: word word")
    assert not description[:-1].endswith(" ")


def test_description_html_block_then_title(tmp_path):
    _write(
        tmp_path / "README.md",
        '<div align="center">\n\n# StrataKB\n\n**A knowledge base.**\nFor teams.\n\n</div>\n',
    )
    assert derive_description(str(tmp_path)) == "StrataKB: **A knowledge base.** For teams."


def test_description_rst_title(tmp_path):
    _write(tmp_path / "README.rst", "=======\nProject\n=======\n\nThe body text.\n")
    assert derive_description(str(tmp_path)) == "Project: The body text."


def test_description_not_in_allowed(tmp_path):
    folder = tmp_path / "notes"
    _write(folder / "README.md", "# Secret plans\n\nDo not share.\n")
    assert derive_description(str(folder), allowed={"other.md"}) == "Documents from notes"
    assert derive_description(str(folder), allowed={"README.md"}) == (
        "Secret plans: Do not share."
    )


# --- owner and target --------------------------------------------------------


def test_owner_from_strata_user():
    assert resolve_owner(_settings("alice")) == "alice"


def test_owner_from_settings_object(monkeypatch):
    from stratakb.config import Settings

    monkeypatch.setenv("STRATA_USER", "bob")
    assert resolve_owner(Settings(_env_file=None)) == "bob"
    monkeypatch.delenv("STRATA_USER")
    with patch("stratakb.knowledge.add_folder.getpass.getuser", return_value="carol"):
        assert resolve_owner(Settings(_env_file=None)) == "os:carol"


def test_owner_from_login():
    with patch("stratakb.knowledge.add_folder.getpass.getuser", return_value="dave"):
        assert resolve_owner(_settings()) == "os:dave"
    with patch("stratakb.knowledge.add_folder.getpass.getuser", side_effect=KeyError("x")):
        assert resolve_owner(_settings()) == "os:unknown"


def test_db_target_hides_credentials():
    target = db_target("postgresql://strata:s3cret_pw@db.example.com:6543/strata")
    assert target == "db.example.com:6543"
    assert "s3cret_pw" not in target and "strata" not in target
    assert db_target("postgresql:///strata") == "localhost:5432"


def test_refused_report_shape():
    report = add_folder.refused("name_taken", "pick another", root="C:/x", layer="x")
    data = report.as_dict()
    assert tuple(data) == REPORT_KEYS
    assert data["counts"] == dict.fromkeys(add_folder.OUTCOME_KEYS, 0)
    assert data["files"] == [] and data["excluded"] == []
    assert data["created"] is False and data["reason"] == "name_taken"
    assert data["exit_code"] == 2
    json.dumps(data)


# --- plan and apply (database) -----------------------------------------------


@pytest.fixture
def db_url(not_broad):
    from stratakb.config import get_settings
    return get_settings().database_url


@pytest.fixture
async def cleanup(db_url):
    """Layer names to delete after the test."""
    from stratakb.knowledge.layers import LayerNotFoundError, delete_layer

    names: list[str] = []
    yield names
    for name in names:
        with contextlib.suppress(LayerNotFoundError):
            await delete_layer(db_url, name=name)


def _name() -> str:
    return f"test_add_{uuid4().hex[:10]}"


def _docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    _write(root / "README.md", "# Team Docs\n\nHow the team works.\n")
    _write(root / "guide" / "setup.md")
    _write(root / "notes.md", "# Notes\n\nMeeting notes.\n")
    _write(root / "tool.py", "print('code')\n")
    _write(root / "client_secret.json", '{"key": "x"}\n')
    return root


def _fake_embed(texts, **_):
    return [FAKE_EMBEDDING for _ in texts]


async def _run(db_url, root, options, *, settings=None, confirmed=frozenset()):
    """plan() then apply() with mocked embeddings."""
    p = await add_folder.plan(db_url, root, options)
    assert p.refusal is None, p.refusal
    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = _fake_embed
        report = await add_folder.apply(
            db_url, settings or _settings("tester"), p, confirmed=set(confirmed)
        )
    return p, report


def _outcomes(report) -> dict[str, tuple[str, str | None]]:
    return {f["path"]: (f["outcome"], f["reason"]) for f in report.files}


async def _layer(db_url, name):
    from stratakb.knowledge.layers import get_layer_status
    return await get_layer_status(db_url, name=name)


async def _audit_rows(db_url, root):
    from stratakb.core.database import get_pool

    pool = await get_pool(db_url)
    return await pool.fetch(
        "SELECT * FROM strata_core.audit_log WHERE tool_name = 'cli:add' "
        "AND parameters->>'root' = $1",
        root,
    )


async def test_first_add(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)

    p, report = await _run(db_url, root, Options(name=name))

    assert p.action == "create" and p.layer == name
    layer = await _layer(db_url, name)
    assert layer["origin"] == "folder"
    assert layer["owner"] == "tester"
    assert layer["root_path"] == root.resolve().as_posix()
    assert layer["display_name"] == "docs"
    assert layer["description"] == "Team Docs: How the team works."
    assert report.created is True
    assert report.exit_code == 0 and report.reason is None
    assert report.counts == {
        "added": 3, "updated": 0, "unchanged": 0, "removed": 0,
        "skipped": 2, "flagged": 0, "failed": 0,
    }
    assert report.skipped_by_reason == {"code": 1, "secret": 1}
    outcomes = _outcomes(report)
    assert outcomes["guide/setup.md"] == ("added", None)
    assert outcomes["tool.py"] == ("skipped", "code")
    assert outcomes["client_secret.json"] == ("skipped", "secret")
    assert tuple(report.as_dict()) == REPORT_KEYS
    json.dumps(report.as_dict())


async def test_unreadable_file_is_read_error(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    original_read = Path.read_bytes

    def locked_read(self):
        if self.name == "notes.md":
            raise PermissionError("locked")
        return original_read(self)

    with patch.object(Path, "read_bytes", locked_read):
        _, report = await _run(db_url, root, Options(name=name))

    assert _outcomes(report)["notes.md"] == ("failed", "read_error")
    assert report.counts["failed"] == 1
    assert {"path": "notes.md", "outcome": "failed", "reason": "read_error"} in (
        report.as_dict()["files"]
    )


async def test_default_name_is_the_folder_slug(db_url, tmp_path, cleanup):
    folder = f"Add-Test {uuid4().hex[:8]}"
    root = tmp_path / folder
    _write(root / "a.md")
    expected = slugify(folder)
    cleanup.append(expected)

    p, report = await _run(db_url, root, Options())

    assert p.layer == expected and report.layer == expected
    assert (await _layer(db_url, expected))["display_name"] == folder


async def test_quiet_rerun(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    await _run(db_url, root, Options(name=name))

    from stratakb.knowledge import embeddings

    model_before = embeddings._local_model
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.action == "sync"
    assert p.planned == {"added": 0, "updated": 0, "unchanged": 3, "removed": 0}
    assert p.work_files == 0 and p.work_bytes == 0
    assert "large_run" not in p.needs
    with (
        patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as embed,
        patch.object(embeddings, "_get_local_model", side_effect=AssertionError("loaded")),
    ):
        report = await add_folder.apply(db_url, _settings("tester"), p, confirmed=set())
    embed.assert_not_called()
    assert embeddings._local_model is model_before
    assert report.counts["unchanged"] == 3
    assert report.counts["added"] == report.counts["updated"] == report.counts["removed"] == 0
    assert report.created is False and report.exit_code == 0


async def test_edit_and_delete(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    await _run(db_url, root, Options(name=name))

    _write(root / "notes.md", "# Notes\n\nChanged meeting notes.\n")
    (root / "guide" / "setup.md").unlink()
    p, report = await _run(db_url, root, Options(name=name))

    assert p.planned == {"added": 0, "updated": 1, "unchanged": 1, "removed": 1}
    assert p.removals == {"guide/setup.md": "deleted"}
    outcomes = _outcomes(report)
    assert outcomes["notes.md"] == ("updated", None)
    assert outcomes["guide/setup.md"] == ("removed", "deleted")
    assert report.counts["updated"] == 1 and report.counts["removed"] == 1


async def test_removed_with_skip_or_exclusion_reason(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    _write(root / "big.md", "# Big\n\n" + "x" * 3000 + "\n")
    name = _name()
    cleanup.append(name)
    await _run(db_url, root, Options(name=name))

    _write(root / ".strataignore", "notes.md\n")
    p, report = await _run(db_url, root, Options(name=name, max_file_size_mb=0.002))

    assert p.removals == {"big.md": "too_large", "notes.md": "ignored_by_strataignore"}
    outcomes = _outcomes(report)
    assert outcomes["big.md"] == ("removed", "too_large")
    assert outcomes["notes.md"] == ("removed", "ignored_by_strataignore")
    assert all(f["reason"] != "not_selected" for f in report.files)


async def test_sync_sets_description_only_when_given(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    await _run(db_url, root, Options(name=name))
    _write(root / "README.md", "# Renamed\n\nNew text.\n")

    await _run(db_url, root, Options(name=name))
    assert (await _layer(db_url, name))["description"] == "Team Docs: How the team works."

    await _run(db_url, root, Options(name=name, description="Hand-written"))
    assert (await _layer(db_url, name))["description"] == "Hand-written"


async def _snapshot(db_url, root, name):
    """The layer row (None when missing) and the audit rows for root."""
    from stratakb.knowledge.layers import LayerNotFoundError

    layer = None
    if name is not None:
        with contextlib.suppress(LayerNotFoundError):
            layer = await _layer(db_url, name)
    return layer, await _audit_rows(db_url, root)


async def _assert_nothing_written(db_url, p, name, *, before=(None, [])):
    """apply() refuses the plan, and the layer row and audit rows stay as before."""
    with pytest.raises(ValueError):
        await add_folder.apply(db_url, _settings("tester"), p, confirmed={"broad_path"})
    assert await _snapshot(db_url, p.root, name) == before


async def test_name_taken(db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import create_layer
    from tests.helpers import ingest

    name = _name()
    cleanup.append(name)
    await create_layer(db_url, name=name, display_name="Manual")
    other = tmp_path / "other"
    _write(other / "x.md")
    await ingest(db_url,name, other)

    root = _docs(tmp_path)
    before = await _snapshot(db_url, root.resolve().as_posix(), name)
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.refusal is not None and p.refusal[0] == "name_taken"
    assert p.layer == name and p.action is None
    with pytest.raises(ValueError):
        await add_folder.apply(db_url, _settings("tester"), p, confirmed=set())
    await _assert_nothing_written(db_url, p, name, before=before)
    assert (await _layer(db_url, name))["root_path"] is None


async def test_root_taken(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    await _run(db_url, root, Options(name=name))
    other = _name()
    before = await _snapshot(db_url, root.resolve().as_posix(), other)
    assert before[0] is None and len(before[1]) == 1

    p = await add_folder.plan(db_url, root, Options(name=other))
    assert p.refusal is not None and p.refusal[0] == "root_taken"
    assert name in p.refusal[1]
    await _assert_nothing_written(db_url, p, other, before=before)


async def test_overlapping_root(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    await _run(db_url, root, Options(name=name))
    _write(tmp_path / "top.md")

    for path in (tmp_path, root / "guide"):
        p = await add_folder.plan(db_url, path, Options(name=_name()))
        assert p.refusal is not None and p.refusal[0] == "overlapping_root", path
        await _assert_nothing_written(db_url, p, None)


async def test_overlapping_root_case_only(db_url, tmp_path, cleanup, monkeypatch):
    from stratakb.knowledge.layers import create_layer

    root = _docs(tmp_path)
    resolved = root.resolve().as_posix()
    stored = resolved[:-4] + "DOCS"
    name = _name()
    cleanup.append(name)
    await create_layer(db_url, name=name, display_name="Upper", root_path=stored)
    monkeypatch.setattr("sys.platform", "win32")

    other = _name()
    p = await add_folder.plan(db_url, root, Options(name=other))
    assert p.refusal is not None and p.refusal[0] == "overlapping_root"
    assert stored in p.refusal[1]
    await _assert_nothing_written(db_url, p, other)


async def test_preset_layer(db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import create_layer

    name = _name()
    cleanup.append(name)
    await create_layer(db_url, name=name, display_name="Preset", origin="preset:x")
    root = _docs(tmp_path)
    before = await _snapshot(db_url, root.resolve().as_posix(), name)
    assert before[0] is not None and before[1] == []

    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.refusal is not None and p.refusal[0] == "preset_layer"
    await _assert_nothing_written(db_url, p, name, before=before)


async def test_nothing_to_add(db_url, tmp_path):
    root = tmp_path / "code_only"
    _write(root / "a.py", "x = 1\n")
    _write(root / "b.py", "y = 2\n")
    name = _name()
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.refusal is not None and p.refusal[0] == "nothing_to_add"
    await _assert_nothing_written(db_url, p, name)


async def test_usage_and_path_refusals(db_url, tmp_path):
    root = _docs(tmp_path)
    p = await add_folder.plan(db_url, root, Options(name="Bad-Name"))
    assert p.refusal is not None and p.refusal[0] == "usage_error"
    assert p.layer is None
    await _assert_nothing_written(db_url, p, None)

    p = await add_folder.plan(db_url, tmp_path / "missing", Options())
    assert p.refusal is not None and p.refusal[0] == "path_not_found"
    await _assert_nothing_written(db_url, p, None)

    p = await add_folder.plan(db_url, root / "notes.md", Options())
    assert p.refusal is not None and p.refusal[0] == "not_a_folder"
    await _assert_nothing_written(db_url, p, None)


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), float("-inf")])
async def test_max_file_size_must_be_finite_and_positive(db_url, tmp_path, size):
    name = _name()
    p = await add_folder.plan(db_url, _docs(tmp_path), Options(name=name, max_file_size_mb=size))
    assert p.refusal is not None and p.refusal[0] == "usage_error"
    assert "--max-file-size" in p.refusal[1]
    await _assert_nothing_written(db_url, p, name)


async def test_adopt_manual_layer(db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import create_layer
    from tests.helpers import ingest

    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    await create_layer(db_url, name=name, display_name="Manual", description="Kept")
    (root / "client_secret.json").unlink()
    await ingest(db_url,name, root.resolve())

    p, report = await _run(db_url, root, Options(name=name))

    assert p.action == "adopt"
    assert report.created is False
    layer = await _layer(db_url, name)
    assert layer["root_path"] == root.resolve().as_posix()
    assert layer["description"] == "Kept"
    assert layer["origin"] is None
    # tool.py was never ingested (unsupported by the plain ingest), the docs are unchanged
    assert report.counts["unchanged"] == 3 and report.counts["added"] == 0


async def test_large_run_from_file_count(db_url, tmp_path, cleanup, monkeypatch):
    root = _docs(tmp_path)
    monkeypatch.setattr(add_folder, "LARGE_RUN_FILES", 2)
    p = await add_folder.plan(db_url, root, Options(name=_name()))
    assert p.needs == {"large_run"}
    with pytest.raises(ValueError):
        await add_folder.apply(db_url, _settings("tester"), p, confirmed=set())

    name = _name()
    cleanup.append(name)
    monkeypatch.setattr(add_folder, "LARGE_RUN_FILES", 2000)
    await _run(db_url, root, Options(name=name), confirmed={"large_run"})


async def test_large_run_from_removals(db_url, tmp_path, cleanup, monkeypatch):
    root = tmp_path / "many"
    for i in range(5):
        _write(root / f"doc{i}.md", f"# Doc {i}\n\nText {i}.\n")
    name = _name()
    cleanup.append(name)
    await _run(db_url, root, Options(name=name))
    monkeypatch.setattr(add_folder, "LARGE_REMOVAL_MIN", 3)

    for i in range(2):
        (root / f"doc{i}.md").unlink()
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.planned["removed"] == 2 and "large_run" not in p.needs

    (root / "doc2.md").unlink()
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.planned["removed"] == 3 and p.tracked == 5
    assert "large_run" in p.needs


async def test_secret_never_hashed(db_url, tmp_path):
    root = _docs(tmp_path)
    with patch(
        "stratakb.knowledge.add_folder._sha256_file", wraps=add_folder._sha256_file
    ) as spy:
        p = await add_folder.plan(db_url, root, Options(name=_name()))
    hashed = {Path(call.args[0]).name for call in spy.call_args_list}
    assert "client_secret.json" not in hashed
    assert "tool.py" not in hashed
    assert hashed == {"README.md", "setup.md", "notes.md"}
    assert all(e.sha256 for e in p.entries)


async def test_owner_from_login_on_create(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    with patch("stratakb.knowledge.add_folder.getpass.getuser", return_value="erin"):
        await _run(db_url, root, Options(name=name), settings=_settings())
    assert (await _layer(db_url, name))["owner"] == "os:erin"


async def test_long_owner_on_sync_refused_before_writing(db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import LayerValidationError

    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    await _run(db_url, root, Options(name=name))
    before = len(await _audit_rows(db_url, root.resolve().as_posix()))

    p = await add_folder.plan(db_url, root, Options(name=name, description="New"))
    with pytest.raises(LayerValidationError):
        await add_folder.apply(db_url, _settings("u" * 201), p, confirmed=set())
    assert (await _layer(db_url, name))["description"] != "New"
    assert len(await _audit_rows(db_url, p.root)) == before


async def test_audit_entry(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    p, report = await _run(db_url, root, Options(name=name), settings=_settings("auditor"))

    rows = await _audit_rows(db_url, p.root)
    assert len(rows) == 1
    row = rows[0]
    assert row["user_id"] == "auditor"
    assert row["action"] == "ingest"
    assert row["tool_name"] == "cli:add"
    params = row["parameters"]
    if isinstance(params, str):
        params = json.loads(params)
    assert set(params) == {"source", "resource_type", "run_id", "root", "created", "counts"}
    assert params["source"] == "cli"
    assert params["resource_type"] == "knowledge_layer"
    assert params["run_id"] == report.run_id
    assert params["root"] == p.root
    assert params["created"] is True
    assert set(params["counts"]) == set(add_folder.OUTCOME_KEYS)
    assert params["counts"]["added"] == 3
    text = json.dumps(params) + (row["result_summary"] or "")
    for file_name in ("README", "setup", "notes", "tool.py", "client_secret"):
        assert file_name not in text


async def test_concurrent_create_raises(db_url, tmp_path, cleanup):
    from stratakb.knowledge.layers import LayerExistsError, create_layer

    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    p = await add_folder.plan(db_url, root, Options(name=name))
    assert p.action == "create"
    await create_layer(db_url, name=name, display_name="Raced")
    with pytest.raises(LayerExistsError):
        await add_folder.apply(db_url, _settings("tester"), p, confirmed=set())


async def test_failed_file_note_once(db_url, tmp_path, cleanup):
    root = _docs(tmp_path)
    name = _name()
    cleanup.append(name)
    p = await add_folder.plan(db_url, root, Options(name=name))
    with patch("stratakb.knowledge.ingestion.embed_texts", new_callable=AsyncMock) as mock:
        mock.side_effect = RuntimeError("no embedder configured")
        report = await add_folder.apply(db_url, _settings("tester"), p, confirmed=set())
    assert report.exit_code == 1
    assert report.counts["failed"] == 3
    assert [n for n in report.notes if "no embedder" in n] == [
        "embed_error: no embedder configured"
    ]
