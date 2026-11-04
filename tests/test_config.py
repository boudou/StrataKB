"""Tests for where settings come from.

Tests cover:
- The source order: environment over STRATAKB_CONFIG over the user-level file
- ./.env in the current folder is not read, and dotenv_warning() says so
- The user-level path on Windows and elsewhere
- Compose-only keys in a config file are ignored
- A STRATAKB_CONFIG that names a missing file is skipped without raising
- A file saved with a BOM keeps its first key
- STRATA_USER counts as set only when a source sets it

No database is needed: every file lives in tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from stratakb.config import (
    DOTENV_WARNING,
    config_file_missing,
    config_files,
    dotenv_warning,
    get_settings,
    user_config_file,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Point the user-level folder and the current folder into tmp_path."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("APPDATA", str(tmp_path / "user"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    for name in ("STRATAKB_CONFIG", "STRATA_USER", "STRATA_ROLE", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _user_file(text: str) -> Path:
    return _write(user_config_file(), text)


def test_no_config_files_by_default():
    assert config_files() == []
    assert get_settings().strata_role == "viewer"


def test_user_file_is_under_the_user_folder(tmp_path):
    assert user_config_file() == tmp_path / "user" / "stratakb" / "config.env"


def test_source_order(tmp_path, monkeypatch):
    user = _user_file("STRATA_ROLE=viewer\nLOG_LEVEL=DEBUG\nSTRATA_USER=u1\n")
    named = _write(tmp_path / "named.env", "STRATA_ROLE=operator\nSTRATA_USER=u2\n")
    monkeypatch.setenv("STRATAKB_CONFIG", str(named))
    monkeypatch.setenv("STRATA_USER", "u3")

    assert config_files() == [user, named]
    settings = get_settings()
    assert settings.log_level == "DEBUG"
    assert settings.strata_role == "operator"
    assert settings.strata_user == "u3"


def test_role_priority(tmp_path, monkeypatch):
    _user_file("STRATA_ROLE=viewer\n")
    assert get_settings().strata_role == "viewer"

    named = _write(tmp_path / "named.env", "STRATA_ROLE=operator\n")
    monkeypatch.setenv("STRATAKB_CONFIG", str(named))
    assert get_settings().strata_role == "operator"

    monkeypatch.setenv("STRATA_ROLE", "admin")
    assert get_settings().strata_role == "admin"


def test_dotenv_in_cwd_not_read():
    _user_file("STRATA_ROLE=viewer\n")
    _write(Path.cwd() / ".env", "STRATA_ROLE=admin\n")

    assert get_settings().strata_role == "viewer"
    assert dotenv_warning() == "./.env is not read; set STRATAKB_CONFIG=.env to use it."
    assert dotenv_warning() == DOTENV_WARNING


@pytest.mark.parametrize("value", [".env", "./.env", "absolute"])
def test_dotenv_read_through_stratakb_config(monkeypatch, value):
    dotenv = _write(Path.cwd() / ".env", "STRATA_ROLE=admin\n")
    monkeypatch.setenv("STRATAKB_CONFIG", str(dotenv) if value == "absolute" else value)

    assert get_settings().strata_role == "admin"
    assert dotenv_warning() is None


def test_no_warning_without_dotenv():
    assert dotenv_warning() is None


def test_user_config_file_windows(tmp_path, monkeypatch):
    with patch("sys.platform", "win32"):
        monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
        assert user_config_file() == tmp_path / "roaming" / "stratakb" / "config.env"

        monkeypatch.delenv("APPDATA")
        with patch.object(Path, "home", return_value=tmp_path):
            expected = tmp_path / "AppData" / "Roaming" / "stratakb" / "config.env"
            assert user_config_file() == expected


def test_user_config_file_posix(tmp_path, monkeypatch):
    with patch("sys.platform", "linux"):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert user_config_file() == tmp_path / "xdg" / "stratakb" / "config.env"

        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        with patch.object(Path, "home", return_value=tmp_path):
            assert user_config_file() == tmp_path / ".config" / "stratakb" / "config.env"


def test_compose_keys_ignored():
    _user_file(
        "DB_PASSWORD=secret\nDB_PORT=5433\nREADONLY_DB_PASSWORD=secret\n"
        "STRATAKB_CONFIG=elsewhere.env\nSTRATA_ROLE=operator\n"
    )
    settings = get_settings()
    assert settings.strata_role == "operator"
    assert not hasattr(settings, "db_password")


def test_missing_stratakb_config(tmp_path, monkeypatch):
    user = _user_file("STRATA_ROLE=operator\n")
    monkeypatch.setenv("STRATAKB_CONFIG", str(tmp_path / "missing.env"))

    assert config_file_missing() is True
    assert config_files() == [user]
    assert get_settings().strata_role == "operator"


def test_stratakb_config_folder_counts_as_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATAKB_CONFIG", str(tmp_path))
    assert config_file_missing() is True
    assert config_files() == []


def test_empty_stratakb_config_is_unset(monkeypatch):
    monkeypatch.setenv("STRATAKB_CONFIG", "")
    assert config_file_missing() is False
    assert config_files() == []


def test_same_file_twice_is_listed_once(monkeypatch):
    user = _user_file("STRATA_ROLE=operator\n")
    monkeypatch.setenv("STRATAKB_CONFIG", str(user))
    assert config_files() == [user]


def test_bom_file_read():
    path = user_config_file()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xef\xbb\xbfSTRATA_ROLE=admin\n")
    assert get_settings().strata_role == "admin"


def test_strata_user_in_fields_set(tmp_path, monkeypatch):
    assert "strata_user" not in get_settings().model_fields_set

    named = _write(tmp_path / "named.env", "STRATA_USER=alice\n")
    monkeypatch.setenv("STRATAKB_CONFIG", str(named))
    assert "strata_user" in get_settings().model_fields_set

    monkeypatch.delenv("STRATAKB_CONFIG")
    assert "strata_user" not in get_settings().model_fields_set
    monkeypatch.setenv("STRATA_USER", "developer")
    assert "strata_user" in get_settings().model_fields_set
