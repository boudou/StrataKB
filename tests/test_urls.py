"""URL parsing in core/urls.py, and that the module needs nothing else from StrataKB."""

from __future__ import annotations

import subprocess
import sys

import pytest

from stratakb.core.urls import url_host, url_host_port, url_parts

_FAKE_URL = "postgresql://strata:secret-pw@db.example:6543/strata"

# Unencoded "/", "?" or "#" in the password: urlsplit reads the user as the
# host and the start of the password as the port.
MALFORMED_URLS = [
    "postgresql://app:123/x@db/strata",
    "postgresql://app:99/secret@db:5432/s",
    "postgresql://app:123?x@db/s",
    "postgresql://app:123#x@db/s",
]


class TestUrlParsing:
    def test_url_parts(self):
        parts = url_parts(_FAKE_URL)
        assert parts is not None
        assert (parts.host, parts.port, parts.database, parts.user, parts.password) == (
            "db.example",
            6543,
            "strata",
            "strata",
            "secret-pw",
        )

    def test_url_parts_without_host_or_port(self):
        parts = url_parts("postgresql:///strata")
        assert parts is not None
        assert (parts.host, parts.port, parts.database) == ("", None, "strata")

    @pytest.mark.parametrize(
        "url",
        ["postgresql://u:p@localhost:99999/x", "http://[::1/x"],
    )
    def test_url_parts_unparsable(self, url):
        assert url_parts(url) is None

    def test_url_parts_repr_hides_password(self):
        assert "secret-pw" not in repr(url_parts(_FAKE_URL))

    def test_url_host_ignores_an_invalid_port(self):
        assert url_host("http://localhost:99999/v1") == "localhost"

    def test_url_host_unparsable_or_missing(self):
        assert url_host("http://[::1/x") == ""
        assert url_host("not a url") == ""

    def test_url_host_port(self):
        assert url_host_port(_FAKE_URL) == "db.example:6543"
        assert url_host_port("postgresql:///strata") == "localhost:5432"
        assert url_host_port("postgresql://u:p@localhost:99999/x") == "localhost:5432"

    def test_url_host_port_never_shows_credentials_or_database(self):
        shown = url_host_port(_FAKE_URL)
        for secret in ("secret-pw", "strata@", "/strata"):
            assert secret not in shown

    def test_url_host_port_never_shows_a_query_string_password(self):
        shown = url_host_port("postgresql://db.example/strata?user=app&password=hunter2")
        assert shown == "db.example:5432"

    @pytest.mark.parametrize("url", MALFORMED_URLS)
    def test_url_host_port_never_shows_credentials_of_a_malformed_url(self, url):
        shown = url_host_port(url)
        assert shown == "unknown host"
        for secret in ("app", "123", "99", "secret"):
            assert secret not in shown

    @pytest.mark.parametrize(
        ("url", "shown"),
        [
            ("postgresql://app:p%2F1%3F2%233%402@db.example:6543/strata", "db.example:6543"),
            ("postgresql://app:123%2Fx@db.example/strata", "db.example:5432"),
            ("postgresql://app@db.example/strata", "db.example:5432"),
            ("postgresql://app:pw@[::1]:6543/strata", "::1:6543"),
            ("postgresql://localhost", "localhost:5432"),
        ],
    )
    def test_url_host_port_of_well_formed_urls(self, url, shown):
        assert url_host_port(url) == shown


def test_egress_does_not_import_the_database_module():
    code = (
        "import sys; import stratakb.core.egress; "
        "sys.exit(any(m in sys.modules for m in ('stratakb.core.database', 'asyncpg')))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
