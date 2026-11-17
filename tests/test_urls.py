"""URL parsing in core/urls.py, and that the module needs nothing else from StrataKB."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pytest
from asyncpg import connect_utils

from stratakb.core.urls import url_host, url_host_port, url_parts, url_targets
from tests.helpers import MALFORMED_URLS

_FAKE_URL = "postgresql://strata:secret-pw@db.example:6543/strata"

# Every password below is six characters or more and appears in no host or
# database name, so the password checks cannot match host text by chance.

# Well-formed URLs, percent-encoded passwords included: (url, host, port, user, password)
WELL_FORMED_URLS = [
    (
        "postgresql://app:p%2F1%3F2%233%402@db.example:6543/strata",
        "db.example",
        6543,
        "app",
        "p%2F1%3F2%233%402",
    ),
    ("postgresql://app:123%2Fx@db.example/strata", "db.example", None, "app", "123%2Fx"),
    ("postgresql://app@db.example/strata", "db.example", None, "app", None),
    ("postgresql://app:pw-7319@[::1]:6543/strata", "::1", 6543, "app", "pw-7319"),
    ("postgresql://localhost", "localhost", None, None, None),
]

# (url, what url_host_port shows) for well-formed URLs
WELL_FORMED_HOST_PORTS = [
    ("postgresql://app:p%2F1%3F2%233%402@db.example:6543/strata", "db.example:6543"),
    ("postgresql://app:123%2Fx@db.example/strata", "db.example:5432"),
    ("postgresql://app@db.example/strata", "db.example:5432"),
    ("postgresql://app:pw-7319@[::1]:6543/strata", "[::1]:6543"),
    ("postgresql://localhost", "localhost:5432"),
]

# (url, what url_host_port shows): the database asyncpg connects to
ASYNCPG_TARGETS = [
    # Host and port in the query string, read when the authority has no host
    ("postgresql:///strata?host=db.example&port=6543", "db.example:6543"),
    ("postgresql:///strata?host=db.example", "db.example:5432"),
    ("postgresql:///strata?port=6543", "localhost:6543"),
    ("postgresql:///strata?host=/var/run/postgresql", "/var/run/postgresql:5432"),
    ("postgresql:///strata?host=h1,h2&port=5432,5433", "h1:5432,h2:5433"),
    ("postgresql:///strata?host=h1,h2&port=6543", "h1:6543,h2:6543"),
    ("postgresql:///strata?host=[::1]&port=6543", "[::1]:6543"),
    # An authority host wins over the query string, as in asyncpg
    ("postgresql://db.example/strata?host=other&port=6543", "db.example:5432"),
    # Several hosts in the authority, each with its own port
    ("postgresql://u:pw-7319@h1:5432,h2:5433/db", "h1:5432,h2:5433"),
    ("postgresql://u:pw-7319@h1,h2/db", "h1:5432,h2:5432"),
    ("postgresql://u:pw-7319@h1:6543,h2/db", "h1:6543,h2:5432"),
    ("postgresql://u:pw-7319@[::1]:6543,db.example/db", "[::1]:6543,db.example:5432"),
    # Query passwords are never read
    ("postgresql://db.example/strata?user=app&password=hunter2", "db.example:5432"),
    ("postgresql:///strata?host=db.example&user=app&password=hunter2", "db.example:5432"),
]

# URLs whose target cannot be read: url_host_port shows "unknown host"
UNREADABLE_URLS = [
    "not a url at all",
    "postgresql://u:pw-7319@db:99999/s",
    "postgresql://u:pw-7319@db:0/s",
    "postgresql://u:pw-7319@h1:5432,h2:99999/s",
    "postgresql://u:pw-7319@h1,,h2/s",
    "postgresql://u:pw-7319@localhost:99999/x",
    "postgresql://u:pw-7319@db.example:99999/x",
    "postgresql://[::1/x",
    "postgresql:///strata?host=db.example&port=abc",
    "postgresql:///strata?host=db.example&port=99999",
    "postgresql:///strata?host=h1,h2,h3&port=5432,5433",
    "postgresql:///strata?port=5432,5433",
    "postgresql:///strata?sslmode",
    "postgresql://db.example/strata?sslmode",
]


def _password_of(url: str) -> str:
    """The password written in url, read by hand: the query one, else the authority one."""
    match = re.search(r"[?&]password=([^&#]+)", url)
    if match:
        return match.group(1)
    start = url.find("//")
    if start < 0:
        return ""
    credentials = url[start + 2 :].rpartition("@")[0]
    return credentials.partition(":")[2]


_PASSWORD_CASES = [
    (url, password)
    for url in [
        _FAKE_URL,
        *(url for url, *_ in WELL_FORMED_URLS),
        *(url for url, _ in ASYNCPG_TARGETS),
        *UNREADABLE_URLS,
        *MALFORMED_URLS,
    ]
    if (password := _password_of(url))
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
        ["postgresql://u:pw-7319@localhost:99999/x", "http://[::1/x"],
    )
    def test_url_parts_unparsable(self, url):
        assert url_parts(url) is None

    @pytest.mark.parametrize("url", MALFORMED_URLS)
    def test_url_parts_refuses_a_malformed_url(self, url):
        assert url_parts(url) is None

    @pytest.mark.parametrize(("url", "host", "port", "user", "password"), WELL_FORMED_URLS)
    def test_url_parts_of_well_formed_urls(self, url, host, port, user, password):
        parts = url_parts(url)
        assert parts is not None
        assert (parts.host, parts.port, parts.user, parts.password) == (host, port, user, password)

    def test_url_parts_repr_hides_password(self):
        assert "secret-pw" not in repr(url_parts(_FAKE_URL))

    def test_url_host_ignores_an_invalid_port(self):
        assert url_host("http://localhost:99999/v1") == "localhost"

    def test_url_host_unparsable_or_missing(self):
        assert url_host("http://[::1/x") == ""
        assert url_host("not a url") == ""

    def test_url_host_keeps_an_ipv6_host_bare(self):
        assert url_host("http://[::1]:11434/v1") == "::1"

    @pytest.mark.parametrize("url", MALFORMED_URLS)
    def test_url_host_never_shows_credentials_of_a_malformed_url(self, url):
        assert url_host(url) == ""

    @pytest.mark.parametrize(("url", "host"), [(url, host) for url, host, *_ in WELL_FORMED_URLS])
    def test_url_host_of_well_formed_urls(self, url, host):
        assert url_host(url) == host

    def test_an_at_after_the_authority_fails_safe(self):
        # A well-formed URL with an "@" in its query reads like the malformed
        # shape, so no host is read from it: failing safe is intended.
        url = "postgresql://db.example/strata?user=me@corp"
        assert url_host(url) == ""
        assert url_parts(url) is None
        assert url_targets(url) is None
        assert url_host_port(url) == "unknown host"

    def test_url_host_port(self):
        assert url_host_port(_FAKE_URL) == "db.example:6543"
        assert url_host_port("postgresql:///strata") == "localhost:5432"

    def test_url_host_port_never_shows_credentials_or_database(self):
        shown = url_host_port(_FAKE_URL)
        for secret in ("secret-pw", "strata@", "/strata"):
            assert secret not in shown

    @pytest.mark.parametrize("url", MALFORMED_URLS)
    def test_url_host_port_never_shows_credentials_of_a_malformed_url(self, url):
        assert url_host_port(url) == "unknown host"

    @pytest.mark.parametrize(("url", "shown"), WELL_FORMED_HOST_PORTS)
    def test_url_host_port_of_well_formed_urls(self, url, shown):
        assert url_host_port(url) == shown

    @pytest.mark.parametrize(("url", "shown"), ASYNCPG_TARGETS)
    def test_url_host_port_names_the_database_asyncpg_connects_to(self, url, shown):
        assert url_host_port(url) == shown

    @pytest.mark.parametrize("url", UNREADABLE_URLS)
    def test_url_host_port_of_an_unreadable_target(self, url):
        assert url_host_port(url) == "unknown host"

    def test_query_port_list_must_match_the_implicit_localhost(self):
        assert url_host_port("postgresql:///strata?port=6543") == "localhost:6543"
        assert url_host_port("postgresql:///strata?port=5432,5433") == "unknown host"

    def test_url_targets(self):
        assert url_targets("postgresql://u:pw-7319@[::1]:6543/x") == [("::1", 6543)]
        assert url_targets("postgresql:///strata") == [("localhost", 5432)]
        assert url_targets("postgresql:///strata", implicit_localhost=False) is None
        assert url_targets("postgresql:///strata?port=6543", implicit_localhost=False) is None
        assert url_targets(
            "postgresql:///strata?host=127.0.0.1&port=5433", implicit_localhost=False
        ) == [("127.0.0.1", 5433)]

    @pytest.mark.parametrize(
        "url",
        [
            "postgresql://app:pw@7319@db.example/strata",
            "postgresql://me@corp:pw-7319@db.example/strata",
        ],
    )
    def test_two_ats_in_the_authority_are_refused(self, url):
        shown = [url_host_port(url), url_host(url), repr(url_parts(url)), repr(url_targets(url))]
        assert shown == ["unknown host", "", "None", "None"]
        for text in shown:
            assert "7319" not in text

    def test_local_database_check_reads_no_host_after_a_second_at(self):
        url = "postgresql://strata:strata_dev_password@x@127.0.0.1:5433/strata"
        assert url_targets(url, implicit_localhost=False) is None

    def test_every_url_case_has_its_password_checked(self):
        # Guards _password_of: a case list it reads nothing from would pass silently.
        assert len(_PASSWORD_CASES) >= 15
        assert ("postgresql://app:123/x@db/strata", "123/x") in _PASSWORD_CASES

    @pytest.mark.parametrize(("url", "password"), _PASSWORD_CASES)
    def test_no_reader_shows_a_password(self, url, password):
        shown = [
            url_host_port(url),
            url_host(url),
            repr(url_parts(url)),
            repr(url_targets(url)),
        ]
        for secret in {password, unquote(password)}:
            for text in shown:
                assert secret not in text


def _asyncpg_targets(url: str) -> list:
    """The addresses asyncpg reads from url (a private function, called by keyword)."""
    parse = connect_utils._parse_connect_dsn_and_args
    arguments = {name: None for name in inspect.signature(parse).parameters} | {"dsn": url}
    return parse(**arguments)[0]


def test_asyncpg_reads_two_ats_differently(monkeypatch):
    # asyncpg splits the authority at the first "@", urlsplit at the last, so
    # they read different hosts; core/urls.py reads none.
    monkeypatch.delenv("PGPORT", raising=False)
    url = "postgresql://app:pw@7319@db.example/strata"
    assert _asyncpg_targets(url) == [("7319@db.example", 5432)]
    assert urlsplit(url).hostname == "db.example"
    assert url_parts(url) is None
    assert url_host_port(url) == "unknown host"


# The rows whose authority names a host: asyncpg's Unix-socket and
# environment defaults apply to the others, which url_targets does not read
_AUTHORITY_HOST_URLS = [
    url
    for url, _ in [*WELL_FORMED_HOST_PORTS, *ASYNCPG_TARGETS]
    if urlsplit(url).netloc.rpartition("@")[2]
]


@pytest.mark.parametrize("url", _AUTHORITY_HOST_URLS)
def test_url_targets_agree_with_asyncpg(monkeypatch, url):
    monkeypatch.delenv("PGPORT", raising=False)
    assert url_targets(url) == [tuple(address) for address in _asyncpg_targets(url)]


def test_egress_does_not_import_the_database_module():
    code = (
        "import sys; import stratakb.core.egress; "
        "sys.exit(any(m in sys.modules for m in ('stratakb.core.database', 'asyncpg')))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
