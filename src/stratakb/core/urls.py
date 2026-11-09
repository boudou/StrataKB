"""Reading URLs: the one place in src/ that parses them. Standard library only."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

_AUTHORITY_END = re.compile(r"[/?#]")


@dataclass(frozen=True)
class UrlParts:
    """What StrataKB reads from a URL. host is "" and port None when the URL names none."""

    host: str
    port: int | None
    database: str  # the path without its leading "/"
    user: str | None
    password: str | None = field(default=None, repr=False)


def _at_after_authority(url: str) -> bool:
    """True when an "@" sits after the authority of url, read by hand.

    An unencoded "/", "?" or "#" in a password ends the authority early, so the
    parsed host and port would be the user and the start of the password.

    A well-formed URL with an "@" in its path, query or fragment
    ("postgresql://db.example/strata?user=me@corp") cannot be told apart from
    that shape, so it is refused too: failing safe is intended.
    """
    start = url.find("//")
    rest = url[start + 2 :] if start >= 0 else url
    end = _AUTHORITY_END.search(rest)
    return end is not None and "@" in rest[end.start() :]


def url_parts(url: str) -> UrlParts | None:
    """The parts of url, or None when it cannot be parsed (an invalid port included).

    None too when an "@" sits after the authority: the host, port and user
    would then be read from the credentials.
    """
    if _at_after_authority(url):
        return None
    try:
        parts = urlsplit(url)
        return UrlParts(
            host=parts.hostname or "",
            port=parts.port,
            database=parts.path.lstrip("/"),
            user=parts.username,
            password=parts.password,
        )
    except ValueError:
        return None


def url_host(url: str) -> str:
    """The host of url, or "" when it names none or cannot be parsed.

    The port is not read, so an invalid port does not hide the host. An "@"
    after the authority gives "", as the host would be read from the user name.
    """
    if _at_after_authority(url):
        return ""
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def url_host_port(database_url: str) -> str:
    """host:port of a database URL, never its user, password, database name or query.

    "unknown host" whenever url_parts cannot read the URL: an invalid port, an
    unclosed bracket, or an "@" after the authority.
    """
    parts = url_parts(database_url)
    if parts is None:
        return "unknown host"
    return f"{parts.host or 'localhost'}:{parts.port or 5432}"
