"""Reading URLs: the one place in src/ that parses them. Standard library only."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class UrlParts:
    """What StrataKB reads from a URL. host is "" and port None when the URL names none."""

    host: str
    port: int | None
    database: str  # the path without its leading "/"
    user: str | None
    password: str | None = field(default=None, repr=False)


def url_parts(url: str) -> UrlParts | None:
    """The parts of url, or None when it cannot be parsed (an invalid port included)."""
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

    The port is not read, so an invalid port does not hide the host.
    """
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def _at_after_authority(url: str) -> bool:
    """True when an "@" sits after the authority of url, read by hand.

    An unencoded "/", "?" or "#" in a password ends the authority early, so the
    parsed host and port would be the user and the start of the password.
    """
    start = url.find("//")
    rest = url[start + 2 :] if start >= 0 else url
    ends = [i for i in (rest.find(c) for c in "/?#") if i >= 0]
    return "@" in rest[min(ends, default=len(rest)) :]


def url_host_port(database_url: str) -> str:
    """host:port of a database URL, never its user, password, database name or query.

    "unknown host" when the URL cannot be read safely: an "@" after the
    authority means the parsed host would come from the credentials.
    """
    if _at_after_authority(database_url):
        return "unknown host"
    parts = url_parts(database_url)
    if parts is None:
        return "localhost:5432"
    return f"{parts.host or 'localhost'}:{parts.port or 5432}"
