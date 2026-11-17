"""Reading URLs: the one place in src/ that parses them. Standard library only."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import SplitResult, parse_qsl, unquote, urlsplit

_AUTHORITY_END = re.compile(r"[/?#]")
_BRACKETED_HOST = re.compile(r"\[([^\]]+)\](?::([0-9]+))?")
_DATABASE_SCHEMES = ("postgresql", "postgres")
_DEFAULT_PORT = 5432


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


def _split(url: str) -> SplitResult | None:
    """url split into its parts, or None when an "@" sits after the authority
    or it cannot be split. Every reader below starts here."""
    if _at_after_authority(url):
        return None
    try:
        return urlsplit(url)
    except ValueError:
        return None


def url_parts(url: str) -> UrlParts | None:
    """The parts of url, or None when it cannot be parsed (an invalid port included).

    None too when an "@" sits after the authority: the host, port and user
    would then be read from the credentials.
    """
    parts = _split(url)
    if parts is None:
        return None
    try:  # .port raises only when read
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
    parts = _split(url)
    return (parts.hostname or "") if parts is not None else ""


def url_host_port(database_url: str) -> str:
    """host:port of a database URL, never its user, password, database name or query.

    The target the URL names, read the way asyncpg reads it: the hosts of the
    authority, else the host and port of the query string, else localhost;
    each host with its own port (5432 when none is given), several joined by
    ",", IPv6 hosts in brackets ("[::1]:6543"). "unknown host" whenever the
    target cannot be read: not a postgresql:// URL, an invalid or out-of-range
    port, an unclosed bracket, an empty host in a list, ports that do not match
    the hosts (an implicit localhost counts as one), or an "@" after the
    authority. Only the URL is read, not PGHOST/PGPORT, a service file or
    asyncpg's Unix-socket defaults, so a URL with no host shows localhost.
    """
    targets = url_targets(database_url)
    if targets is None:
        return "unknown host"
    return ",".join(_host_port_text(host, port) for host, port in targets)


def _host_port_text(host: str, port: int) -> str:
    """host:port, with an IPv6 host in brackets so the two can be told apart."""
    if ":" in host and not host.startswith("/"):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def url_targets(
    database_url: str, *, implicit_localhost: bool = True
) -> list[tuple[str, int]] | None:
    """The (host, port) pairs asyncpg reads from database_url, or None when it cannot.

    Hosts are bare ("::1"). A URL that names no host gives localhost, or None
    when implicit_localhost is False. See url_host_port for what is read.
    """
    parts = _split(database_url)
    if parts is None or parts.scheme not in _DATABASE_SCHEMES:
        return None
    host_text = port_text = ""
    try:  # asyncpg parses the query strictly first, whatever the authority holds
        for key, value in parse_qsl(parts.query, strict_parsing=True) if parts.query else ():
            if key == "host":
                host_text = value
            elif key == "port":
                port_text = value
    except ValueError:
        return None
    hostlist = parts.netloc.rpartition("@")[2]
    if hostlist:  # asyncpg ignores the query host and port then
        return _host_ports(hostlist, None, unquote_parts=True)
    ports = None
    if port_text:
        ports = [_port(text) for text in port_text.split(",")]
        if None in ports:
            return None
    if not host_text:
        if not implicit_localhost:
            return None
        host_text = "localhost"
    return _host_ports(host_text, ports, unquote_parts=False)


def _host_ports(
    hostlist: str, ports: list[int] | None, *, unquote_parts: bool
) -> list[tuple[str, int]] | None:
    """Read a comma-separated host list the way asyncpg does.

    ports, from the query string, replaces the ports in the list: one port
    serves every host, several must match the hosts one to one.
    """
    specs = hostlist.split(",")
    if ports is not None:
        if len(ports) == 1:
            ports = ports * len(specs)
        elif len(ports) != len(specs):
            return None
    targets = []
    for i, spec in enumerate(specs):
        if spec.startswith("/"):  # a Unix socket directory
            host, port_text = spec, ""
        elif spec.startswith("["):
            match = _BRACKETED_HOST.fullmatch(spec)
            if match is None:
                return None
            host, port_text = match.group(1), match.group(2) or ""
        else:
            host, _, port_text = spec.partition(":")
        if unquote_parts:
            host, port_text = unquote(host), unquote(port_text)
        if not host:
            return None
        if ports is not None:
            port = ports[i]
        elif port_text:
            port = _port(port_text)
            if port is None:
                return None
        else:
            port = _DEFAULT_PORT
        targets.append((host, port))
    return targets


def _port(text: str) -> int | None:
    """text as a TCP port, or None when it is not one."""
    if not (text.isascii() and text.isdigit()):
        return None
    port = int(text)
    return port if 0 < port <= 65535 else None
