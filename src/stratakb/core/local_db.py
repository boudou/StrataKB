"""A local database in Docker, for `stratakb db up` and `stratakb db down`.

Needs only the docker CLI: the container's settings live here and the schema
ships in the package, so no file from a checkout is read. Every failure is a
LocalDbError whose message is the one line the user sees.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

import asyncpg

from stratakb.config import (
    LOCAL_DB_NAME,
    LOCAL_DB_PASSWORD,
    LOCAL_DB_PORT,
    LOCAL_DB_USER,
    local_database_url,
)

CONTAINER = "stratakb-db"
VOLUME = "stratakb-data"
IMAGE = "pgvector/pgvector:pg16"

_READY_TIMEOUT_S = 60
_INFO_TIMEOUT_S = 20

START_HINT = "Run stratakb db up to start a local database."

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Docker Desktop on Windows says "Ports are not available" when a host process holds the port
_PORT_IN_USE = ("port is already allocated", "address already in use", "ports are not available")

UpState = Literal["created", "started", "running"]
DownOutcome = Literal["stopped", "not_running", "missing"]


class LocalDbError(Exception):
    """A failure of db up or db down; the message is the line the user sees."""


@dataclass
class UpResult:
    state: UpState
    port: int
    url: str


async def _docker(*args: str, timeout: float | None = None) -> tuple[int, str, str]:
    """Run docker; the only place that does. Returns (exit code, stdout, stderr)."""
    path = shutil.which("docker")
    if path is None:
        raise LocalDbError(
            "docker is not on PATH: install Docker (https://docs.docker.com/get-docker/), "
            "then run stratakb db up again."
        )
    try:
        proc = await asyncio.create_subprocess_exec(
            path,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        raise LocalDbError(
            f"docker on PATH could not be run ({type(e).__name__}): check the Docker installation."
        ) from e
    try:
        if timeout is None:
            out, err = await proc.communicate()
        else:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", "timed out"
    # communicate() waits for the process, so returncode is set
    assert proc.returncode is not None
    return (
        proc.returncode,
        out.decode("utf-8", errors="replace").strip(),
        err.decode("utf-8", errors="replace").strip(),
    )


def _first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[0] if lines else "no message"


async def _check_daemon() -> None:
    code, _, _ = await _docker("info", "--format", "{{.ServerVersion}}", timeout=_INFO_TIMEOUT_S)
    if code != 0:
        raise LocalDbError(
            "Docker is installed but its daemon is not running: start Docker Desktop "
            "(or the docker service), then run stratakb db up again."
        )


async def _status() -> str | None:
    """The container's state, or None when there is no such container."""
    code, out, err = await _docker(
        "container", "inspect", "--format", "{{.State.Status}}", CONTAINER
    )
    if code == 0:
        return out
    if "no such container" in err.lower():
        return None
    raise LocalDbError(f"docker container inspect {CONTAINER} failed: {_first_line(err)}")


async def _host_port() -> int:
    """The host port of a running container; docker prints none for a stopped one."""
    code, out, err = await _docker("port", CONTAINER, "5432/tcp")
    if code != 0:
        raise LocalDbError(f"docker port {CONTAINER} failed: {_first_line(err)}")
    try:
        return int(_first_line(out).rsplit(":", 1)[-1])
    except ValueError:
        raise LocalDbError(f"docker port {CONTAINER} printed no port.") from None


def points_here(database_url: str, port: int) -> bool:
    """True when database_url is the container's database on this port."""
    try:
        parts = urlsplit(database_url)
        return (
            parts.hostname in _LOCAL_HOSTS
            and (parts.port or 5432) == port
            and parts.path.lstrip("/") == LOCAL_DB_NAME
            and parts.username == LOCAL_DB_USER
            and parts.password == LOCAL_DB_PASSWORD
        )
    except ValueError:
        return False


async def _wait_ready(url: str) -> None:
    """Connect until PostgreSQL answers; Docker's proxy accepts TCP before it does."""
    deadline = time.monotonic() + _READY_TIMEOUT_S
    last: BaseException | None = None
    while True:
        try:
            conn = await asyncpg.connect(url, timeout=5)
        except (
            OSError,
            asyncpg.PostgresConnectionError,
            asyncpg.CannotConnectNowError,
            asyncpg.InterfaceError,
        ) as e:
            last = e
        else:
            await conn.close()
            return
        if time.monotonic() >= deadline:
            raise LocalDbError(
                f"The database in {CONTAINER} did not accept connections within "
                f"{_READY_TIMEOUT_S} s ({type(last).__name__}). See docker logs {CONTAINER}."
            )
        await asyncio.sleep(0.5)


def _port_in_use(err: str) -> bool:
    return any(text in err.lower() for text in _PORT_IN_USE)


def _port_message(port: int) -> str:
    return (
        f"Port {port} on 127.0.0.1 is already in use (another database, such as the "
        "docker compose container strata-postgres?). Stop it, or run "
        "stratakb db up --port <another port>."
    )


async def up(port: int = LOCAL_DB_PORT, notify: Callable[[str], None] | None = None) -> UpResult:
    """Start the container, creating it on first use, and bring its schema up to date."""
    say = notify or (lambda _text: None)
    await _check_daemon()
    status = await _status()

    state: UpState
    if status is None:
        say(f"Creating the container {CONTAINER} (the first run downloads the image {IMAGE})…")
        code, _, err = await _docker(
            "run", "-d",
            "--name", CONTAINER,
            "--restart", "unless-stopped",
            "-e", f"POSTGRES_DB={LOCAL_DB_NAME}",
            "-e", f"POSTGRES_USER={LOCAL_DB_USER}",
            "-e", f"POSTGRES_PASSWORD={LOCAL_DB_PASSWORD}",
            "-p", f"127.0.0.1:{port}:5432",
            "-v", f"{VOLUME}:/var/lib/postgresql/data",
            IMAGE,
        )  # fmt: skip
        if code != 0:
            # A failed port bind leaves a created container; the volume stays
            await _docker("rm", CONTAINER)
            if _port_in_use(err):
                raise LocalDbError(_port_message(port))
            raise LocalDbError(f"docker run failed: {_first_line(err)}")
        state = "created"
    elif status in ("running", "restarting"):
        state = "running"
    elif status == "paused":
        code, _, err = await _docker("unpause", CONTAINER)
        if code != 0:
            raise LocalDbError(f"docker unpause {CONTAINER} failed: {_first_line(err)}")
        state = "started"
    else:
        say(f"Starting the container {CONTAINER}…")
        code, _, err = await _docker("start", CONTAINER)
        if code != 0:
            if _port_in_use(err):
                raise LocalDbError(
                    f"{_port_message(port)} docker rm {CONTAINER} lets "
                    f"stratakb db up --port recreate it, with the data kept in {VOLUME}."
                )
            raise LocalDbError(f"docker start {CONTAINER} failed: {_first_line(err)}")
        state = "started"

    actual = await _host_port()
    if actual != port and state != "created":
        say(
            f"{CONTAINER} already uses port {actual}; "
            "--port applies only when the container is created."
        )
    url = local_database_url(actual)
    say("Waiting for the database…")
    await _wait_ready(url)

    # Looked up at call time, so a patched ensure_schema is the one called
    from stratakb.core import migrations

    await migrations.ensure_schema(url)
    return UpResult(state=state, port=actual, url=url)


async def down() -> DownOutcome:
    """Stop the container and keep its volume."""
    await _check_daemon()
    status = await _status()
    if status is None:
        return "missing"
    if status in ("running", "restarting", "paused"):
        code, _, err = await _docker("stop", CONTAINER)
        if code != 0:
            raise LocalDbError(f"docker stop {CONTAINER} failed: {_first_line(err)}")
        return "stopped"
    return "not_running"
