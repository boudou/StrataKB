"""Where data leaves this machine, and the STRATA_LOCAL_ONLY switch.

A provider is outbound when StrataKB would send it content (documents,
questions or layer descriptions) at a host that is not loopback. The
database is not a provider: DATABASE_URL may point at a team server on
purpose. Nothing here reports key values, URL credentials, paths or ports.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import structlog

if TYPE_CHECKING:
    from stratakb.config import Settings

logger = structlog.get_logger()

_OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
_ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"


@dataclass(frozen=True)
class OutboundProvider:
    """A provider that receives content from this machine."""

    provider: str  # "anthropic" or "embeddings"
    purpose: str
    sends: str
    destination: str  # host name only
    setting: str  # the setting that turns it on, as NAME or NAME=value

    @property
    def setting_name(self) -> str:
        """The name of the setting to unset, without its value."""
        return self.setting.split("=", 1)[0]


def _host(url: str) -> str:
    """Return the host of a URL, or "" when it cannot be parsed."""
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def is_loopback(url: str) -> bool:
    """True when the URL's host is this machine (localhost, 127.0.0.0/8, ::1)."""
    host = _host(url)
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _embedding_url(settings: Settings) -> str:
    """The URL the openai backend sends to (the OpenAI SDK reads OPENAI_BASE_URL)."""
    return (
        settings.embedding_base_url
        or os.environ.get("OPENAI_BASE_URL")
        or _OPENAI_DEFAULT_URL
    )


def _anthropic_url() -> str:
    """The URL query rewriting sends to (the Anthropic SDK reads ANTHROPIC_BASE_URL)."""
    return os.environ.get("ANTHROPIC_BASE_URL") or _ANTHROPIC_DEFAULT_URL


def anthropic_provider(
    settings: Settings, api_key: str | None = None
) -> OutboundProvider | None:
    """Return query rewriting as a provider when it would leave this machine, else None.

    Args:
        settings: The settings to check.
        api_key: The key the rewrite would use; defaults to ANTHROPIC_API_KEY.
    """
    key = settings.anthropic_api_key if api_key is None else api_key
    url = _anthropic_url()
    if not key or is_loopback(url):
        return None
    return OutboundProvider(
        provider="anthropic",
        purpose="query rewriting",
        sends="questions and layer descriptions",
        destination=_host(url) or "unknown host",
        setting="ANTHROPIC_API_KEY",
    )


def embedding_provider(settings: Settings, backend: str = "") -> OutboundProvider | None:
    """Return embedding as a provider when it would leave this machine, else None.

    Any backend other than "local" goes down the OpenAI path, so a misspelt
    EMBEDDING_BACKEND is outbound too.

    Args:
        settings: The settings to check.
        backend: Embedding backend override, as passed to embed_texts.
    """
    backend = backend or settings.embedding_backend
    if backend == "local":
        return None
    url = _embedding_url(settings)
    if is_loopback(url):
        return None
    return OutboundProvider(
        provider="embeddings",
        purpose="embedding",
        sends="documents and questions",
        destination=_host(url) or "unknown host",
        setting=(
            "EMBEDDING_BASE_URL"
            if settings.embedding_base_url
            else f"EMBEDDING_BACKEND={backend}"
        ),
    )


def outbound_providers(settings: Settings, backend: str = "") -> list[OutboundProvider]:
    """List the providers that would receive content with these settings."""
    return [
        p
        for p in (anthropic_provider(settings), embedding_provider(settings, backend))
        if p
    ]


def refusal_line(p: OutboundProvider) -> str:
    """Say why STRATA_LOCAL_ONLY refuses a provider, and how to fix it."""
    return (
        f"STRATA_LOCAL_ONLY is on, but {p.setting} is set: {p.sends} would go to "
        f"{p.destination} for {p.purpose}. Unset {p.setting_name} or turn "
        "STRATA_LOCAL_ONLY off."
    )


def enforce_local_only(settings: Settings) -> None:
    """Refuse to start with an outbound provider, and keep Hugging Face offline.

    Called first in main(), before anything imports huggingface_hub, which
    reads its offline flags once at import.
    """
    if not settings.strata_local_only:
        return
    if providers := outbound_providers(settings):
        logger.error("local_only_refused", settings=[p.setting_name for p in providers])
        raise SystemExit("\n".join(refusal_line(p) for p in providers))
    # Assign, not setdefault: a user's HF_HUB_OFFLINE=0 must not win
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
