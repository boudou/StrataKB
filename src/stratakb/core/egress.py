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
# Satisfies the SDK's construction check; never sent, see keyless_request_options
PLACEHOLDER_API_KEY = "unused"


@dataclass(frozen=True)
class OutboundProvider:
    """A provider that receives content from this machine."""

    provider: str  # "anthropic", "openai_compatible" or "embeddings"
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


def rewrite_model_unset(model: str) -> bool:
    """True when an OpenAI-compatible rewrite has no model of its own.

    The QUERY_REWRITE_MODEL default is an Anthropic model name, which the
    endpoint would not know, so the rewrite is skipped and sends nothing.
    """
    from stratakb.config import Settings

    return model == Settings.model_fields["query_rewrite_model"].default


def openai_compatible_client(api_key: str, base_url: str, **options):
    """Build an AsyncOpenAI client for a base URL from StrataKB's own settings.

    The key is always a string, so the SDK reads no OPENAI_API_KEY, and
    admin_api_key is empty, so it reads no OPENAI_ADMIN_KEY. With no
    organization or project the SDK reads OPENAI_ORG_ID and OPENAI_PROJECT_ID
    and sends them as headers; Omit() drops both, so neither reaches the
    configured endpoint.

    An empty key means a keyless server (a default Ollama, llama.cpp or
    vLLM). The SDK refuses to build a client without a key, so it gets
    PLACEHOLDER_API_KEY; every call passes keyless_request_options(api_key),
    which sends no Authorization header at all.
    """
    import openai

    return openai.AsyncOpenAI(
        api_key=api_key or PLACEHOLDER_API_KEY,
        admin_api_key="",
        base_url=base_url,
        default_headers={"OpenAI-Organization": openai.Omit(), "OpenAI-Project": openai.Omit()},
        **options,
    )


def keyless_request_options(api_key: str) -> dict:
    """Return the per-call options for an openai_compatible_client request.

    With no key the request carries no Authorization header. The SDK accepts
    an omitted Authorization only from the call's own headers, not from the
    client defaults, so this goes into every call. With a key it is empty.
    """
    if api_key:
        return {}
    import openai

    return {"extra_headers": {"Authorization": openai.Omit()}}


def unauthorized_detail(error: Exception, api_key: str, setting: str) -> str | None:
    """Explain a 401 from an OpenAI-compatible server, naming the key setting.

    Returns None for any other error. Names the setting only, never its
    value, the URL or the error text.
    """
    if getattr(error, "status_code", None) != 401:
        return None
    if not api_key:
        return (
            f"The server answered 401 (unauthorized) and {setting} is not set. "
            "Set it to the key the server expects, then restart StrataKB."
        )
    return (
        f"The server answered 401 (unauthorized) to {setting}. "
        "Check the key, then restart StrataKB."
    )


def rewrite_provider(
    settings: Settings, api_key: str | None = None, model: str = ""
) -> OutboundProvider | None:
    """Return query rewriting as a provider when it would leave this machine, else None.

    Follows QUERY_REWRITE_PROVIDER: "anthropic" sends to the Anthropic API,
    "openai_compatible" to QUERY_REWRITE_BASE_URL, but only once a model is
    set (see rewrite_model_unset). Any other value rewrites nothing, so it is
    not outbound.

    Args:
        settings: The settings to check.
        api_key: The Anthropic key the rewrite would use; defaults to ANTHROPIC_API_KEY.
        model: The model the rewrite would use; defaults to QUERY_REWRITE_MODEL.
    """
    provider = settings.query_rewrite_provider
    if provider == "anthropic":
        return anthropic_provider(settings, api_key)
    if provider != "openai_compatible":
        return None
    if rewrite_model_unset(model or settings.query_rewrite_model):
        return None
    url = settings.query_rewrite_base_url
    if not url or is_loopback(url):
        return None
    return OutboundProvider(
        provider="openai_compatible",
        purpose="query rewriting",
        sends="questions and layer descriptions",
        destination=_host(url) or "unknown host",
        setting="QUERY_REWRITE_BASE_URL",
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
        for p in (rewrite_provider(settings), embedding_provider(settings, backend))
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
    keep_hugging_face_offline()


def keep_hugging_face_offline() -> None:
    """Make the Hugging Face libraries use local files only.

    Call before anything imports huggingface_hub, which reads these once.
    """
    # Assign, not setdefault: a user's HF_HUB_OFFLINE=0 must not win
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
