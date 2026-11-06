"""Where data leaves this machine, and the STRATA_LOCAL_ONLY switch.

A provider is outbound when StrataKB would send it content (documents,
questions or layer descriptions) at a host that is not loopback. The
database is not a provider: DATABASE_URL may point at a team server on
purpose. Nothing here reports key values, URL credentials, paths or ports.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import structlog

from stratakb.config import DEFAULT_QUERY_REWRITE_MODEL
from stratakb.core.database import url_host

if TYPE_CHECKING:
    from stratakb.config import Settings

logger = structlog.get_logger()

_OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
_ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"
# Satisfies the SDK's construction check; never sent, see keyless_request_options
PLACEHOLDER_API_KEY = "unused"

# The QUERY_REWRITE_PROVIDER values; each has one entry in REWRITE_EGRESS below
# and one in stratakb.rag.query_rewriter.REWRITE_PROVIDERS
RewriteProviderName = Literal["anthropic", "openai_compatible"]
ProviderKind = Literal[RewriteProviderName, "embeddings"]


@dataclass(frozen=True)
class OutboundProvider:
    """A provider that receives content from this machine."""

    provider: ProviderKind
    purpose: str
    sends: str
    destination: str  # host name only
    setting_name: str  # the setting that turns it on, and the one to unset
    setting_value: str = ""  # shown with the name when the value turns it on

    @property
    def setting(self) -> str:
        """The setting that turns it on, as NAME or NAME=value."""
        if self.setting_value:
            return f"{self.setting_name}={self.setting_value}"
        return self.setting_name

    def report(self) -> dict[str, str]:
        """Describe the provider for strata_admin_health."""
        return {
            "provider": self.provider,
            "purpose": self.purpose,
            "sends": self.sends,
            "destination": self.destination,
            "setting": self.setting,
        }


def is_loopback(url: str) -> bool:
    """True when the URL's host is this machine (localhost, 127.0.0.0/8, ::1)."""
    host = url_host(url)
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


class RewriteRequest(NamedTuple):
    """What one rewrite would use; every rewrite provider function takes this."""

    settings: Settings
    anthropic_api_key: str
    model: str


class RewriteEgress(NamedTuple):
    """What one QUERY_REWRITE_PROVIDER value sends where, and when it would not."""

    name: RewriteProviderName
    # The setting that turns sending on, for the egress report
    setting: str
    # Why nothing would be asked, or None
    skip_reason: Callable[[RewriteRequest], str | None]
    # Where the provider's SDK sends by itself; None means QUERY_REWRITE_BASE_URL
    sdk_url: Callable[[], str] | None = None
    sends: str = "questions and layer descriptions"

    def url(self, settings: Settings) -> str:
        """Where the request goes."""
        return self.sdk_url() if self.sdk_url else settings.query_rewrite_base_url


def _anthropic_skip(request: RewriteRequest) -> str | None:
    return None if request.anthropic_api_key else "no_api_key"


def _openai_compatible_skip(request: RewriteRequest) -> str | None:
    if not request.settings.query_rewrite_base_url:
        return "no_base_url"
    if request.model == DEFAULT_QUERY_REWRITE_MODEL:
        # An Anthropic model name the endpoint would not know: nothing is sent
        return "model_not_set"
    return None


# Adding a provider: one entry here, its name in RewriteProviderName, and how
# to ask it in stratakb.rag.query_rewriter.REWRITE_PROVIDERS
REWRITE_EGRESS: dict[str, RewriteEgress] = {
    e.name: e
    for e in (
        RewriteEgress(
            name="anthropic",
            setting="ANTHROPIC_API_KEY",
            skip_reason=_anthropic_skip,
            sdk_url=_anthropic_url,
        ),
        RewriteEgress(
            name="openai_compatible",
            setting="QUERY_REWRITE_BASE_URL",
            skip_reason=_openai_compatible_skip,
        ),
    )
}


def rewrite_skip_reason(entry: RewriteEgress | None, request: RewriteRequest) -> str | None:
    """Say why no model would be asked to rewrite, or None when one would.

    Args:
        entry: The QUERY_REWRITE_PROVIDER entry; None when the value has none.
        request: What the rewrite would use.
    """
    if entry is None:
        return "unknown_provider"
    return entry.skip_reason(request)


def outbound_rewrite(entry: RewriteEgress, settings: Settings) -> OutboundProvider | None:
    """Return a rewrite that would ask a model as a provider, or None at a loopback URL."""
    url = entry.url(settings)
    if is_loopback(url):
        return None
    return OutboundProvider(
        provider=entry.name,
        purpose="query rewriting",
        sends=entry.sends,
        destination=url_host(url) or "unknown host",
        setting_name=entry.setting,
    )


def rewrite_provider(settings: Settings) -> OutboundProvider | None:
    """Return query rewriting as a provider when it would leave this machine, else None.

    The rewrite is outbound when the QUERY_REWRITE_PROVIDER entry would ask a
    model at a URL that is not loopback. A value with no entry rewrites
    nothing, so it is not outbound.

    Args:
        settings: The settings to check.
    """
    entry = REWRITE_EGRESS.get(settings.query_rewrite_provider)
    request = RewriteRequest(settings, settings.anthropic_api_key, settings.query_rewrite_model)
    if entry is None or rewrite_skip_reason(entry, request):
        return None
    return outbound_rewrite(entry, settings)


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
    by_url = bool(settings.embedding_base_url)
    return OutboundProvider(
        provider="embeddings",
        purpose="embedding",
        sends="documents and questions",
        destination=url_host(url) or "unknown host",
        setting_name="EMBEDDING_BASE_URL" if by_url else "EMBEDDING_BACKEND",
        setting_value="" if by_url else backend,
    )


def outbound_providers(settings: Settings) -> list[OutboundProvider]:
    """List the providers that would receive content with these settings."""
    return [p for p in (rewrite_provider(settings), embedding_provider(settings)) if p]


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
