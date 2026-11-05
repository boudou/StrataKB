"""Query rewriting engine.

Asks a model to rewrite a plain-language question into 2-4 search queries
that use the terminology of the knowledge base, improving recall. The model
is the Anthropic API or any OpenAI-compatible endpoint (QUERY_REWRITE_PROVIDER).
The prompt is domain-neutral; what the knowledge base covers comes from the
layer descriptions passed in as domain_context.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from stratakb.config import Settings

logger = structlog.get_logger()

_PROVIDERS = ("anthropic", "openai_compatible")

# Seconds per request to an OpenAI-compatible endpoint, with no retries: the
# SDK default (600 s, 2 retries) would hold strata_ask on a dead local server
_OPENAI_COMPATIBLE_TIMEOUT = 60.0

# Replies with no usable JSON array since the server started
_parse_failures = 0


def _safe_domain_context(domain_context: str | None) -> str:
    """Return domain context as safely-quoted data for prompt interpolation.

    The context may originate from DB-backed descriptions. Treat it as data
    (not instructions) by neutralizing common prompt-boundary tokens and
    JSON-quoting the final value.
    """
    raw = domain_context or _DEFAULT_DOMAIN_CONTEXT
    cleaned = (
        raw.replace("```", "`\u200b``")
        .replace("<system>", "< system >")
        .replace("</system>", "< /system >")
        .replace("<assistant>", "< assistant >")
        .replace("</assistant>", "< /assistant >")
        .replace("\x00", " ")
    )
    return json.dumps(cleaned, ensure_ascii=False)


_SYSTEM_PROMPT = """\
You are a query rewriter for a retrieval system. Take the user's question and produce 2-4 alternative search queries that use the precise terminology of the knowledge base described below.

{domain_context}

Rules:
1. Output ONLY a JSON array of 2-4 strings, with no explanation and no markdown
2. Keep the original intent, and add the domain terms, identifiers and acronyms a document on the topic would use
3. Expand abbreviations and add common synonyms where useful
4. Keep queries concise (under 50 words each)

Example:
Input: "What happens when a payment is rejected?"
Output: ["payment rejection reason codes", "payment status report rejected transaction", "rejected payment handling process", "return and reject reason code list"]
"""

_DEFAULT_DOMAIN_CONTEXT = "The knowledge base contains technical and business documentation."


def build_domain_context(descriptions: dict[str, str]) -> str:
    """Describe the knowledge base from its layers' display names and descriptions."""
    if not descriptions:
        return _DEFAULT_DOMAIN_CONTEXT
    lines = [f"- **{name}**: {text}" for name, text in descriptions.items()]
    return "The knowledge base contains these layers:\n" + "\n".join(lines)


def _skip_reason(settings: Settings, anthropic_api_key: str, model: str) -> str | None:
    """Say why no model would be asked to rewrite, or None when one would."""
    from stratakb.core.egress import rewrite_model_unset

    provider = settings.query_rewrite_provider
    if provider == "anthropic":
        return None if anthropic_api_key else "no_api_key"
    if provider == "openai_compatible":
        if not settings.query_rewrite_base_url:
            return "no_base_url"
        if rewrite_model_unset(model):
            return "model_not_set"
        return None
    return "unknown_provider"


def rewrite_configured(settings: Settings, anthropic_api_key: str | None = None) -> bool:
    """True when rewrite_query would ask a model with these settings.

    Does not apply STRATA_LOCAL_ONLY; rewrite_query checks that itself.

    Args:
        settings: The settings to check.
        anthropic_api_key: The Anthropic key; defaults to ANTHROPIC_API_KEY.
    """
    key = settings.anthropic_api_key if anthropic_api_key is None else anthropic_api_key
    return _skip_reason(settings, key, settings.query_rewrite_model) is None


def _extract_queries(text: str | None) -> list[str] | None:
    """Return the first JSON array of strings in a reply, or None when there is none.

    The array may sit inside a code fence or among other text. Strings are
    stripped and empty ones dropped; an array left empty counts as none.
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    start = text.find("[")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list) and all(isinstance(q, str) for q in value):
            queries = [q.strip() for q in value if q.strip()]
            return queries[:4] or None
        start = text.find("[", start + 1)
    return None


async def _call_anthropic(api_key: str, model: str, system: str, query: str) -> str | None:
    """Ask the Anthropic API and return the reply text."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text if response.content else None


async def _call_openai_compatible(
    settings: Settings, model: str, system: str, query: str
) -> str | None:
    """Ask the OpenAI-compatible endpoint in QUERY_REWRITE_BASE_URL and return the reply text."""
    from stratakb.core.egress import keyless_request_options, openai_compatible_client

    client = openai_compatible_client(
        settings.query_rewrite_api_key,
        settings.query_rewrite_base_url,
        timeout=_OPENAI_COMPATIBLE_TIMEOUT,
        max_retries=0,
    )
    response = await client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        **keyless_request_options(settings.query_rewrite_api_key),
    )
    if not response.choices:
        return None
    return response.choices[0].message.content


async def rewrite_query(
    query: str,
    *,
    anthropic_api_key: str = "",
    model: str = "",
    learning_store=None,
    domain_context: str = "",
) -> list[str]:
    """Rewrite a user query into 2-4 domain-specific search queries.

    If a learning store is provided, consults learned rewrites and synonyms
    before asking a model. High-confidence learned rewrites bypass the model
    entirely. The model is chosen by QUERY_REWRITE_PROVIDER.

    Args:
        query: The user's natural language question.
        anthropic_api_key: Anthropic API key. If empty with the anthropic
            provider, returns [query] unchanged; unused with openai_compatible.
        model: Model to use. Defaults to config's query_rewrite_model.
        learning_store: Optional LearningStore for consulting learned patterns.
        domain_context: What the knowledge base covers (see build_domain_context).

    Returns:
        List of 2-4 rewritten queries. Falls back to [query] on any error.
    """
    enriched_query = query

    # Consult learned patterns if available
    if learning_store:
        try:
            from stratakb.config import get_settings
            _ls = get_settings()

            # Check synonyms
            synonyms = await learning_store.find_synonyms(query)
            if synonyms:
                enriched_query = learning_store.apply_synonyms(query, synonyms)
                logger.info("learning_synonyms_applied", count=len(synonyms))

            # Check learned rewrites
            learned = await learning_store.find_rewrites(
                query, min_success_rate=_ls.learning_min_success_rate
            )
            if (
                learned
                and learned.success_rate >= _ls.learning_rewrite_bypass_threshold
                and learned.total_count >= _ls.learning_rewrite_bypass_min_samples
            ):
                logger.info(
                    "learning_rewrite_bypass",
                    pattern=learned.pattern[:50],
                    success_rate=learned.success_rate,
                )
                return learned.queries

        except Exception as e:
            logger.debug("learning_consult_failed", error=str(e))

    from stratakb.config import get_settings
    from stratakb.core.egress import rewrite_provider

    settings = get_settings()
    provider = settings.query_rewrite_provider
    model = model or settings.query_rewrite_model

    reason = _skip_reason(settings, anthropic_api_key, model)
    if reason == "no_api_key":
        logger.debug("query_rewrite_skipped", reason=reason)
        return [enriched_query]
    if reason == "unknown_provider":
        logger.warning(
            "query_rewrite_skipped",
            reason=reason,
            provider=provider,
            accepted=list(_PROVIDERS),
        )
        return [enriched_query]
    if reason:
        logger.warning("query_rewrite_skipped", reason=reason, provider=provider)
        return [enriched_query]

    # Learned rewrites and synonyms above are local and stay on
    if settings.strata_local_only and rewrite_provider(settings, anthropic_api_key, model):
        logger.info("query_rewrite_skipped", reason="local_only")
        return [enriched_query]

    system = _SYSTEM_PROMPT.format(domain_context=_safe_domain_context(domain_context))
    try:
        if provider == "openai_compatible":
            text = await _call_openai_compatible(settings, model, system, enriched_query)
        else:
            text = await _call_anthropic(anthropic_api_key, model, system, enriched_query)
    except Exception as e:
        hint = None
        if provider == "openai_compatible":
            from stratakb.core.egress import unauthorized_detail

            hint = unauthorized_detail(e, settings.query_rewrite_api_key, "QUERY_REWRITE_API_KEY")
        if hint:
            logger.warning("query_rewrite_failed", error=str(e), query=query[:100], hint=hint)
        else:
            logger.warning("query_rewrite_failed", error=str(e), query=query[:100])
        return [enriched_query]

    queries = _extract_queries(text)
    if queries is None:
        # Never log the reply itself: it may echo the question and layer descriptions
        global _parse_failures
        _parse_failures += 1
        logger.warning(
            "query_rewrite_json_error",
            provider=provider,
            failures=_parse_failures,
            reply_chars=len(text or ""),
            query=query[:100],
        )
        return [enriched_query]

    logger.info(
        "query_rewritten",
        original=query[:100],
        rewritten_count=len(queries),
        model=model,
        provider=provider,
    )
    return queries
