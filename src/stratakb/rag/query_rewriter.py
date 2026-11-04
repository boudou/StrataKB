"""Query rewriting engine.

Calls the Anthropic API to rewrite a plain-language question into 2-4 search
queries that use the terminology of the knowledge base, improving recall.
The prompt is domain-neutral; what the knowledge base covers comes from the
layer descriptions passed in as domain_context.
"""

from __future__ import annotations

import json

import structlog

logger = structlog.get_logger()

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
    before calling the Anthropic API. High-confidence learned rewrites bypass
    the API call entirely.

    Args:
        query: The user's natural language question.
        anthropic_api_key: Anthropic API key. If empty, returns [query] unchanged.
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

    if not anthropic_api_key:
        logger.debug("query_rewrite_skipped", reason="no_api_key")
        return [enriched_query]

    if not model:
        from stratakb.config import get_settings
        settings = get_settings()
        model = settings.query_rewrite_model

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

        response = await client.messages.create(
            model=model,
            max_tokens=300,
            system=_SYSTEM_PROMPT.format(
                domain_context=domain_context or _DEFAULT_DOMAIN_CONTEXT
            ),
            messages=[{"role": "user", "content": enriched_query}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        queries = json.loads(text)

        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            logger.info(
                "query_rewritten",
                original=query[:100],
                rewritten_count=len(queries),
                model=model,
            )
            return queries[:4]

    except json.JSONDecodeError:
        logger.warning("query_rewrite_json_error", query=query[:100])
    except Exception as e:
        logger.warning("query_rewrite_failed", error=str(e), query=query[:100])

    return [enriched_query]
