"""Audit logging for MCP tool operations.

Searches, ingestion and admin actions are recorded in the
strata_core.audit_log table with user context and parameter redaction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

from stratakb.core.database import get_pool

logger = structlog.get_logger()

# Fields to redact from parameters before logging
SENSITIVE_FIELDS = {
    "api_key",
    "password",
    "secret",
    "token",
    "openai_api_key",
    "anthropic_api_key",
    "authorization",
}


def redact_params(params: dict | None) -> dict | None:
    """Redact sensitive fields from parameters before audit logging."""
    if not params:
        return params

    redacted = {}
    for key, value in params.items():
        if key.lower() in SENSITIVE_FIELDS:
            redacted[key] = "***REDACTED***"
        elif isinstance(value, dict):
            redacted[key] = redact_params(value)
        else:
            redacted[key] = value
    return redacted


async def write_audit_log(
    database_url: str,
    *,
    action: str,
    tool_name: str,
    user_id: str | None = None,
    parameters: dict | None = None,
    result_summary: str | None = None,
) -> None:
    """Write an entry to the audit log.

    Args:
        database_url: PostgreSQL connection URL.
        action: Action type (e.g., "create", "delete", "query", "ingest").
        tool_name: Name of the MCP tool invoked.
        user_id: ID of the user performing the action; defaults to the
            current user (STRATA_USER).
        parameters: Tool parameters (sensitive fields auto-redacted).
        result_summary: Brief description of the result.
    """
    if user_id is None:
        # Resolved at call time and outside the try, so a failure is not
        # swallowed as audit_log_failed. Imported here so tests that patch
        # stratakb.core.auth.get_current_user take effect.
        from stratakb.core.auth import get_current_user

        user_id, _ = get_current_user()

    safe_params = redact_params(parameters)

    try:
        pool = await get_pool(database_url)
        await pool.execute(
            """
            INSERT INTO strata_core.audit_log
                (user_id, action, tool_name, parameters, result_summary)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            """,
            user_id,
            action,
            tool_name,
            json.dumps(safe_params) if safe_params else None,
            result_summary,
        )
        logger.debug(
            "audit_logged",
            user=user_id,
            action=action,
            tool=tool_name,
        )
    except Exception as e:
        # Audit logging should never break the main operation
        logger.error("audit_log_failed", error=str(e), tool=tool_name)


async def query_audit_log(
    database_url: str,
    *,
    user_id: str | None = None,
    action: str | None = None,
    tool_name: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Query audit log entries with optional filters.

    Args:
        database_url: PostgreSQL connection URL.
        user_id: Filter by user.
        action: Filter by action type.
        tool_name: Filter by tool name.
        from_date: ISO date string for start of range.
        to_date: ISO date string for end of range.
        limit: Maximum entries to return (default 50, max 500).

    Returns:
        List of audit log entries, newest first.
    """
    limit = min(max(limit, 1), 500)

    conditions = []
    params = []
    idx = 1

    if user_id:
        conditions.append(f"user_id = ${idx}")
        params.append(user_id)
        idx += 1

    if action:
        conditions.append(f"action = ${idx}")
        params.append(action)
        idx += 1

    if tool_name:
        conditions.append(f"tool_name = ${idx}")
        params.append(tool_name)
        idx += 1

    if from_date:
        conditions.append(f"created_at >= ${idx}")
        params.append(datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc))
        idx += 1

    if to_date:
        conditions.append(f"created_at <= ${idx}")
        params.append(datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc))
        idx += 1

    where = " AND ".join(conditions) if conditions else "TRUE"

    params.append(limit)
    query = f"""
        SELECT id, user_id, action, tool_name, parameters,
               result_summary, created_at
        FROM strata_core.audit_log
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
    """

    pool = await get_pool(database_url)
    rows = await pool.fetch(query, *params)

    result = []
    for row in rows:
        entry = dict(row)
        entry["created_at"] = entry["created_at"].isoformat() if entry["created_at"] else None
        result.append(entry)

    return result
