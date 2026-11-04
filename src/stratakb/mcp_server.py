"""StrataKB MCP server entry point.

Exposes knowledge search, layer management, learning feedback and read-only
database access as MCP tools.
Run with: python -m stratakb
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import structlog
from mcp.server.fastmcp import FastMCP

from stratakb.config import get_settings
from stratakb.core.logging import configure_logging, log_tool_call

_settings = get_settings()

# Configure structured logging (centralized)
configure_logging(_settings.log_level)

from stratakb.core.audit import write_audit_log, query_audit_log
from stratakb.core.auth import get_current_user, require_role
from stratakb.core.database import check_connection, close_pool, get_pool
from stratakb.core.migrations import ensure_ingestion_schema
from stratakb.core.users import ensure_user, list_users, update_user_role
from stratakb.database_access.tools import register_database_tools
from stratakb.knowledge.tools import register_knowledge_tools
from stratakb.learning.tools import register_learning_tools
from stratakb.rag.tools import register_rag_tools

logger = structlog.get_logger()


@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Warm up DB pool and embedding model before accepting tool calls."""

    # Pre-warm DB connection pool (fast — ~1s)
    try:
        pool = await get_pool(_settings.database_url)
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        logger.info("db_pool_warmed")
    except Exception as e:
        logger.warning("db_warmup_failed", error=str(e))

    # Bring an existing database up to the current schema (no-op when current)
    try:
        await ensure_ingestion_schema(_settings.database_url)
    except Exception as e:
        logger.warning("db_migration_failed", error=str(e))

    # Pre-load embedding model in thread (17s+ for PyTorch import on first call).
    # Must complete before yielding — otherwise the first tool call races with
    # model loading, blocks the event loop on _local_model_lock, and causes
    # MCP transport timeout.
    if _settings.embedding_backend == "local":
        try:
            await asyncio.to_thread(_preload_embedding_model_sync)
        except Exception as e:
            logger.warning("embedding_preload_failed", error=str(e))

    try:
        yield {}
    finally:
        await close_pool()
        logger.info("shutting_down")


def _preload_embedding_model_sync() -> None:
    """Load the sentence-transformers model (runs in thread to avoid blocking)."""
    from stratakb.knowledge.embeddings import _get_local_model

    logger.info("preloading_embedding_model", model=_settings.embedding_model)
    _get_local_model(_settings.embedding_model)
    logger.info("embedding_model_ready", model=_settings.embedding_model)


# Initialize MCP server with lifespan for async warmup
mcp = FastMCP(
    "StrataKB",
    instructions=(
        "StrataKB is a knowledge base of documents organised in layers, one per "
        "domain (for example a standard, a product's docs or a team's runbooks). "
        "Use strata_ask to answer questions with cited context, strata_search for "
        "exact terms and identifiers, and strata_list_knowledge_layers to see what "
        "the knowledge base covers. Admin tools create layers, ingest documents and "
        "apply layer presets."
    ),
    lifespan=_lifespan,
)

settings = _settings

# Register module tools
register_knowledge_tools(mcp, settings.database_url, settings.openai_api_key)
register_rag_tools(mcp, settings.database_url, settings.openai_api_key)
register_database_tools(
    mcp, settings.database_url, settings.readonly_database_url or settings.database_url
)
register_learning_tools(mcp, settings.database_url)


# ==========================================
# Admin Tools
# ==========================================


@mcp.tool()
@require_role("viewer")
@log_tool_call
async def strata_admin_health() -> dict:
    """Check StrataKB system health.

    Returns status of all components: PostgreSQL, pgvector extension,
    database schemas, and which optional API keys are configured.
    """
    result: dict[str, object] = {}

    # Check database
    db_status = await check_connection(settings.database_url)
    result["database"] = db_status

    # Check API keys configured
    result["anthropic_api"] = {
        "status": "configured" if settings.anthropic_api_key else "not_configured"
    }
    result["openai_api"] = {
        "status": "configured" if settings.openai_api_key else "not_configured"
    }

    # Overall status
    all_ok = db_status.get("status") == "ok"
    result["overall"] = "ok" if all_ok else "degraded"

    # Current user context
    result["user"] = {
        "user_id": settings.strata_user,
        "role": settings.strata_role,
    }

    return result


@mcp.tool()
@require_role("admin")
@log_tool_call
async def strata_admin_audit_log(
    user_id: str | None = None,
    action: str | None = None,
    tool_name: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 50,
) -> dict:
    """Query the audit log for recorded operations.

    Returns audit log entries filtered by optional criteria. Only accessible
    to admin users. Results are ordered newest first.

    Args:
        user_id: Filter by user ID.
        action: Filter by action type (e.g., "create", "delete", "ingest").
        tool_name: Filter by MCP tool name.
        from_date: ISO date string for start of range (e.g., "2026-03-01").
        to_date: ISO date string for end of range (e.g., "2026-03-31").
        limit: Maximum entries to return (default 50, max 500).
    """
    try:
        entries = await query_audit_log(
            settings.database_url,
            user_id=user_id,
            action=action,
            tool_name=tool_name,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
        )
        return {
            "status": "ok",
            "entries": entries,
            "count": len(entries),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==========================================
# User Management Tools
# ==========================================


@mcp.tool()
@require_role("viewer")
@log_tool_call
async def strata_whoami() -> dict:
    """Show current user identity and role.

    Returns the user's ID, role, and ensures they exist in the database
    (auto-creates on first call).
    """
    try:
        user = await ensure_user(
            settings.database_url,
            external_id=settings.strata_user,
            role=settings.strata_role,
        )
        return {"status": "ok", "user": user}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
@require_role("admin")
@log_tool_call
async def strata_admin_list_users() -> dict:
    """List all StrataKB users and their roles.

    Admin only. Returns all registered users with their roles and last login times.
    """
    try:
        users = await list_users(settings.database_url)
        return {"status": "ok", "users": users, "count": len(users)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
@require_role("admin")
@log_tool_call
async def strata_admin_update_role(
    user_external_id: str,
    new_role: str,
) -> dict:
    """Update a user's role.

    Admin only. Changes the role for a specified user.

    Args:
        user_external_id: The external ID of the user to update.
        new_role: The new role to assign. Must be one of: viewer, operator, admin.
    """
    try:
        user = await update_user_role(
            settings.database_url,
            user_external_id,
            new_role,
        )

        # Audit log
        admin_id, _ = get_current_user()
        await write_audit_log(
            settings.database_url,
            user_id=admin_id,
            action="update_role",
            tool_name="strata_admin_update_role",
            parameters={"target_user": user_external_id, "new_role": new_role},
            result_summary=f"Role updated to {new_role}",
        )

        return {"status": "ok", "user": user}
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==========================================
# Server lifecycle
# ==========================================


def main() -> None:
    """Run the StrataKB MCP server."""
    logger.info(
        "starting_stratakb",
        user=settings.strata_user,
        role=settings.strata_role,
    )

    # lifespan handles warmup (DB + embedding model) after transport is ready
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
