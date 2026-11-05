"""StrataKB MCP server entry point.

Exposes knowledge search, layer management, learning feedback and read-only
database access as MCP tools.
Run with: python -m stratakb
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

import structlog
from mcp.server.fastmcp import FastMCP

from stratakb.config import Settings, config_file_missing, dotenv_warning, get_settings
from stratakb.core.audit import query_audit_log, write_audit_log
from stratakb.core.auth import get_current_user, require_role
from stratakb.core.database import check_connection, close_pool, get_pool
from stratakb.core.egress import enforce_local_only, outbound_providers
from stratakb.core.local_db import START_HINT
from stratakb.core.logging import configure_logging, log_tool_call
from stratakb.core.migrations import ensure_schema
from stratakb.core.users import ensure_user, list_users, update_user_role
from stratakb.database_access.tools import (
    register_database_tools,
    secure_sql_tools,
    sql_tools_status,
)
from stratakb.knowledge.embedding_checks import (
    embedding_checks_degraded,
    embedding_checks_status,
    run_embedding_checks,
)
from stratakb.knowledge.tools import register_knowledge_tools
from stratakb.learning.tools import register_learning_tools
from stratakb.rag.tools import register_rag_tools

logger = structlog.get_logger()


@asynccontextmanager
async def _lifespan(server: FastMCP, settings: Settings):
    """Warm up the DB pool, then answer the client while the model loads.

    The MCP SDK reads no message until this yields, so only the fast database
    steps are awaited here. A config file that is not read is logged first.
    """

    # Neither warning carries the path: paths hold the user name
    if hint := dotenv_warning():
        logger.warning("dotenv_not_read", hint=hint)
    if config_file_missing():
        logger.warning("config_file_not_found", setting="STRATAKB_CONFIG")

    # Pre-warm DB connection pool (fast, ~1s; ~2s to fail when refused)
    db_ready = False
    try:
        pool = await get_pool(settings.database_url)
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        logger.info("db_pool_warmed")
    except Exception as e:
        logger.warning("db_warmup_failed", error=str(e), hint=START_HINT)
    else:
        db_ready = True
        # Bring an existing database up to the current schema (no-op when
        # current). Skipped when the database is unreachable: ingestion runs
        # it on first use anyway.
        try:
            await ensure_schema(settings.database_url)
        except Exception as e:
            logger.warning("db_migration_failed", error=str(e))

    # Drop the SQL tools unless the read-only role is safe (fast, at most 5s).
    # Runs before the first message, so tools/list never shows them otherwise.
    await secure_sql_tools(server, settings.readonly_database_url)

    # Load the embedding model in the background (17s+ for the PyTorch import
    # on a cold start). A search that arrives first waits for it in its worker
    # thread on _local_model_lock, never on the event loop. A daemon thread
    # does not hold the process open on shutdown.
    if settings.embedding_backend == "local":
        threading.Thread(
            target=_preload_embedding_model_sync,
            args=(settings.embedding_model,),
            name="embedding-preload",
            daemon=True,
        ).start()

    # Check the embedding model and the vector column in the background, once
    # per start; strata_admin_health reports the result. Held in a local so
    # the task is not garbage-collected mid-run.
    checks = asyncio.create_task(run_embedding_checks(settings, db_ready))

    try:
        yield {}
    finally:
        # Cancel before the pool closes: a column read may still be in flight
        checks.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await checks
        await close_pool()
        logger.info("shutting_down")


def _preload_embedding_model_sync(model_name: str) -> None:
    """Load the sentence-transformers model (runs in a background thread)."""
    try:
        from stratakb.knowledge.embeddings import _get_local_model

        logger.info("preloading_embedding_model", model=model_name)
        _get_local_model(model_name)
        logger.info("embedding_model_ready", model=model_name)
    except Exception as e:
        logger.warning("embedding_preload_failed", error=str(e))


_INSTRUCTIONS = (
    "StrataKB is a knowledge base of documents organised in layers, one per "
    "domain (for example a standard, a product's docs or a team's runbooks). "
    "Use strata_ask to answer questions with cited context, strata_search for "
    "exact terms and identifiers, and strata_list_knowledge_layers to see what "
    "the knowledge base covers. Admin tools create layers, ingest documents and "
    "apply layer presets."
)


def build_server(settings: Settings) -> FastMCP:
    """Build a StrataKB server whose tools and lifespan use `settings`."""
    mcp = FastMCP(
        "StrataKB",
        instructions=_INSTRUCTIONS,
        lifespan=functools.partial(_lifespan, settings=settings),
    )

    # Register module tools
    register_knowledge_tools(mcp, settings.database_url, settings.openai_api_key)
    register_rag_tools(mcp, settings.database_url, settings.openai_api_key)
    register_database_tools(mcp, settings.database_url, settings.readonly_database_url)
    register_learning_tools(mcp, settings.database_url)
    _register_admin_tools(mcp, settings)
    return mcp


def _register_admin_tools(mcp: FastMCP, settings: Settings) -> None:
    """Register the admin and user management tools on `mcp`."""

    # ==========================================
    # Admin Tools
    # ==========================================

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_admin_health() -> dict:
        """Check StrataKB system health.

        Returns status of all components: PostgreSQL, pgvector extension,
        database schemas, which optional API keys are configured, whether
        STRATA_LOCAL_ONLY is on, and the providers that receive content from
        this machine (name, purpose, what it sends, destination host, setting),
        whether the SQL tools (strata_db_query, strata_db_explain) are
        enabled or, when disabled, why, and the embedding checks run once at
        startup: whether the server in EMBEDDING_BASE_URL lists the embedding
        model, and whether the vector column's size matches EMBEDDING_DIMENSIONS.
        A missing model or a size mismatch turns overall to degraded; restart
        the server after fixing either.
        """
        result: dict[str, object] = {}
        current = get_settings()

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

        # Embedding model and vector column, checked once at startup
        result["embedding"] = embedding_checks_status()

        # Overall status
        all_ok = db_status.get("status") == "ok" and not embedding_checks_degraded()
        result["overall"] = "ok" if all_ok else "degraded"

        # Where data goes (host names and setting names only, never values)
        result["local_only"] = current.strata_local_only
        result["outbound"] = [asdict(p) for p in outbound_providers(current)]

        # SQL tools: on only with a safe READONLY_DATABASE_URL role
        result["sql_tools"] = sql_tools_status()

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


def main(settings: Settings | None = None) -> None:
    """Run the StrataKB MCP server, built from `settings` (default: the environment)."""
    if settings is None:
        settings = get_settings()

    # Configure structured logging (centralized)
    configure_logging(settings.log_level)

    # Before anything imports huggingface_hub, which reads its offline flags once
    enforce_local_only(settings)

    logger.info(
        "starting_stratakb",
        user=settings.strata_user,
        role=settings.strata_role,
    )

    # lifespan warms up the DB, then starts the embedding model load in the
    # background so the client gets its first answer at once
    build_server(settings).run(transport="stdio")


def __getattr__(name: str) -> FastMCP:
    """Return a new server built from the current settings for `mcp`.

    Builds a new server on every access and never caches it: a cached server
    would fix the database URL for the whole process again. Exists only until
    the tests build their own servers with build_server.
    """
    if name == "mcp":
        return build_server(get_settings())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
