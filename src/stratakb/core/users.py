"""User management for StrataKB.

Handles auto-creation of users on first connection and role management.
Users are stored in strata_core.users table.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from stratakb.core.database import get_pool

logger = structlog.get_logger()


async def ensure_user(
    database_url: str,
    *,
    external_id: str,
    display_name: str | None = None,
    role: str = "viewer",
) -> dict:
    """Ensure a user exists in the database.

    Creates the user on first connection with the default role.
    Updates last_login on subsequent connections.

    Args:
        database_url: PostgreSQL connection URL.
        external_id: Unique user identifier (from env var or JWT).
        display_name: Human-readable name (optional).
        role: Default role for new users.

    Returns:
        Dict with user details.
    """
    pool = await get_pool(database_url)

    # Try to find existing user
    row = await pool.fetchrow(
        "SELECT id, external_id, display_name, email, role, created_at, last_login "
        "FROM strata_core.users WHERE external_id = $1",
        external_id,
    )

    if row:
        # Update last_login
        await pool.execute(
            "UPDATE strata_core.users SET last_login = $1 WHERE external_id = $2",
            datetime.now(timezone.utc),
            external_id,
        )
        logger.debug("user_login", user=external_id, role=row["role"])
        return {
            "id": str(row["id"]),
            "external_id": row["external_id"],
            "display_name": row["display_name"] or external_id,
            "role": row["role"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "last_login": datetime.now(timezone.utc).isoformat(),
        }
    else:
        # Create new user
        new_row = await pool.fetchrow(
            """
            INSERT INTO strata_core.users (external_id, display_name, role, last_login)
            VALUES ($1, $2, $3, $4)
            RETURNING id, external_id, display_name, role, created_at, last_login
            """,
            external_id,
            display_name or external_id,
            role,
            datetime.now(timezone.utc),
        )
        logger.info("user_created", user=external_id, role=role)
        return {
            "id": str(new_row["id"]),
            "external_id": new_row["external_id"],
            "display_name": new_row["display_name"],
            "role": new_row["role"],
            "created_at": new_row["created_at"].isoformat() if new_row["created_at"] else None,
            "last_login": new_row["last_login"].isoformat() if new_row["last_login"] else None,
        }


async def get_user(database_url: str, external_id: str) -> dict | None:
    """Get a user by external ID.

    Returns:
        User dict or None if not found.
    """
    pool = await get_pool(database_url)
    row = await pool.fetchrow(
        "SELECT id, external_id, display_name, email, role, created_at, last_login "
        "FROM strata_core.users WHERE external_id = $1",
        external_id,
    )
    if not row:
        return None

    return {
        "id": str(row["id"]),
        "external_id": row["external_id"],
        "display_name": row["display_name"],
        "email": row["email"],
        "role": row["role"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "last_login": row["last_login"].isoformat() if row["last_login"] else None,
    }


async def update_user_role(
    database_url: str,
    external_id: str,
    new_role: str,
) -> dict:
    """Update a user's role. Admin only.

    Args:
        database_url: PostgreSQL connection URL.
        external_id: User to update.
        new_role: New role (viewer, operator, admin).

    Returns:
        Updated user dict.

    Raises:
        ValueError: If role is invalid or user not found.
    """
    valid_roles = {"viewer", "operator", "admin"}
    if new_role not in valid_roles:
        raise ValueError(f"Invalid role '{new_role}'. Must be one of: {', '.join(sorted(valid_roles))}")

    pool = await get_pool(database_url)

    result = await pool.execute(
        "UPDATE strata_core.users SET role = $1 WHERE external_id = $2",
        new_role,
        external_id,
    )

    if result == "UPDATE 0":
        raise ValueError(f"User '{external_id}' not found.")

    logger.info("user_role_updated", user=external_id, new_role=new_role)

    user = await get_user(database_url, external_id)
    return user  # type: ignore[return-value]


async def list_users(database_url: str) -> list[dict]:
    """List all users.

    Returns:
        List of user dicts.
    """
    pool = await get_pool(database_url)
    rows = await pool.fetch(
        "SELECT id, external_id, display_name, email, role, created_at, last_login "
        "FROM strata_core.users ORDER BY created_at"
    )

    return [
        {
            "id": str(row["id"]),
            "external_id": row["external_id"],
            "display_name": row["display_name"],
            "email": row["email"],
            "role": row["role"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "last_login": row["last_login"].isoformat() if row["last_login"] else None,
        }
        for row in rows
    ]
