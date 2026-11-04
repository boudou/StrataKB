"""Role-based access control for MCP tools.

Provides a `require_role` decorator that enforces minimum role level
before allowing tool execution. User identity comes from environment
configuration (stdio transport) or JWT (SSE transport, future).

Role hierarchy: viewer < operator < admin
"""

from __future__ import annotations

import functools
from typing import Callable

import structlog

logger = structlog.get_logger()

# Role hierarchy — higher index = more privilege
ROLE_LEVELS = {
    "viewer": 0,
    "operator": 1,
    "admin": 2,
}


class PermissionDeniedError(Exception):
    """User does not have the required role."""

    def __init__(self, user_id: str, user_role: str, required_role: str, tool_name: str):
        self.user_id = user_id
        self.user_role = user_role
        self.required_role = required_role
        self.tool_name = tool_name
        super().__init__(
            f"Permission denied: user '{user_id}' has role '{user_role}', "
            f"but tool '{tool_name}' requires '{required_role}'"
        )


def get_current_user() -> tuple[str, str]:
    """Get current user identity from settings.

    Returns:
        Tuple of (user_id, role).
    """
    from stratakb.config import get_settings

    settings = get_settings()
    return settings.strata_user, settings.strata_role


async def ensure_current_user_exists(database_url: str) -> dict:
    """Ensure the current user exists in the database.

    Auto-creates the user on first connection using env var identity.
    Called during server startup or first tool invocation.

    Returns:
        User dict from strata_core.users.
    """
    from stratakb.core.users import ensure_user

    user_id, role = get_current_user()
    return await ensure_user(
        database_url,
        external_id=user_id,
        role=role,
    )


def check_role(user_role: str, required_role: str) -> bool:
    """Check if user_role meets or exceeds required_role."""
    user_level = ROLE_LEVELS.get(user_role, -1)
    required_level = ROLE_LEVELS.get(required_role, 999)
    return user_level >= required_level


def require_role(role: str) -> Callable:
    """Decorator that enforces a minimum role for an MCP tool function.

    Usage:
        @require_role("admin")
        async def my_admin_tool(...):
            ...

    Raises PermissionDeniedError if the current user's role is insufficient.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id, user_role = get_current_user()
            tool_name = func.__name__

            if not check_role(user_role, role):
                logger.warning(
                    "permission_denied",
                    user=user_id,
                    role=user_role,
                    required=role,
                    tool=tool_name,
                )
                raise PermissionDeniedError(user_id, user_role, role, tool_name)

            return await func(*args, **kwargs)

        # Preserve the original function's attributes for FastMCP
        wrapper.__wrapped__ = func
        return wrapper

    return decorator
