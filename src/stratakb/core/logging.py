"""Structured logging configuration for StrataKB.

Provides centralized structlog setup with:
- ISO-timestamped JSON output (ConsoleRenderer in DEBUG mode)
- Sensitive data masking in log events
- Tool invocation decorator with duration tracking
- Context variable integration for user_id, tool_name, layer
"""

from __future__ import annotations

import functools
import logging
import re
import sys
import time
from typing import Callable

import structlog


# Patterns for sensitive data that may appear in log values
_ACCOUNT_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b")  # IBAN
_CARD_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")  # Card number

# Keys whose values should always be masked
_SENSITIVE_KEYS = {
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "openai_api_key",
    "anthropic_api_key",
    "account_number",
    "card_number",
    "iban",
}


def _mask_iban(value: str) -> str:
    """Mask IBAN, keeping country code and last 4."""
    def _replace(m: re.Match) -> str:
        iban = m.group(0)
        if len(iban) > 8:
            return iban[:4] + "****" + iban[-4:]
        return "****"
    return _ACCOUNT_PATTERN.sub(_replace, value)


def _mask_card(value: str) -> str:
    """Mask card numbers, keeping last 4 digits."""
    def _replace(m: re.Match) -> str:
        digits = re.sub(r"[\s-]", "", m.group(0))
        return "****-****-****-" + digits[-4:]
    return _CARD_PATTERN.sub(_replace, value)


def _mask_value(value: object) -> object:
    """Mask sensitive patterns in a string value."""
    if not isinstance(value, str):
        return value
    value = _mask_iban(value)
    value = _mask_card(value)
    return value


def sensitive_data_filter(
    logger: object, method_name: str, event_dict: dict
) -> dict:
    """Structlog processor that masks sensitive data in log events."""
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "***MASKED***"
        else:
            event_dict[key] = _mask_value(event_dict[key])
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with StrataKB processors.

    Args:
        log_level: Python log level name (DEBUG, INFO, WARNING, ERROR).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            sensitive_data_filter,
            structlog.dev.ConsoleRenderer() if log_level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # stdout carries the MCP protocol on the stdio transport; logs go to stderr
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def log_tool_call(func: Callable) -> Callable:
    """Decorator that logs MCP tool invocation with timing and context.

    Logs: tool_name, user_id, duration_ms, layer (if present in kwargs),
    and result status.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        from stratakb.core.auth import get_current_user

        tool_name = func.__name__
        user_id, user_role = get_current_user()
        layer = kwargs.get("layer_name") or kwargs.get("name") or None

        log = structlog.get_logger()
        log_ctx = {
            "tool": tool_name,
            "user": user_id,
            "role": user_role,
        }
        if layer:
            log_ctx["layer"] = layer

        log.info("tool_call_started", **log_ctx)

        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration_ms = round((time.perf_counter() - start) * 1000, 1)

            status = "ok"
            if isinstance(result, dict):
                status = result.get("status", "ok")

            log.info(
                "tool_call_completed",
                **log_ctx,
                duration_ms=duration_ms,
                status=status,
            )
            return result
        except Exception as e:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            log.error(
                "tool_call_failed",
                **log_ctx,
                duration_ms=duration_ms,
                error=type(e).__name__,
                error_msg=str(e),
            )
            raise

    wrapper.__wrapped__ = getattr(func, "__wrapped__", func)
    return wrapper
