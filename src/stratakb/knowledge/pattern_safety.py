"""Safety check for the regular expressions a layer stores.

Routing and structure patterns run on every question and on every line of
every document ingested. One that can backtrack exponentially would stall
the server, so every write path refuses it, and every reader skips one that
reached the layer another way (an older export, a full restore, a layer
stored by an older version). The rule is written once, here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from re import _constants as _sre
from re import _parser as _sre_parser

import structlog

logger = structlog.get_logger()

NESTED_REPEAT_REASON = (
    "it repeats a repeated part, which can take very long on some lines. "
    "The check is strict and also refuses some harmless patterns, such as '(x+y)*z'."
)

_REPEATS = (_sre.MAX_REPEAT, _sre.MIN_REPEAT, _sre.POSSESSIVE_REPEAT)


def _subpatterns(value: object):
    """The parsed subpatterns inside a node's argument, at any depth."""
    if isinstance(value, _sre_parser.SubPattern):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _subpatterns(item)


def _nested_repeat(items, unbounded_outside: bool = False) -> bool:
    """True when a variable repeat of more than one sits inside an unbounded repeat.

    Such a pattern ('(a+)+', '(\\w+\\s?)*') can backtrack exponentially on a
    line it almost matches; a fixed count inside ('(\\d{3})+') cannot.
    """
    for op, av in items:
        if op in _REPEATS:
            lo, hi, body = av
            if unbounded_outside and hi > 1 and lo != hi:
                return True
            if _nested_repeat(body, unbounded_outside or hi == _sre.MAXREPEAT):
                return True
            continue
        for sub in _subpatterns(av):
            if _nested_repeat(sub, unbounded_outside):
                return True
    return False


def has_nested_repeat(pattern: str) -> bool:
    """True when pattern nests a variable repeat inside an unbounded one.

    Call it only on a pattern that compiles (parsing raises re.error otherwise).
    """
    return _nested_repeat(_sre_parser.parse(pattern))


def pattern_problem(pattern: object, *, line_starts: bool = False) -> str | None:
    """Why pattern must not run, or None when it is safe.

    With line_starts the pattern marks section starts, matched at the start
    of every line, and it must not match an empty line either.
    """
    if not isinstance(pattern, str):
        return "it is not a string."
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return str(e)
    if line_starts and compiled.match(""):
        return "it matches an empty line."
    if has_nested_repeat(pattern):
        return NESTED_REPEAT_REASON
    return None


def usable_patterns(
    patterns: Iterable[object],
    *,
    kind: str,
    layer: str | None = None,
    flags: int = 0,
    line_starts: bool = False,
) -> list[re.Pattern[str]]:
    """Compile the safe patterns in order, skipping the others with a warning."""
    usable: list[re.Pattern[str]] = []
    for pattern in patterns:
        problem = pattern_problem(pattern, line_starts=line_starts)
        if problem is not None:
            logger.warning(
                f"unsafe_{kind}_pattern",
                layer=layer,
                pattern=pattern if isinstance(pattern, str) else repr(pattern),
                reason=problem,
            )
            continue
        usable.append(re.compile(pattern, flags))
    return usable
