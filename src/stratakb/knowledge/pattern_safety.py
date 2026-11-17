"""Safety check for the regular expressions a layer stores.

Routing and structure patterns run on every question and on every line of
every document ingested. One that can backtrack exponentially would stall
the server, one too long slows every match, one with a lone surrogate or a
NUL cannot be stored, and one Python warns it will read differently may
change meaning later. Every write path refuses them, and every reader skips
one that reached the layer another way (an older export, a full restore, a
layer stored by an older version). The rule is written once, here.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from re import _constants as _sre
from re import _parser as _sre_parser

import structlog

logger = structlog.get_logger()

# Characters one routing, structure or archived files pattern may hold
MAX_PATTERN_LENGTH = 500

FUTURE_WARNING_PROBLEM = (
    "it uses [ or a doubled -, &, ~ or | inside [...], which a later Python reads differently"
)

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


def pattern_text_problem(pattern: str) -> str | None:
    """Why pattern's text cannot be stored or is too long to match, or None.

    jsonb refuses a lone surrogate and a NUL (JSON writes both as escapes);
    a very long pattern would be matched against every path or question.
    """
    if len(pattern) > MAX_PATTERN_LENGTH:
        return f"it is longer than {MAX_PATTERN_LENGTH} characters."
    if any(c == "\0" or "\ud800" <= c <= "\udfff" for c in pattern):
        return "it holds a lone surrogate or a NUL, which the database cannot store."
    return None


def shown_pattern(pattern: object) -> str:
    """The pattern for a message: quoted as it is, or, when it is not a string
    or its text has a problem, its first 40 characters escaped (a lone
    surrogate cannot be encoded as UTF-8 in a tool reply or on stdout)."""
    if isinstance(pattern, str) and pattern_text_problem(pattern) is None:
        return f"'{pattern}'"
    text = pattern if isinstance(pattern, str) else repr(pattern)
    return repr(text[:40]) + ("..." if len(text) > 40 else "")


def future_warning(regex: str) -> bool:
    """True when Python warns that a later version reads regex differently.

    Parsed, not compiled: re caches compiled patterns, and a cached one warns
    no more. Only a FutureWarning counts, so a stray warning does not refuse
    a good pattern.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        try:
            _sre_parser.parse(regex)
        except re.error:
            return False
    return any(issubclass(w.category, FutureWarning) for w in caught)


def pattern_problem(pattern: object, *, line_starts: bool = False) -> str | None:
    """Why pattern must not run, or None when it is safe.

    The full check for a stored routing or structure pattern: its type, its
    text (length, NUL, lone surrogate), a Python FutureWarning, then
    regex_problem. With line_starts the pattern marks section starts,
    matched at the start of every line, and it must not match an empty line
    either.
    """
    if not isinstance(pattern, str):
        return "it is not a string."
    # First, so a very long pattern is never parsed or compiled
    problem = pattern_text_problem(pattern)
    if problem is not None:
        return problem
    # Before compiling, so a warning pattern never warns at read time
    if future_warning(pattern):
        return f"{FUTURE_WARNING_PROBLEM}; escape it with a backslash."
    return regex_problem(pattern, line_starts=line_starts)


def regex_problem(pattern: str, *, line_starts: bool = False) -> str | None:
    """Why the regular expression pattern must not run, or None.

    Only the regex checks: it does not compile, it matches an empty line
    (with line_starts) or it repeats a repeated part.
    """
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
