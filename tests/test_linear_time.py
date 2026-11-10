"""Every module-level pattern that reads document text runs in linear time.

A line of a document can hold 100,000 blanks or brackets, and the re module
holds the GIL, so one quadratic pattern stalls the whole server. This test
collects the compiled patterns of the modules that read document text and
times each on hostile lines, so a pattern added later is covered without a new
test. The rules for writing such patterns are in the comment at the top of
stratakb/knowledge/structure.py.
"""

from __future__ import annotations

import importlib
import re
import string
import time
from collections.abc import Callable, Iterator

import pytest

from stratakb.rag import extractive

# The modules under src/ with module-level patterns that run on document text
MODULES = (
    "stratakb.knowledge.structure",
    "stratakb.knowledge.supersedes",
    "stratakb.knowledge.document_dates",
    "stratakb.knowledge.parsers.markdown_parser",
    "stratakb.knowledge.parsers.docx_parser",
    "stratakb.knowledge.add_folder",
    "stratakb.rag.extractive",
    # strata_verify runs these on the stored passage text as well as the draft
    "stratakb.rag.verify",
    "stratakb.rag.verify_tools",
)

N = 100_000
# Seconds for one call: far above the milliseconds a linear pattern takes, far
# below the many seconds a quadratic one takes on N characters
BOUND = 1.0

# Patterns the code calls only on a line start or a bounded window, probed the
# same way (rule 3 of the linear-time rules)
CALLED_AS: dict[str, Callable[[re.Pattern[str], str], object]] = {
    "stratakb.rag.extractive._HEADING": lambda pattern, text: pattern.match(text),
    "stratakb.rag.extractive._RULE": lambda pattern, text: pattern.match(text),
    "stratakb.rag.extractive._BLOCK_START": lambda pattern, text: pattern.match(text),
    "stratakb.rag.extractive._TOKEN_BEFORE_DOT": lambda pattern, text: pattern.search(
        text, max(0, len(text) - extractive._TOKEN_REACH)
    ),
    # Called only right after a sentence end, with .match(text, pos)
    "stratakb.rag.verify._TRAILING_CITATIONS": lambda pattern, text: pattern.match(text),
}


def _collect() -> list[tuple[str, re.Pattern[str]]]:
    """Each module's patterns and those in its tuples and lists, once each.

    One level further down too, for lists of (kind, pattern) pairs.
    """
    found: list[tuple[str, re.Pattern[str]]] = []
    seen: set[int] = set()
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if isinstance(value, tuple | list):
                items = []
                for i, item in enumerate(value):
                    label = f"{module_name}.{name}[{i}]"
                    if isinstance(item, tuple):
                        items.extend((label, inner) for inner in item)
                    else:
                        items.append((label, item))
            else:
                items = [(f"{module_name}.{name}", value)]
            for label, item in items:
                if isinstance(item, re.Pattern) and id(item) not in seen:
                    seen.add(id(item))
                    found.append((label, item))
    return found


PATTERNS = _collect()


def _literal_punctuation(pattern: re.Pattern[str]) -> set[str]:
    """The punctuation the pattern looks for in the text, not its regex syntax.

    Read from the parse tree (re._parser, Python 3.11+): single characters and
    the ranges of character classes.
    """
    found: set[int] = set()

    def walk(node: object) -> None:
        if isinstance(node, re._parser.SubPattern):
            node = node.data
        if not isinstance(node, list | tuple):
            return
        if len(node) == 2 and node[0] in (re._constants.LITERAL, re._constants.NOT_LITERAL):
            found.add(node[1])
        elif len(node) == 2 and node[0] is re._constants.RANGE:
            found.update(range(node[1][0], node[1][1] + 1))
        else:
            for item in node:
                walk(item)

    walk(re._parser.parse(pattern.pattern, pattern.flags))
    return {char for char in string.punctuation if ord(char) in found}


def _hostile_lines(pattern: re.Pattern[str]) -> Iterator[str]:
    """Long runs of blanks, of blanks after a key, and of the pattern's own delimiters."""
    blanks = " " * N
    yield f"a{blanks}b"
    yield "a" + "\t" * N + "b"
    punctuation = _literal_punctuation(pattern)
    words = {"key"}
    if ":" in punctuation:
        # A pattern that reads 'key: value' meets its own keys in front of the colon
        words |= {word.lower() for word in re.findall(r"[A-Za-z]{2,}", pattern.pattern)}
    for word in sorted(words):
        yield f"{word}:{blanks}b"
        yield f"{word}: a{blanks}b"
        yield f"{word}: 'a'{blanks}b"
        yield f"{word}{blanks}:"
    for char in sorted(punctuation):
        yield char * N
        yield char * N + "x"
        yield f"a{char * N}b"
        yield (char + " ") * (N // 2)
        yield (char + "a") * (N // 2)


def _probes(pattern: re.Pattern[str]) -> Iterator[str]:
    """Each hostile line as it is and, for a pattern anchored to a line, inside front matter."""
    anchored = "^" in pattern.pattern or "\\A" in pattern.pattern or pattern.flags & re.MULTILINE
    for line in _hostile_lines(pattern):
        yield line
        if anchored:
            yield f"---\n{line}\n---\n"


@pytest.mark.parametrize(("label", "pattern"), PATTERNS, ids=[label for label, _ in PATTERNS])
def test_pattern_reads_hostile_lines_in_linear_time(label: str, pattern: re.Pattern[str]) -> None:
    call = CALLED_AS.get(label, lambda compiled, text: compiled.search(text))
    for text in _probes(pattern):
        began = time.perf_counter()
        call(pattern, text)
        elapsed = time.perf_counter() - began
        # Stop at the first slow probe: a quadratic pattern can take 40 s on one
        assert elapsed < BOUND, (
            f"{label} took {elapsed:.1f} s on {text[:20]!r}...: see the linear-time "
            "rules at the top of stratakb/knowledge/structure.py"
        )


def test_the_named_patterns_are_collected() -> None:
    labels = {label for label, _ in PATTERNS}
    named = {
        "stratakb.knowledge.structure._FRONTMATTER_RE",
        "stratakb.knowledge.structure._YAML_KEY_RE",
        "stratakb.knowledge.document_dates._DATE_KEY_RE",
        "stratakb.knowledge.supersedes._SUPERSEDES_KEY_RE",
        "stratakb.knowledge.add_folder._LINK",
        "stratakb.rag.verify._QUANTITY",
        "stratakb.rag.verify._DATES[0]",
        "stratakb.rag.verify_tools._LEGACY_PREFIX",
    }
    assert named | set(CALLED_AS) <= labels
