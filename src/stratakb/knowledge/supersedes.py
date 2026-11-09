"""The file a stored document version says it replaces.

A document declares it with a supersedes key in its front matter, naming one
file relative to its own folder, like a Markdown link. The value is only ever
a string: it is never opened and never logged, and whatever names nothing
usable gives None, so it cannot fail an ingest. Kept apart from documents.py,
which the retrieval code imports, because the path limit is in the ingest
policy, which needs the parser registry.
"""

from __future__ import annotations

import posixpath
import re
from typing import TYPE_CHECKING

from stratakb.knowledge.policy import MAX_RELATIVE_PATH
from stratakb.knowledge.structure import frontmatter_end, frontmatter_source

if TYPE_CHECKING:
    from stratakb.knowledge.parsers.registry import Format

# A top-level key with a value on its own line. re.ASCII: under IGNORECASE
# alone, the long s would also spell the key. The value takes its trailing
# blanks, which are stripped below (the linear-time rules in knowledge/structure.py).
_SUPERSEDES_KEY_RE = re.compile(
    r"^supersedes[ \t]*:[ \t]+(\S.*)",
    re.ASCII | re.IGNORECASE | re.MULTILINE,
)

# How an unquoted YAML value starts when it is a comment, a list, a mapping, a
# block or a reference: the key names one file, and none of these is read
_NOT_A_FILE_NAME = "#[{|>&*!"

_YAML_COMMENT_RE = re.compile(r"[ \t]#")

_DRIVE_RE = re.compile(r"[A-Za-z]:")


def frontmatter_supersedes(text: str) -> str | None:
    """The supersedes value in the text's leading front matter, or None.

    The first top-level supersedes key counts, in any letter case. A quoted
    value is what lies between the quotes; an unquoted one ends at a comment.
    No YAML parser, as for the document's date: a failure in one must not reach
    an ingest. text must have \\n line ends.
    """
    match = _SUPERSEDES_KEY_RE.search(text[:frontmatter_end(text)])
    if match is None:
        return None
    value = match[1]
    if value[0] in "\"'":
        closing = value.find(value[0], 1)
        if closing < 0 or not value[1:closing].strip():
            return None
        return value[1:closing]
    if value[0] in _NOT_A_FILE_NAME:
        return None
    return _YAML_COMMENT_RE.split(value, maxsplit=1)[0].rstrip()


def resolve_supersedes(value: str, relative_path: str) -> str | None:
    """The path below the source root that value names from relative_path, or None.

    None for a control character, an absolute path, a drive, a URL, a path that
    leaves the root, the document itself and an over-long path. Whether the
    target exists is not checked: ingest order is not fixed.
    """
    value = value.replace("\\", "/")
    if (
        any(c < " " for c in value)
        or value.startswith("/")
        or _DRIVE_RE.match(value)
        or "://" in value
    ):
        return None
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(relative_path), value))
    if (
        resolved in (".", "..", relative_path)
        or resolved.startswith("../")
        or len(resolved) > MAX_RELATIVE_PATH
    ):
        return None
    return resolved


def declared_supersedes(fmt: Format | None, data: bytes, relative_path: str) -> str | None:
    """What a file's bytes declare it replaces, resolved, or None.

    Read from the bytes because the parser strips the front matter. Only text
    formats count, and PDF or Word bytes are never decoded.
    """
    if fmt is None or fmt.binary:
        return None
    value = frontmatter_supersedes(frontmatter_source(data))
    return None if value is None else resolve_supersedes(value, relative_path)
