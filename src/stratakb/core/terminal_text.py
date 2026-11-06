"""Text safe for a terminal: one rule for what counts as unprintable.

Document text, file names, model ids and the caller's question and draft can
carry ANSI or OSC sequences, a carriage return, a backspace, a one-byte CSI
(U+009B) or a bidi override that would hide or rewrite what is shown. A
character is unprintable when str.isprintable() says so; newlines and tabs
are layout and stay.
"""

from __future__ import annotations

import json
from typing import Any

_LAYOUT = "\n\t"


def unprintable(char: str) -> bool:
    """True for a character that must not reach a terminal as it is."""
    return not char.isprintable() and char not in _LAYOUT


def visible(value: Any) -> str:
    """The text with every unprintable character shown as an escape (\\x1b, \\u202e)."""
    out = []
    for char in str(value):
        if not unprintable(char):
            out.append(char)
        elif ord(char) <= 0xFF:
            out.append(f"\\x{ord(char):02x}")
        elif ord(char) <= 0xFFFF:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(f"\\U{ord(char):08x}")
    return "".join(out)


def printable_only(text: str) -> str:
    """The text with every unprintable character dropped (newlines and tabs too)."""
    return "".join(c for c in text if c.isprintable())


def _json_escape(char: str) -> str:
    """The JSON escape of one character; a surrogate pair above U+FFFF."""
    code = ord(char)
    if code <= 0xFFFF:
        return f"\\u{code:04x}"
    code -= 0x10000
    return f"\\u{0xD800 + (code >> 10):04x}\\u{0xDC00 + (code & 0x3FF):04x}"


def terminal_safe_json(obj: Any, **kwargs: Any) -> str:
    """json.dumps with non-ASCII text kept and every unprintable character escaped.

    json.dumps escapes only U+0000-U+001F; with ensure_ascii=False it writes
    DEL, the C1 controls and bidi format characters as they are. Here they
    become \\uXXXX escapes, so the output stays valid JSON and decodes to the
    same value. Outside strings json.dumps writes only spaces and, with an
    indent, newlines, so only string contents are escaped.
    """
    text = json.dumps(obj, ensure_ascii=False, **kwargs)
    return "".join(_json_escape(c) if unprintable(c) else c for c in text)
