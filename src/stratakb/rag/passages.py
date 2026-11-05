"""The passages strata_ask returns: packed to a token budget, each with its citation.

A passage is one retrieved chunk. Passages keep the retrieval order and are
numbered S1, S2, ... in it, so S<n> is the n-th retrieved chunk the
learning loop records. Each passage carries a ref that names the exact span
of the stored document text it holds, and a citation a developer can open:
the file's relative path, the page, the line range and the clause path.
"""

from __future__ import annotations

import math
from typing import Any

from stratakb.knowledge.documents import locate

RESPONSE_FORMATS = ("concise", "detailed")

_NEXT_STEPS = {
    "not_found": (
        "Nothing in the knowledge base answers this. If no layer covers the topic, "
        "add a folder with `stratakb add`; for an exact identifier or term, try strata_search."
    ),
    "partly_found": (
        "The passages answer this only in part. Say what they leave open, or ask "
        "again with the knowledge base's own terms or a named layer."
    ),
}


def estimate_tokens(text: str) -> int:
    """A rough token count: about four characters per token."""
    return math.ceil(len(text) / 4)


def pack_passages(chunks: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """The leading chunks, in order, whose text fits the token budget.

    Packing stops before the chunk that would pass the budget; the first
    chunk is always kept, however long.
    """
    kept: list[dict[str, Any]] = []
    used = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk["content"])
        if kept and used + cost > budget:
            break
        kept.append(chunk)
        used += cost
    return kept


def passage_ref(chunk: dict[str, Any]) -> str:
    """d<document_id>:<char_start>-<char_end>, or c<chunk_id> for a chunk without offsets."""
    if chunk.get("document_id") is not None and chunk.get("char_start") is not None:
        return f"d{chunk['document_id']}:{chunk['char_start']}-{chunk['char_end']}"
    return f"c{chunk['chunk_id']}"


def _clause_path(metadata: dict[str, Any]) -> str | None:
    """The clause or heading chain the chunk sits under, joined with " > "."""
    path = metadata.get("clause_path") or metadata.get("section_path")
    if isinstance(path, list):
        path = " > ".join(str(part) for part in path if part)
    return path or None


def cite(chunk: dict[str, Any], document: dict[str, Any] | None) -> dict[str, Any]:
    """Where a passage comes from: file, page, lines, clause path and layer.

    With its stored document the file is the relative path and the page and
    line range come from the chunk's span. A chunk ingested before documents
    were stored is cited by its file name and, for a PDF, its page; its
    lines are unknown.
    """
    metadata = chunk.get("metadata") or {}
    citation: dict[str, Any] = {
        "file": chunk.get("source_file") or None,
        "page": None,
        "lines": None,
        "clause_path": _clause_path(metadata),
        "layer": chunk.get("layer"),
    }
    if document is not None and chunk.get("char_start") is not None:
        where = locate(
            document["text"], document["page_map"], chunk["char_start"], chunk["char_end"]
        )
        citation["file"] = document["relative_path"]
        citation["page"] = where["page"]
        if where["page_end"] != where["page"]:
            citation["page_end"] = where["page_end"]
        start, end = where["line_start"], where["line_end"]
        citation["lines"] = f"{start}-{end}" if end != start else str(start)
    elif chunk.get("doc_type") == "pdf":
        citation["page"] = metadata.get("page")
    return citation


def build_passages(
    chunks: list[dict[str, Any]], documents: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One passage per chunk, in order: id, ref, the chunk's text and its citation."""
    return [
        {
            "id": f"S{n}",
            "ref": passage_ref(chunk),
            "text": chunk["content"],
            "citation": cite(chunk, documents.get(chunk.get("document_id"))),
        }
        for n, chunk in enumerate(chunks, start=1)
    ]


def next_step(status: str) -> str | None:
    """What to do when the answer was not, or only partly, found."""
    return _NEXT_STEPS.get(status)


def shape_ask_response(full: dict[str, Any], response_format: str) -> dict[str, Any]:
    """The strata_ask response in the requested format.

    detailed returns everything, scores and chunk metadata included.
    concise keeps the answer status, the mode, the passages, the next step
    when there is one, and the interaction_id for strata_feedback. An error
    is returned as it is.
    """
    if full.get("status") != "ok" or response_format == "detailed":
        return full
    shaped: dict[str, Any] = {
        "status": full["answer_status"],
        "mode": full["mode"],
        "passages": full["passages"],
    }
    if "next_step" in full:
        shaped["next_step"] = full["next_step"]
    if "interaction_id" in full:
        shaped["interaction_id"] = full["interaction_id"]
    return shaped
