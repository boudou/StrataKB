"""The passages strata_ask returns: packed to a token budget, each with its citation.

A passage is one retrieved chunk. Passages keep the retrieval order and are
numbered S1, S2, ... in it, so S<n> is the n-th retrieved chunk the
learning loop records. Each passage carries a ref that names the exact span
of the stored document text it holds, and a citation a developer can open:
the file's relative path, the page, the line range and the clause path.
The budget counts each passage's id, ref and citation with its text, and an
answer holds at most MAX_PASSAGES passages, all of which the learning loop
stores, so feedback by passage id resolves for every one.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from stratakb.knowledge.documents import locate

RESPONSE_FORMATS = ("concise", "detailed")

# The most passages an answer returns, and results the learning loop stores
MAX_PASSAGES = 50
# The tokens a passage costs in the response beyond its own indent-2 JSON: the
# deeper indentation inside the response's passages list and the page, lines
# and page_end a stored document adds (measured on FastMCP's indent-2 JSON
# with six-digit spans and three-digit pages: about 81 characters)
RESPONSE_SLACK_TOKENS = 21

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


def ask_chunk(
    *,
    chunk_id: str | None,
    content: str,
    score: float,
    similarity: float | None,
    layer: str,
    source_file: str,
    section: str | None,
    doc_type: str | None,
    metadata: dict[str, Any],
    document_id: int | None,
    char_start: int | None,
    char_end: int | None,
    clause_path: str | None,
) -> dict[str, Any]:
    """The one shape of a strata_ask chunk; the reranker fills rerank_score."""
    return {
        "chunk_id": chunk_id,
        "content": content,
        "score": score,
        "similarity": similarity,
        "rerank_score": None,
        "layer": layer,
        "source_file": source_file,
        "section": section,
        "doc_type": doc_type,
        "metadata": metadata,
        "document_id": document_id,
        "char_start": char_start,
        "char_end": char_end,
        "clause_path": clause_path,
    }


def estimate_tokens(text: str) -> int:
    """A rough token count: about four characters per token."""
    return math.ceil(len(text) / 4)


def passage_cost(chunk: dict[str, Any]) -> int:
    """The tokens a chunk costs as a passage: its id, ref, text and citation as JSON."""
    passage = json.dumps(build_passages([chunk], {})[0], indent=2, ensure_ascii=False)
    return estimate_tokens(passage) + RESPONSE_SLACK_TOKENS


def pack_passages(chunks: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """The leading chunks, in order, whose passages fit the token budget.

    Each passage costs its text plus its id, ref and citation (passage_cost).
    Packing stops before the chunk that would pass the budget, and after
    MAX_PASSAGES passages; the first chunk is always kept, however long.
    """
    kept: list[dict[str, Any]] = []
    used = 0
    for chunk in chunks[:MAX_PASSAGES]:
        cost = passage_cost(chunk)
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


@dataclass(frozen=True)
class DocumentRef:
    """A span of a stored document's text: d<document_id>:<char_start>-<char_end>."""

    document_id: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ChunkRef:
    """A stored chunk named by its id: c<chunk_id>."""

    chunk_id: str


_DOCUMENT_REF = re.compile(r"d(\d{1,18}):(\d{1,9})-(\d{1,9})")
_CHUNK_REF = re.compile(r"c([0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12})")


def parse_ref(ref: str) -> DocumentRef | ChunkRef | None:
    """The span or chunk a ref written by passage_ref names; None when malformed."""
    if m := _DOCUMENT_REF.fullmatch(ref):
        return DocumentRef(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if m := _CHUNK_REF.fullmatch(ref):
        return ChunkRef(m.group(1).lower())
    return None


def _clause_path(chunk: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    """The clause or heading chain the chunk sits under, joined with " > ".

    The structure map's path (chunks.clause_path) wins; chunks stored before
    it fall back to the parser's section path in the metadata.
    """
    path = chunk.get("clause_path") or metadata.get("clause_path") or metadata.get("section_path")
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
        "clause_path": _clause_path(chunk, metadata),
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
