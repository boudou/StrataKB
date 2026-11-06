"""The README opens with a Quick start in plain words; the reference below it has a Glossary."""

import re
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"

BANNED = re.compile(
    r"\b(chunks?|chunking|embeddings?|vectors?|pgvector|hnsw|rrf|rerank|reranker|reranking"
    r"|cross-encoder|top_k|full-text|hybrid|confidence|retrieval|retrieves?|semantic|tokens?)\b",
    re.IGNORECASE,
)

GLOSSARY_TERMS = [
    "Chunk",
    "Confidence",
    "Embedding",
    "HNSW",
    "Hybrid search",
    "interaction_id",
    "Layer",
    "Query rewriting",
    "Reranking",
    "Routing patterns",
    "RRF",
    "top_k",
]

EXTERNAL_ANCHORS = ["quick-start", "layers-and-presets", "configuration", "roadmap", "tools"]


def _lines() -> list[str]:
    return README.read_text(encoding="utf-8").splitlines()


def _headings(lines: list[str]) -> list[tuple[int, str]]:
    """(index, line) for every Markdown heading outside fenced code blocks."""
    found = []
    in_fence = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and re.match(r"#{1,6} ", line):
            found.append((i, line))
    return found


def _slug(title: str) -> str:
    """GitHub's heading anchor."""
    kept = re.sub(r"[^\w\- ]", "", title.strip().lower())
    return kept.replace(" ", "-")


def _section(lines: list[str], title: str) -> tuple[int, int]:
    """Start and end line index of a `## ` section."""
    tops = [i for i, line in _headings(lines) if line.startswith("## ")]
    matches = [i for i in tops if lines[i] == f"## {title}"]
    assert matches, f"README.md has no '## {title}' section"
    start = matches[0]
    end = next((i for i in tops if i > start), len(lines))
    return start, end


def _mermaid_block(lines: list[str], start: int, end: int) -> list[str] | None:
    """Lines of the first Mermaid block between start and end, or None."""
    for i in range(start, end):
        if lines[i].strip() == "```mermaid":
            close = next(
                (j for j in range(i + 1, end) if lines[j].startswith("```")), end
            )
            return lines[i + 1 : close]
    return None


def test_quick_start_is_first_section():
    lines = _lines()
    tops = [line for _, line in _headings(lines) if line.startswith("## ")]
    assert tops[0] == "## Quick start"
    start, end = _section(lines, "Quick start")
    steps = [
        line[4:] for i, line in _headings(lines) if start < i < end and line.startswith("### ")
    ]
    assert steps == ["1. Install", "2. Add a folder", "3. Connect your client", "4. Ask"]


def test_quick_start_has_the_diagram():
    lines = _lines()
    start, end = _section(lines, "Quick start")
    block = _mermaid_block(lines, start, end)
    assert block is not None, "the Quick start has no Mermaid diagram"
    nodes = re.findall(r'\w+\["([^"]+)"\]', "\n".join(block))
    assert nodes == ["Install", "Start the database", "Add a folder", "Connect your client", "Ask"]


def test_top_has_no_retrieval_terms():
    lines = _lines()
    _, end = _section(lines, "Quick start")
    matches = [
        f"line {n + 1}: {m.group(0)}"
        for n, line in enumerate(lines[:end])
        for m in BANNED.finditer(line)
    ]
    assert not matches, "retrieval terms in the Quick start: " + "; ".join(matches)


def test_glossary_has_the_terms():
    lines = _lines()
    quick, _ = _section(lines, "Quick start")
    start, end = _section(lines, "Glossary")
    assert start > quick
    body = "\n".join(lines[start:end])
    missing = [term for term in GLOSSARY_TERMS if f"**{term}**" not in body]
    assert not missing, f"missing from the Glossary: {missing}"


def test_in_page_links_resolve():
    lines = _lines()
    anchors = {_slug(re.sub(r"^#+ ", "", line)) for _, line in _headings(lines)}
    targets = set(re.findall(r"\]\(#([^)]+)\)", "\n".join(lines)))
    assert targets - anchors == set()
    assert set(EXTERNAL_ANCHORS) <= anchors
