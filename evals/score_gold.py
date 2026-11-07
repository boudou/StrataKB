"""Score gold.jsonl on one layer through strata_ask. argv: gold.jsonl, layer (peps_small|peps_full), out.json.

On peps_small: small-tier questions are answerable; full-tier questions listed in
cross_refusals.json and all refusals must be refused.
On peps_full: all 90 are answerable; refusals must be refused.
"""
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

from stratakb.config import get_settings
from stratakb.core.logging import configure_logging
from stratakb.mcp_server import build_server

configure_logging("ERROR")
gold_path, layer, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
gold = [json.loads(x) for x in gold_path.read_text(encoding="utf-8").splitlines()]
cross = set(json.loads(gold_path.with_name("cross_refusals.json").read_text(encoding="utf-8")))


def payload(result):
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return result
    return json.loads(result[0].text)


def expected(row):
    if row["kind"] == "refusal":
        return "refuse", row["group"]
    if layer == "peps_full" or row["tier"] == "small":
        return "answer", row["kind"]
    return ("refuse", "other_layer") if row["id"] in cross else (None, None)


def rank(row, passages):
    for i, p in enumerate(passages, 1):
        text = re.sub(r"\s+", " ", p.get("text") or "")
        if all(re.search(rx, text, re.I) for rx in row["evidence"]):
            return i
    return None


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def summary(rows):
    out_ = {}
    ans = [r for r in rows if r["want"] == "answer"]
    for name, group in [("all", ans), ("random_sample", [r for r in ans if r["sampled"]])] + [(k, [r for r in ans if r["group"] == k]) for k in sorted({r["group"] for r in ans})]:
        n = len(group)
        out_[f"answer:{name}"] = {
            "n": n,
            "r@1": pct(sum(r["rank"] == 1 for r in group), n),
            "r@3": pct(sum((r["rank"] or 99) <= 3 for r in group), n),
            "any": pct(sum(r["rank"] is not None for r in group), n),
            "mrr": round(sum(1 / r["rank"] for r in group if r["rank"]) / n, 3) if n else None,
            "status": {s: sum(r["status"] == s for r in group) for s in sorted({r["status"] for r in group})},
            "found_without_evidence": sum(r["status"] == "found" and r["rank"] is None for r in group),
            "not_found_with_evidence": sum(r["status"] == "not_found" and r["rank"] is not None for r in group),
        }
    ref = [r for r in rows if r["want"] == "refuse"]
    # Questions answered by another layer stay out of the "not in any document" total.
    outside = [r for r in ref if r["group"] != "other_layer"]
    for name, group in [("not_in_corpus", outside)] + [(k, [r for r in ref if r["group"] == k]) for k in sorted({r["group"] for r in ref})]:
        n = len(group)
        out_[f"refuse:{name}"] = {
            "n": n,
            "refused": pct(sum(r["status"] == "not_found" for r in group), n),
            "said_found": pct(sum(r["status"] == "found" for r in group), n),
            "said_partly": pct(sum(r["status"] == "partly_found" for r in group), n),
            "status": {s: sum(r["status"] == s for r in group) for s in sorted({r["status"] for r in group})},
        }
    lat = sorted(r["seconds"] for r in rows[1:])
    out_["latency"] = {"median_s": round(statistics.median(lat), 2), "p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2)}
    return out_


async def main():
    server = build_server(get_settings())
    rows = []
    for row in gold:
        want, group = expected(row)
        if want is None:
            continue
        t = time.perf_counter()
        r = payload(await server.call_tool("strata_ask", {"question": row["question"], "layers": [layer], "response_format": "detailed"}))
        seconds = time.perf_counter() - t
        ps = r.get("passages") or []
        rows.append({
            "id": row["id"], "question": row["question"], "want": want, "group": group, "sampled": row.get("sampled", False),
            "status": r.get("answer_status") or r.get("status"), "seconds": round(seconds, 2),
            "rank": rank(row, ps) if want == "answer" else None,
            "top": [{"file": (p.get("citation") or {}).get("file"), "text": (p.get("text") or "")[:300]} for p in ps[:3]],
            "n_passages": len(ps),
        })
    s = summary(rows)
    out.write_text(json.dumps({"layer": layer, "summary": s, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))


asyncio.run(main())
