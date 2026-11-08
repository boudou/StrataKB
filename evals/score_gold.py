"""Score a gold.jsonl on one layer through strata_ask, on the database in DATABASE_URL.

On peps_small: small-tier questions are answerable; full-tier questions listed in the
cross-refusals file and all refusals must be refused.
On peps_full: every question but the refusals is answerable; refusals must be refused.

Of the files, only the --out file is written, and only by a run that finished. Learning is off
whatever the settings say, so a question writes only its audit row: see evals/README.md.

Stops with a non-zero exit before any question on a layer that is missing or empty, and at the
first question that fails.
"""
import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

FROZEN = Path(__file__).resolve().parent / "gold"


class RunStoppedError(Exception):
    """The run cannot produce a score worth reading; the message says why."""


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, required=True, help="the gold.jsonl to score")
    parser.add_argument(
        "--layer", required=True, choices=["peps_small", "peps_full"], help="the layer to ask"
    )
    parser.add_argument("--out", type=Path, required=True, help="file to write the results to")
    parser.add_argument(
        "--cross-refusals",
        type=Path,
        help="the checker's cross-layer refusal ids (default: cross_refusals.json beside --gold)",
    )
    args = parser.parse_args(argv)
    if args.cross_refusals is None:
        args.cross_refusals = args.gold.with_name("cross_refusals.json")
    for name, path in (("--gold", args.gold), ("--cross-refusals", args.cross_refusals)):
        if not path.is_file():
            parser.error(f"{name}: no such file: {path}")
    if args.out.resolve().is_relative_to(FROZEN):
        parser.error(f"--out: {FROZEN.name}/ is frozen, write somewhere else")
    if not args.out.parent.is_dir():
        parser.error(f"--out: no such folder: {args.out.parent}")
    return args


def payload(result):
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return result
    return json.loads(result[0].text)


def expected(row, layer, cross):
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
    for name, group in [("all", ans), ("random_sample", [r for r in ans if r["sampled"]])] + [
        (k, [r for r in ans if r["group"] == k]) for k in sorted({r["group"] for r in ans})
    ]:
        n = len(group)
        out_[f"answer:{name}"] = {
            "n": n,
            "r@1": pct(sum(r["rank"] == 1 for r in group), n),
            "r@3": pct(sum((r["rank"] or 99) <= 3 for r in group), n),
            "any": pct(sum(r["rank"] is not None for r in group), n),
            "mrr": round(sum(1 / r["rank"] for r in group if r["rank"]) / n, 3) if n else None,
            "status": {
                s: sum(r["status"] == s for r in group)
                for s in sorted({r["status"] for r in group})
            },
            "found_without_evidence": sum(
                r["status"] == "found" and r["rank"] is None for r in group
            ),
            "not_found_with_evidence": sum(
                r["status"] == "not_found" and r["rank"] is not None for r in group
            ),
        }
    ref = [r for r in rows if r["want"] == "refuse"]
    # Questions answered by another layer stay out of the "not in any document" total.
    outside = [r for r in ref if r["group"] != "other_layer"]
    by_group = [(k, [r for r in ref if r["group"] == k]) for k in sorted({r["group"] for r in ref})]
    for name, group in [("not_in_corpus", outside)] + by_group:
        n = len(group)
        out_[f"refuse:{name}"] = {
            "n": n,
            "refused": pct(sum(r["status"] == "not_found" for r in group), n),
            "said_found": pct(sum(r["status"] == "found" for r in group), n),
            "said_partly": pct(sum(r["status"] == "partly_found" for r in group), n),
            "status": {
                s: sum(r["status"] == s for r in group)
                for s in sorted({r["status"] for r in group})
            },
        }
    # The first question also loads the models, so it counts only when it is alone.
    lat = sorted(r["seconds"] for r in rows[1:] or rows)
    out_["latency"] = {
        "median_s": round(statistics.median(lat), 2) if lat else None,
        "p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2) if lat else None,
    }
    return out_


async def check_layer(database_url, layer):
    from stratakb.knowledge.layers import LayerNotFoundError, get_layer_status

    try:
        status = await get_layer_status(database_url, name=layer)
    except LayerNotFoundError:
        raise RunStoppedError(f"layer {layer} does not exist, build it first") from None
    except Exception as e:
        raise RunStoppedError(f"cannot read layer {layer}: {type(e).__name__}: {e}") from e
    if not status["chunk_count"]:
        raise RunStoppedError(f"layer {layer} holds no chunks, build it first")


async def ask(server, row, layer):
    try:
        r = payload(await server.call_tool("strata_ask", {
            "question": row["question"], "layers": [layer], "response_format": "detailed",
        }))
    except Exception as e:
        raise RunStoppedError(f"question {row['id']} failed: {type(e).__name__}: {e}") from e
    if r.get("status") != "ok":
        raise RunStoppedError(f"question {row['id']} failed: {r.get('error') or r.get('status')}")
    return r


async def score(gold, layer, cross, out):
    from stratakb.config import get_settings
    from stratakb.core.database import close_pool
    from stratakb.mcp_server import build_server

    # Set here, so neither the environment nor a config file can turn it back on
    settings = get_settings().model_copy(update={"learning_enabled": False})
    rows = []
    try:
        await check_layer(settings.database_url, layer)
        # From here on, a file at --out can only be this run's, written when it finishes
        out.unlink(missing_ok=True)
        server = build_server(settings)
        for row in gold:
            want, group = expected(row, layer, cross)
            if want is None:
                continue
            t = time.perf_counter()
            r = await ask(server, row, layer)
            seconds = time.perf_counter() - t
            ps = r.get("passages") or []
            rows.append({
                "id": row["id"], "question": row["question"], "want": want, "group": group,
                "sampled": row.get("sampled", False),
                "status": r["answer_status"], "seconds": round(seconds, 2),
                "rank": rank(row, ps) if want == "answer" else None,
                "top": [
                    {
                        "file": (p.get("citation") or {}).get("file"),
                        "text": (p.get("text") or "")[:300],
                    }
                    for p in ps[:3]
                ],
                "n_passages": len(ps),
            })
    finally:
        await close_pool()
    s = summary(rows)
    result = {"layer": layer, "summary": s, "rows": rows}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))


def main(argv=None):
    args = parse_args(argv)
    gold = [json.loads(x) for x in args.gold.read_text(encoding="utf-8").splitlines()]
    cross = set(json.loads(args.cross_refusals.read_text(encoding="utf-8")))

    from stratakb.core.logging import configure_logging

    configure_logging("ERROR")
    try:
        asyncio.run(score(gold, args.layer, cross, args.out))
    except RunStoppedError as e:
        sys.exit(f"Stopped, no results written: {e}")


if __name__ == "__main__":
    main()
