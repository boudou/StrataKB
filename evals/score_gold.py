"""Score a gold.jsonl on one layer through strata_ask, on the database in DATABASE_URL.

On peps_small: small-tier questions are answerable; full-tier questions listed in the
cross-refusals file and all refusals must be refused.
On peps_full: every question but the refusals is answerable; refusals must be refused.

Of the files, only the --out file is written, and only by a run that finished. Learning is off
whatever the settings say, so a question writes only its audit row, and an extractive answer
that quotes also its receipt: see evals/README.md.

With --mode extractive, each row also records the top score of the sentences the server scored
for its question, quoted or not, and the summary says what share would be answered and refused
at each candidate mark, so the extractive mark can be chosen from one results file.

Stops with a non-zero exit before any question on a layer that is missing, empty or whose last
ingestion run failed, and at the first question that fails or takes longer than --timeout.
The results file starts with a "run" block: the commit, the time, the layer's chunk settings
and counts, and the answer settings, so two results files say whether they can be compared.
"""
import argparse
import asyncio
import contextlib
import dataclasses
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

FROZEN = Path(__file__).resolve().parent / "gold"
REPO = Path(__file__).resolve().parent.parent
# strata_ask's own default, passed so the results file can say what was asked
TOP_K = 10
RESPONSE_FORMAT = "detailed"
DEFAULT_TIMEOUT = 120.0
MODES = ("grounded", "extractive")
# The candidate extractive marks the summary sweeps, with the run's own mark added
MARKS = [round(0.30 + 0.05 * i, 2) for i in range(14)]


class RunStoppedError(Exception):
    """The run cannot produce a score worth reading; the message says why."""


class QuestionTimeoutError(RunStoppedError):
    """A question took longer than the timeout."""


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
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds one question may take before the run stops (default: {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        help="the answer mode to ask for (default: none, so the layer's own mode decides)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="score a layer whose last ingestion run ended partial (some files failed)",
    )
    args = parser.parse_args(argv)
    if not args.timeout > 0:
        parser.error("--timeout: must be more than 0 seconds")
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


def quoted_at(row, mark):
    """Whether an extractive row would quote at mark: a named clause, or a score reaching it."""
    score = row["top_sentence_score"]
    return row["named_clause"] or (score is not None and score >= mark)


def marks(rows, mark):
    """For each candidate mark, the % of answerable rows quoted and of each refuse group not.

    Judged by the score alone: a row the server refused for another reason (none of the
    names the question asks about is in the documents) still counts as quoted here.
    """
    rows = [r for r in rows if r["mode"] == "extractive"]
    ans = [r for r in rows if r["want"] == "answer"]
    ref = [r for r in rows if r["want"] == "refuse"]
    groups = [("not_in_corpus", [r for r in ref if r["group"] != "other_layer"])] + [
        (k, [r for r in ref if r["group"] == k]) for k in sorted({r["group"] for r in ref})
    ]
    return [
        {
            "mark": m,
            "answered": {"all": pct(sum(quoted_at(r, m) for r in ans), len(ans))},
            "refused": {
                name: pct(sum(not quoted_at(r, m) for r in group), len(group))
                for name, group in groups
            },
        }
        for m in sorted({*MARKS, mark})
    ]


def summary(rows, mark=None):
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
    if any(r["mode"] == "extractive" for r in rows):
        out_["marks"] = marks(rows, mark)
    return out_


def when(value):
    return value.isoformat() if isinstance(value, datetime) else value


async def check_layer(database_url, layer, allow_partial=False):
    """The layer's status and its latest ingestion run; stops on one not fit to score.

    A run that ended partial stops it too, unless allow_partial: the pinned corpus
    ingests whole, so missing documents would be scored as retrieval misses.
    """
    from stratakb.knowledge.layers import (
        LayerNotFoundError,
        get_layer_status,
        last_ingestion_run,
    )

    try:
        status = await get_layer_status(database_url, name=layer)
    except LayerNotFoundError:
        raise RunStoppedError(f"layer {layer} does not exist, build it first") from None
    except Exception as e:
        raise RunStoppedError(f"cannot read layer {layer}: {type(e).__name__}: {e}") from e
    if not status["chunk_count"]:
        raise RunStoppedError(f"layer {layer} holds no chunks, build it first")
    try:
        run = await last_ingestion_run(database_url, status["id"])
    except Exception as e:
        raise RunStoppedError(
            f"cannot read the ingestion runs of layer {layer}: {type(e).__name__}: {e}"
        ) from e
    if run is not None:
        run = {k: when(v) for k, v in run.items()}
        if run["status"] not in ("completed", "partial"):
            raise RunStoppedError(
                f"layer {layer}: its last ingestion run ({run['completed_at']}) is"
                f" {run['status']}, build it again"
            )
        if run["status"] == "partial" and not allow_partial:
            raise RunStoppedError(
                f"layer {layer}: its last ingestion run ({run['completed_at']}) is partial,"
                f" {run['docs_failed']} documents failed; build it again, or pass"
                " --allow-partial to score it without them"
            )
    print(
        f"Layer {layer}: {status.get('doc_count')} documents, {status['chunk_count']} chunks,"
        f" last ingested {when(status.get('last_ingested'))}, last run "
        + (f"{run['status']} at {run['completed_at']}" if run else "not recorded"),
        file=sys.stderr,
    )
    if run is not None and run["status"] == "partial":
        print(
            f"Warning: {run['docs_failed']} documents failed in that run; the layer lacks them",
            file=sys.stderr,
        )
    return {"status": status, "last_run": run, "allow_partial": allow_partial}


def commit():
    """HEAD of the repository, with "-dirty" when the tree has changes; None without git."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if changes else head


def extractive_mark(settings):
    """The sentence score an extractive answer quotes at: the reranker profile's medium score.

    Extractive answers score with the reranking model even when reranking is off.
    """
    from stratakb.rag.reranker import reranker_profile

    return reranker_profile(settings.reranking_model).thresholds.medium_score


def run_block(settings, layer, checked, timeout, started, mode=None):
    """What was scored and how: enough to tell whether two results files compare."""
    from stratakb.knowledge.layers import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
    from stratakb.rag.query_rewriter import rewrite_configured
    from stratakb.rag.reranker import ranking_settings

    status = checked["status"]
    chunk_config = status.get("chunk_config") or {}
    thresholds, multiplier = ranking_settings(settings)
    return {
        "commit": commit(),
        "started": started,
        "finished": datetime.now(UTC).isoformat(),
        "layer": {
            "name": layer,
            "chunk_size": chunk_config.get("chunk_size", DEFAULT_CHUNK_SIZE),
            "chunk_overlap": chunk_config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            "doc_count": status.get("doc_count"),
            "chunk_count": status.get("chunk_count"),
            "last_ingested": when(status.get("last_ingested")),
            "last_run": checked["last_run"],
            "allow_partial": checked["allow_partial"],
            "answer_mode": status.get("answer_mode"),
        },
        "answer": {
            "mode": mode,
            "top_k": TOP_K,
            "response_format": RESPONSE_FORMAT,
            "answer_token_budget": settings.answer_token_budget,
            "embedding_model": settings.embedding_model,
            "hybrid_search_enabled": settings.hybrid_search_enabled,
            "reranking_enabled": settings.reranking_enabled,
            "reranking_model": settings.reranking_model if settings.reranking_enabled else None,
            "reranking_candidate_multiplier": multiplier,
            "thresholds": dataclasses.asdict(thresholds),
            "query_rewrite": rewrite_configured(settings),
            "learning_enabled": settings.learning_enabled,
            "extractive": {
                "reranking_model": settings.reranking_model,
                "mark": extractive_mark(settings),
            },
        },
        "timeout_s": timeout,
    }


@contextlib.contextmanager
def sentence_scores():
    """Record the sentence scores of each extractive answer while the block runs.

    Yields a dict: "top" is the highest score the cross-encoder gave a sentence (None when
    none was scored) and "exact" whether a candidate was a named clause, quoted without a
    score. Clear it before each question. strata_ask returns only the sentences it quotes,
    so this wraps select_sentences and records what score_pairs returns inside it: the
    sentences are scored once, by the server itself. Both names are put back on the way
    out, however the block ends.
    """
    from stratakb.rag import reranker, tools

    record = {"top": None, "exact": False}
    original = tools.select_sentences

    def select(question, candidates, *args, **kwargs):
        record["exact"] = record["exact"] or any(c.exact for c in candidates)
        scoring = reranker.score_pairs

        def recorder(*a, **k):
            scores = scoring(*a, **k)
            if len(scores):
                top = max(scores)
                record["top"] = top if record["top"] is None else max(record["top"], top)
            return scores

        reranker.score_pairs = recorder
        try:
            return original(question, candidates, *args, **kwargs)
        finally:
            reranker.score_pairs = scoring

    tools.select_sentences = select
    try:
        yield record
    finally:
        tools.select_sentences = original


def extractive_fields(r, record):
    """The row keys of an extractive answer: its top sentence score and what it quoted."""
    top = record["top"]
    return {
        "top_sentence_score": None if top is None else round(float(top), 4),
        "named_clause": record["exact"],
        "quoted": [
            {
                "file": (item.get("citation") or {}).get("file"),
                "text": (item.get("text") or "")[:300],
                "score": item.get("score"),
            }
            for item in (r.get("answer") or [])[:3]
        ],
    }


async def ask(server, row, layer, timeout=DEFAULT_TIMEOUT, mode=None):
    arguments = {
        "question": row["question"], "layers": [layer], "top_k": TOP_K,
        "response_format": RESPONSE_FORMAT,
    }
    if mode is not None:
        arguments["mode"] = mode
    call = asyncio.ensure_future(server.call_tool("strata_ask", arguments))
    # asyncio.wait, not wait_for: a TimeoutError the tool raises itself is a failure, not this
    done, _ = await asyncio.wait({call}, timeout=timeout)
    if not done:
        call.cancel()
        raise QuestionTimeoutError(f"question {row['id']} timed out after {timeout:g} s")
    try:
        r = payload(call.result())
    except Exception as e:
        raise RunStoppedError(f"question {row['id']} failed: {type(e).__name__}: {e}") from e
    if r.get("status") != "ok":
        raise RunStoppedError(f"question {row['id']} failed: {r.get('error') or r.get('status')}")
    return r


async def score(
    gold, layer, cross, out, timeout=DEFAULT_TIMEOUT, *, mode=None, allow_partial=False
):
    from stratakb.config import get_settings
    from stratakb.core.database import close_pool
    from stratakb.mcp_server import build_server

    # Set here, so neither the environment nor a config file can turn it back on
    settings = get_settings().model_copy(update={"learning_enabled": False})
    started = datetime.now(UTC).isoformat()
    rows = []
    try:
        checked = await check_layer(settings.database_url, layer, allow_partial)
        # From here on, a file at --out can only be this run's, written when it finishes
        out.unlink(missing_ok=True)
        server = build_server(settings)
        with sentence_scores() as record:
            for row in gold:
                want, group = expected(row, layer, cross)
                if want is None:
                    continue
                record.update(top=None, exact=False)
                t = time.perf_counter()
                r = await ask(server, row, layer, timeout, mode)
                seconds = time.perf_counter() - t
                ps = r.get("passages") or []
                rows.append({
                    "id": row["id"], "question": row["question"], "want": want, "group": group,
                    "sampled": row.get("sampled", False), "mode": r.get("mode"),
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
                    **(extractive_fields(r, record) if r.get("mode") == "extractive" else {}),
                })
    finally:
        await close_pool()
    s = summary(rows, extractive_mark(settings))
    run = run_block(settings, layer, checked, timeout, started, mode)
    result = {"layer": layer, "run": run, "summary": s, "rows": rows}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))


def run_loop(coro):
    """asyncio.run without waiting on worker threads at the end.

    A question that timed out may leave a thread behind that never ends (a stalled
    model download, say); asyncio.run would wait for it before returning.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)
        loop.close()


def main(argv=None):
    args = parse_args(argv)
    gold = [json.loads(x) for x in args.gold.read_text(encoding="utf-8").splitlines()]
    cross = set(json.loads(args.cross_refusals.read_text(encoding="utf-8")))

    from stratakb.core.logging import configure_logging

    configure_logging("ERROR")
    try:
        run_loop(score(
            gold, args.layer, cross, args.out, args.timeout,
            mode=args.mode, allow_partial=args.allow_partial,
        ))
    except QuestionTimeoutError as e:
        # The interpreter would wait on the way out for a thread the timed-out question
        # left running, so leave without waiting
        sys.stderr.write(f"Stopped, no results written: {e}\n")
        sys.stderr.flush()
        os._exit(1)
    except RunStoppedError as e:
        sys.exit(f"Stopped, no results written: {e}")


if __name__ == "__main__":
    main()
