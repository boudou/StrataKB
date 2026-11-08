# Evaluation gold set

Frozen questions for measuring `strata_ask` against a public corpus, the Python Enhancement Proposals.

- **Corpus:** `https://github.com/python/peps` at commit `d94bba4a5dff241fd418f30223d3bf05de27a771`.
- **Layers:** `peps_small` holds the 8 PEPs listed in `examples/peps/peps.txt` and `peps_full` holds all of them.

| File | Rows | sha256 |
| --- | --- | --- |
| `gold/batch1/gold.jsonl` | 125 | `b98bb221db2ba95cbeb5077cb7df6f1cda686c150e71da0a6107a358aa104ce4` |
| `gold/batch2/gold.jsonl` | 90 | `a384421dbb93d319501d165530e05cbd31a2723f85add74251e81c94358100f9` |

The rows are frozen, and `tests/test_gold_set.py` fails when a byte of them changes. A label is corrected only with a reason taken from the corpus text, and the correction is logged here.

Report refusals per group, never as one total. Batch 2's refusals were written by hand after small-layer scores had been seen, and its identifier rows all come from the full corpus, so do not use it to compare the two layers by identifier.

## Run it

Run these from the repository root. They keep the corpus and every result outside the repository; `evals/gold/` stays frozen, and both scripts refuse to write under it.

**1. Get the corpus at the pinned commit.**

```bash
git clone --filter=blob:none https://github.com/python/peps.git ../peps-eval/clone
git -C ../peps-eval/clone checkout d94bba4a5dff241fd418f30223d3bf05de27a771
mkdir ../peps-eval/full ../peps-eval/small
cp ../peps-eval/clone/peps/pep-*.rst ../peps-eval/full/
grep -v '^#' examples/peps/peps.txt | while read -r name; do cp "../peps-eval/full/$name" ../peps-eval/small/; done
```

`full/` then holds 746 PEPs and `small/` holds 8.

**2. Check the gold set against the corpus.** This needs no server and no database, and takes about a minute. It prints any row whose evidence does not match or whose source file the corpus lacks, then writes the ids of the full-tier questions the small layer cannot answer to `--out`. It exits non-zero when it reports a problem; a `WEAK` line alone is not one.

```bash
uv run python evals/check_gold.py --gold evals/gold/batch1/gold.jsonl \
  --corpus ../peps-eval/full --small ../peps-eval/small --out ../peps-eval/cross_batch1.json
cmp ../peps-eval/cross_batch1.json evals/gold/batch1/cross_refusals.json
```

At the pinned commit the output reports no problems and `cmp` finds the two files identical. Do the same with `batch2`.

**3. Use a database built for evaluation.** Scoring adds audit rows to the database it is pointed at (below), so do not point it at one you use. With the container from `stratakb db up`:

```bash
docker exec stratakb-db psql -U strata -c 'CREATE DATABASE strata_eval'
export DATABASE_URL=postgresql://strata:strata_dev_password@127.0.0.1:5433/strata_eval
```

**4. Build the two layers.** Embedding 746 PEPs takes a while on a CPU.

```bash
uv run stratakb add ../peps-eval/small --name peps_small
uv run stratakb add ../peps-eval/full --name peps_full
```

**5. Score each batch on each layer.** The scorer takes its database from `DATABASE_URL`, as the server does, and writes the results only to `--out`. It reads `cross_refusals.json` beside `--gold` unless `--cross-refusals` names another file.

```bash
uv run python evals/score_gold.py --gold evals/gold/batch1/gold.jsonl \
  --layer peps_small --out ../peps-eval/batch1_small.json
```

The scorer stops with a non-zero exit in two cases, so a broken run cannot pass for a normal one:

- **The layer is missing or empty.** It stops before any question, because such a layer would score every refusal as correct. A file already at `--out` is left alone.
- **A question fails.** It stops at that question and leaves no file at `--out`. A file from an earlier run is deleted when the questions start, so what is there after a run is that run's.

Both scripts print their usage with `--help`, or when an argument is missing, and stop before doing anything else.

## What scoring writes

Each question goes through the real `strata_ask`, with learning turned off by the scorer itself, whatever `LEARNING_ENABLED` or a config file says. A run writes no row to any `strata_learning` table and does not read what is already there, so a score does not depend on the order of the questions or on what was scored before. The layers and their documents are not changed.

| Table | Written | When |
| --- | --- | --- |
| `strata_core.audit_log` | One row per question | Always |
| `strata_core.receipts`, `strata_core.receipt_documents` | One receipt per extractive answer | Only on a layer with `answer_mode: extractive`, which the layers above are not |
