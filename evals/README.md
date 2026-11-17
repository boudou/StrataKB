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

Run these from the repository root. They keep the corpus and every result outside the repository; `evals/gold/` stays frozen, and the scripts refuse to write under it.

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

Before the first question it prints the layer's document and chunk counts, when it was last ingested and how its last ingestion run ended, so a layer left from an older corpus shows up. It stops with a non-zero exit in these cases, so a broken run cannot pass for a normal one:

- **The layer is missing or empty, or its last ingestion run failed.** It stops before any question, because such a layer would score every refusal as correct, or score a half-built layer as if it were whole. A file already at `--out` is left alone. A run is recorded only when it ends, so the scorer cannot see one still in progress: let the ingest finish first. A run that ended `partial` (some files failed) stops it too, since the pinned corpus ingests whole and the missing documents would count as retrieval misses. `--allow-partial` scores such a layer anyway, with a warning naming how many documents failed, and the run block records `allow_partial`.
- **A question fails, or takes longer than `--timeout` seconds** (default 120). It stops at that question and leaves no file at `--out`. A file from an earlier run is deleted when the questions start, so what is there after a run is that run's.

The results file opens with a `run` block, so two files say whether they can be compared:

| Key | What it holds |
| --- | --- |
| `commit`, `started`, `finished` | The repository's `HEAD`, with `-dirty` when tracked files have changes, and the UTC start and end of the run |
| `layer` | The layer's `chunk_size` and `chunk_overlap`, document and chunk counts, last ingest time, last ingestion run, `allow_partial` and `answer_mode` |
| `answer` | What `strata_ask` was asked and with which settings: the `mode` asked for (null without `--mode`), `top_k`, `response_format`, `answer_token_budget`, embedding and reranking models, the reranker's candidate multiplier, the confidence `thresholds`, whether hybrid search, query rewriting and learning were on, and under `extractive` the model extractive answers score sentences with and its `mark` |
| `timeout_s` | The per-question timeout |

All three scripts print their usage with `--help`, or when an argument is missing, and stop before doing anything else.

### Score extractive answers

`--mode extractive` asks every question in extractive mode, whatever the layer's own mode; `--mode grounded` asks for grounded answers, which a layer in extractive mode still answers extractively whenever passages come back (a question with no passages is answered in the mode asked for). Without `--mode` the layer's mode decides. Each row records the `mode` the answer came in.

An extractive answer quotes a sentence when its score reaches the reranker profile's medium threshold, the `mark` in `run.answer.extractive`: 0.5 for the default model, 0.86 for `Alibaba-NLP/gte-reranker-modernbert-base`. Extractive answers score with that model even when reranking is off. To choose the mark from one results file, each extractive row also holds:

| Key | What it holds |
| --- | --- |
| `top_sentence_score` | The highest score the server gave a sentence for that question, quoted or not, rounded to 4 decimals; null when no sentence was scored |
| `named_clause` | Whether a sentence came from a clause the question names, which is quoted without a score |
| `quoted` | Up to 3 quoted sentences: `file`, the first 300 characters of `text`, and `score` |

The sentences are scored once, by the server itself: the scorer records the scores `strata_ask` computes while the run lasts, and puts its hook back when the run ends or stops.

The summary then has `marks`: for each candidate mark from 0.30 to 0.95 in steps of 0.05, plus the run's own mark, the % of answerable questions that would be quoted (`answered.all`) and the % of each refusal group that would be refused (`refused`, with `not_in_corpus` and each group on its own). A row counts as quoted at a mark when `named_clause` is true or `top_sentence_score` reaches the mark. That is judged by the score alone: the server also refuses when none of the names the question asks about is in the documents, and turns `found` into `partly_found` when a clause or a name is missing, so a row's `status` can differ from what `marks` says.

### Count the sentences too long to quote

An extractive answer never quotes a sentence longer than 2,000 characters (`MAX_SENTENCE_CHARS`). `evals/sentence_lengths.py` counts them, and the named clauses (sections) holding one, so the share a layer cannot quote is known. Give it one input:

```bash
uv run python evals/sentence_lengths.py --database --layer peps_full   # --layer repeatable; omit for every layer
uv run python evals/sentence_lengths.py --folder ../contracts          # add --preset P for P's structure_patterns
uv run python evals/sentence_lengths.py --preset examples/sepa/preset.yaml
```

`--database` reads the layers of the database in `DATABASE_URL`, using only the document versions their chunks cite: the exact view of what was ingested. It only reads, and a `--layer` that does not exist stops it before any output. `--folder` reads every file the preset dry run would, which can be more than `stratakb add` keeps (build files, ignored paths). `--preset` alone reads each of the preset's source folders and names a missing one on stderr. Files with secret-looking names are never opened, and a file that fails to parse counts under `failed` and is named on stderr.

It prints JSON on stdout, and `--out` writes it to a file too (never under `evals/gold/`). The report holds `max_sentence_chars` and one entry under `sources` per layer, folder or preset source:

| Key | What it holds |
| --- | --- |
| `name`, `documents` | The layer, folder or source group, and the documents read; `failed` counts the files that did not parse (folder and preset only) |
| `sentences`, `long_sentences`, `long_pct` | The sentences, those longer than the limit, and their % |
| `clauses`, `clauses_with_long`, `clauses_with_long_pct` | The sections, those holding a long sentence, and their %. A section covers its subsections, so the sections enclosing a long clause count too |
| `longest` | The `file` and `chars` of the longest sentence, or null |
| `chunks_without_document` | Chunks ingested before stored documents existed, whose sentences cannot be counted (database only) |

A percentage is null when there is nothing to count, and has 2 decimals, so one long sentence in a large layer does not show as 0.

## What scoring writes

Each question goes through the real `strata_ask`, with learning turned off by the scorer itself, whatever `LEARNING_ENABLED` or a config file says. A run writes no row to any `strata_learning` table and does not read what is already there, so a score does not depend on the order of the questions or on what was scored before. The layers and their documents are not changed.

| Table | Written | When |
| --- | --- | --- |
| `strata_core.audit_log` | One row per question | Always |
| `strata_core.receipts`, `strata_core.receipt_documents` | One receipt per extractive answer that quotes | With `--mode extractive`, or on a layer with `answer_mode: extractive`, which the layers above are not |

`sentence_lengths.py` writes nothing to the database.
