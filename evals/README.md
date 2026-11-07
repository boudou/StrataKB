# Evaluation gold set

Frozen questions for measuring `strata_ask` against a public corpus, the Python Enhancement Proposals.

- **Corpus:** `https://github.com/python/peps` at commit `d94bba4a5dff241fd418f30223d3bf05de27a771`.
- **Layers:** `peps_small` holds 8 PEPs and `peps_full` holds all of them.

| File | Rows | sha256 |
| --- | --- | --- |
| `gold/batch1/gold.jsonl` | 125 | `b98bb221db2ba95cbeb5077cb7df6f1cda686c150e71da0a6107a358aa104ce4` |
| `gold/batch2/gold.jsonl` | 90 | `a384421dbb93d319501d165530e05cbd31a2723f85add74251e81c94358100f9` |

The rows are frozen. A label is corrected only with a reason taken from the corpus text, and the correction is logged here.

Report refusals per group, never as one total. Batch 2's refusals were written by hand after small-layer scores had been seen, and its identifier rows all come from the full corpus, so do not use it to compare the two layers by identifier.
