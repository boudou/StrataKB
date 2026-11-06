# Example layer presets

Each folder holds a preset for one knowledge layer. Source folders are relative to the preset file, so download the documents into the folders shown below and apply the preset:

```bash
uv run python scripts/apply_preset.py examples/sepa/preset.yaml --dry-run   # check the files parse
uv run python scripts/apply_preset.py examples/sepa/preset.yaml             # build the layer
```

To keep the documents elsewhere, pass `--docs-root /path/to/docs` (or `docs_root` to the `strata_apply_preset` tool). Missing folders are skipped and reported with their download link.

Standards documents are published under their publishers' terms, so `.gitignore` keeps downloaded PDFs, DOCX, XSD and ZIP files under `examples/` out of git. It also keeps the PEPs clone (`peps/source/`) out of git.

## Python PEPs (`peps/`)

The quickest example to try: Python Enhancement Proposals need no account and no manual download, only git. Nothing is committed to this repo; the files stay in your clone under `peps/source/`.

**A few PEPs** (the default, a few seconds):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/python/peps.git examples/peps/source
git -C examples/peps/source sparse-checkout set --no-cone /peps/pep-0008.rst /peps/pep-0020.rst /peps/pep-0257.rst /peps/pep-0484.rst /peps/pep-0572.rst /peps/pep-0634.rst /peps/pep-0695.rst /peps/pep-0723.rst
```

In Git Bash on Windows, put `MSYS_NO_PATHCONV=1` in front of the `sparse-checkout` command; otherwise Git Bash rewrites the `/peps/...` paths and nothing is checked out.

The list of PEPs is kept in `peps/peps.txt`. To change it, edit that file and the `sparse-checkout` commands here, in the main README and in `.github/workflows/ci.yml`; the tests fail while they differ.

**All PEPs:** `git -C examples/peps/source sparse-checkout set peps`. That is about 770 files and 12,000 chunks, which take a while to embed on a CPU.

**Build the layer:**

```bash
uv run python scripts/apply_preset.py examples/peps/preset.yaml --dry-run   # parse only, no database
uv run python scripts/apply_preset.py examples/peps/preset.yaml             # create the layer and ingest
```

**Update later:** `git -C examples/peps/source pull`, then apply the preset again. Applying rebuilds the layer and re-embeds every file.

**Without the preset:** `uv run stratakb add examples/peps/source/peps` builds a layer named `peps` that has no routing patterns. A later run re-embeds only the files that changed. Use either the preset or `stratakb add` on this folder, not both, or every answer comes back twice.

Questions to try:

| Ask | The answer should cite |
| --- | --- |
| How long should a line of Python be? | PEP 8 |
| What does the walrus operator do? | PEP 572 |
| How do I declare inline script dependencies? | PEP 723 |

Licence: each PEP is placed in the public domain or under CC0-1.0-Universal (each file says which in its Copyright section).

## SEPA payments (`sepa/`)

Rulebooks and implementation guidelines are published by the European Payments Council (EPC). On [europeanpaymentscouncil.eu](https://www.europeanpaymentscouncil.eu/), open **What we do → EPC payment schemes**, choose a scheme (SEPA Credit Transfer, SCT Inst, SEPA Direct Debit Core or B2B) and download its rulebook and implementation guidelines.

| Folder | Put here |
| --- | --- |
| `rulebooks/` | Rulebook PDFs, e.g. the SCT, SCT Inst and SDD Core rulebooks |
| `guidelines/` | Customer-to-PSP and inter-PSP implementation guideline PDFs |
| `reason_codes/` | Reason code lists as CSV; a 12-code sample ships here |

## ISO 20022 messaging (`iso20022/`)

Published by the ISO 20022 Registration Authority on [iso20022.org](https://www.iso20022.org/).

| Folder | Put here | Where |
| --- | --- | --- |
| `message_definitions/` | Message Definition Reports (PDF) for the message sets you use, e.g. Payments Initiation (pain), Payments Clearing and Settlement (pacs), Cash Management (camt) | [ISO 20022 message definitions](https://www.iso20022.org/iso-20022-message-definitions): select a message set, then download its documents |
| `schemas/` | The XSD schemas for the same message sets | Same page, per message set |
| `external_code_sets/` | External code sets in XSD or JSON (XLSX is not parsed) | [External code sets](https://www.iso20022.org/catalogue-messages/additional-content-messages/external-code-sets) |

## Writing your own

Copy a preset, change `name`, `display_name`, `description`, `routing_patterns` and `sources` (add `answer_mode: extractive` for a layer that should answer with the documents' own sentences, and `structure_patterns` when your documents number their sections with identifiers of their own, as the SEPA preset does for its attribute definitions, `Identification: AT-R051`, and the ISO 20022 preset for message identifiers such as `pacs.008.001.08`), and point the source paths at your folders. The [main README](../README.md#layers-and-presets) describes every key.
