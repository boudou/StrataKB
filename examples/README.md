# Example layer presets

Each folder holds a preset for one knowledge layer. Source folders are relative to the preset file, so download the documents into the folders shown below and apply the preset:

```bash
uv run python scripts/apply_preset.py examples/sepa/preset.yaml --dry-run   # check the files parse
uv run python scripts/apply_preset.py examples/sepa/preset.yaml             # build the layer
```

To keep the documents elsewhere, pass `--docs-root /path/to/docs` (or `docs_root` to the `strata_apply_preset` tool). Missing folders are skipped and reported with their download link.

Standards documents are published under their publishers' terms, so `.gitignore` keeps downloaded PDFs, DOCX, XSD and ZIP files under `examples/` out of git.

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

Copy a preset, change `name`, `display_name`, `description`, `routing_patterns` and `sources`, and point the source paths at your folders. The [main README](../README.md#layers-and-presets) describes every key.
