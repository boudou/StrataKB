<div align="center">

# StrataKB

**A layered, self-learning knowledge base served over the Model Context Protocol.**
Point it at your documentation, organise it in layers, and query it from any MCP client.

[![CI](https://github.com/boudou/StrataKB/actions/workflows/ci.yml/badge.svg)](https://github.com/boudou/StrataKB/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-informational)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick start](#quick-start) · [How it works](#how-it-works) · [Layers and presets](#layers-and-presets) · [Tools](#tools) · [Configuration](#configuration) · [Glossary](#glossary) · [Roadmap](#roadmap)

</div>

StrataKB turns folders of your documents into a knowledge base that your AI tools can ask through the Model Context Protocol (MCP). Each folder becomes a **layer**: a searchable collection named after the folder. One PostgreSQL database holds every layer, so a whole team can share one server, with roles and an audit log.

StrataKB finds the passages that answer a question and names the files they come from; your MCP client writes the answer from them.

## Quick start

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/) and [Git](https://git-scm.com/). uv installs Python 3.12 if you don't have it.

```mermaid
flowchart LR
    i["Install"] --> d["Start the database"] --> a["Add a folder"] --> c["Connect your client"] --> q["Ask"]
```

### 1. Install

```bash
git clone https://github.com/boudou/StrataKB.git
cd StrataKB
uv sync
uv run stratakb db up        # a local database in Docker, on 127.0.0.1:5433
```

Run `uv run stratakb db up` again any time: it starts the database if it was stopped and keeps the data. `uv run stratakb db down` stops it. If it prints a `DATABASE_URL=` line, put that line where it says.

### 2. Add a folder

```bash
uv run stratakb add path/to/docs
```

It creates a layer named after the folder and reads the documents in it: PDF, Word, Markdown, HTML, plain text and more. It skips files that look like secrets, code or build output, and tells you what it skipped and why (`--json` lists every file). Run it again after you edit the documents: it reads only what changed.

The first run downloads a model of about 420 MB into your Hugging Face cache, and the first question downloads one of about 90 MB. To keep everything on this machine, see [Where your data goes](#where-your-data-goes).

### 3. Connect your client

Add the server to your client's MCP configuration (for example a project `.mcp.json`):

```json
{
  "mcpServers": {
    "stratakb": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/StrataKB", "python", "-m", "stratakb"],
      "env": {
        "DATABASE_URL": "postgresql://strata:strata_dev_password@127.0.0.1:5433/strata",
        "STRATA_USER": "developer",
        "STRATA_ROLE": "admin"
      }
    }
  }
}
```

Replace /path/to/StrataKB with the folder you cloned into, then restart the client. It starts StrataKB itself.

### 4. Ask

Ask your client a question about the documents, for example *"How do we fail over the database?"*. The client calls StrataKB's `strata_ask` tool, gets back the passages that match and the files they come from, and writes the answer.

No documents at hand? Try the [Python PEPs example](#try-an-example-preset).

The reference below explains how StrataKB works, every tool and every setting; the terms it uses are in the [Glossary](#glossary).

## Features

- **Knowledge layers**: create a layer per domain at runtime, each with its own chunking and routing rules; no code changes to add a new domain
- **Layer presets**: describe a layer and its source folders in one YAML file, then build or rebuild it in one call
- **Hybrid retrieval**: pgvector HNSW similarity plus PostgreSQL full-text search, fused with Reciprocal Rank Fusion, then reranked by a cross-encoder
- **Query routing**: when no layers are given, questions are routed to the right layers by each layer's patterns. A routed search that finds fewer results than asked for, or only low-confidence ones, runs again across all layers, and the result says so (`routing_fallback`, with the reason). Layers you name yourself are never broadened
- **Query rewriting** (optional): with an Anthropic API key or any OpenAI-compatible endpoint (including a local Ollama), `strata_ask` expands a question into several searches in the knowledge base's own terminology
- **Learns from use**: records searches, detects rephrased questions and layer switches, and takes explicit feedback on each result's `interaction_id`. It then learns synonyms, query rewrites that worked (a proven rewrite skips the LLM call) and which layers a question belongs to
- **Many formats**: PDF, DOCX, Markdown, HTML, plain text, reStructuredText, AsciiDoc, CSV (up to 10,000 rows per file), JSON, XML/XSD and OpenAPI/AsyncAPI YAML, each split along its natural structure
- **Local embeddings by default**: `all-mpnet-base-v2` runs on CPU with no API key; OpenAI or any OpenAI-compatible server (Ollama, LM Studio, vLLM) is optional
- **Roles and audit**: viewer, operator and admin roles on every tool, and an append-only audit log of searches and changes
- **Local-only switch**: `STRATA_LOCAL_ONLY=true` refuses to start while any setting would send content off this machine, and keeps model downloads off; see [Where your data goes](#where-your-data-goes)
- **Privacy guards**: IBANs and card numbers are masked in logs, and IBANs, BICs and account numbers are scrubbed from stored queries and audited questions; secrets are redacted from audit entries; files whose names look like secrets (`.env`, keys, certificates, credentials) are never read or ingested
- **Read-only SQL**: inspect the store with schema browsing, sample chunks, vector statistics and `EXPLAIN`

## How it works

```mermaid
flowchart LR
    client["MCP client"] <-->|"stdio"| server["StrataKB server"]
    server <--> pg[("PostgreSQL + pgvector<br/>layers, chunks, learning, audit")]
```

**Start**

```mermaid
flowchart LR
    start["python -m stratakb"] --> lo{"STRATA_LOCAL_ONLY on and a<br/>setting sends data off?"}
    lo -->|"yes"| refuse["Refuse to start,<br/>name each setting"]
    lo -->|"no"| db["Connect to PostgreSQL,<br/>update the schema"] --> ro{"READONLY_DATABASE_URL<br/>set, and a safe role?"}
    ro -->|"yes"| ready["Answer the client"]
    ro -->|"no"| off["Drop strata_db_query<br/>and strata_db_explain"] --> ready
    db -.->|"background"| m["Load the local embedding model<br/>(local backend only)"] -.-> s["First search<br/>(waits if still loading)"]
```

**Ingest**

```mermaid
flowchart LR
    add["stratakb add"] --> sel["Select files<br/>(.gitignore, .strataignore)"] --> pol{"Secret, code, build file,<br/>unsupported format,<br/>path too long or too large?"}
    pol -->|"yes"| k["Skip, never opened"]
    pol -->|"no"| h
    f["strata_ingest_documents"] --> n{"Secret-looking<br/>name?"}
    n -->|"yes"| k2["Skip, never opened;<br/>remove if ingested before"]
    n -->|"no"| h{"Content<br/>changed?"}
    h -->|"no"| s["Skip"]
    h -->|"new or changed"| z{"Empty or<br/>whitespace only?"}
    z -->|"yes"| t["No text:<br/>skip or remove"]
    z -->|"no"| p["Parse<br/>per format"]
    p -->|"no text"| t
    p --> sm["Structure map<br/>(headings, numbered clauses)"] --> c["Chunk inside each section<br/>(the file's own text)"] --> x["Add context for the embedding only<br/>[layer > file > headings or Page N]"] --> e["Embed"] --> v[("Store the document text<br/>(new version if it changed)<br/>and that file's chunks")]
    p -.->|"scanned-looking PDF,<br/>CSV over 10,000 rows"| fl["Stored, reported<br/>as flagged"]
    h -->|"can't read"| fe["Failed, keeps earlier chunks<br/>(read_error, parse_error)"]
    p -->|"parser rejects"| fe
```

Ingesting a folder again is a sync: unchanged files are skipped, changed files are replaced, new files are added, and files deleted from that folder are removed. Other folders in the same layer are left alone. Files whose names look like secrets (`.env`, `client_secret.json`, `id_rsa`, `*.pem`) are skipped without being opened, and one ingested earlier is removed. Empty files are reported as no text rather than as failures. A PDF that looks scanned (fewer than half its pages have text, or under 200 characters per page on average) is stored for the text it has and reported as flagged. A CSV file over 10,000 rows is stored for its first 10,000 rows and reported as flagged (`truncated`); the log says how many rows were left out. A CSV row with fewer fields than the header is stored with the values it has; one with more keeps the extra values, labelled by position (`Column 4`). A column with an empty header is labelled by position too (`Column 2`), and a header name used twice gets a number on its second use (`Name (2)`), so no value is lost. A file that cannot be read (locked, or no permission) or that its parser rejects is reported as failed and keeps its earlier chunks; the rest of the run carries on.

Every file gets an outcome, and most outcomes a reason. `stratakb add --json` lists both for every file; `strata_ingest_documents` returns the counts, and each failed file in `errors` with its reason and a short message. Errors returned and kept in the run history name a file by its path relative to the ingested folder, not its absolute path. The ingest logs on stderr name a failure by its error type, not its message. `stratakb add` also reports the reasons of its own file selection (such as `code` or `ignored_by_gitignore`) for files it never reads.

| Outcome | Reasons |
|---|---|
| `added`, `updated`, `unchanged` | none |
| `flagged`: stored, and also counted as added or updated | `probably_scanned`, `truncated` |
| `skipped`: not stored | `secret`, `no_text` |
| `removed` | `deleted`, `not_selected`, `secret`, `no_text` |
| `failed`: keeps its earlier chunks | `read_error` (could not be read), `parse_error` (the parser rejected it), `embed_error` (embedding or storing failed) |

Each stored chunk is the document's own text as its parser extracted it (for Markdown and plain text, the file's characters; for PDF, HTML and other formats, the extracted text): search results quote that text exactly, and a Markdown or HTML section holds its own heading once. Where the chunk sits (`[layer > file > headings]` for Markdown and HTML, `[layer > file > Page N]` otherwise) is added only to the text that is embedded, so semantic search keeps that context; the heading path is also in the chunk's metadata as `section_path`. In Markdown, the front matter (a leading `---` block of `key: value` lines) is never part of the text, `#` lines inside code fences (```` ``` ```` or `~~~`) are not headings, so a shell comment does not split a code block, and headings may be indented up to three spaces or underlined with `===` / `---`. An HTML page's first `<title>` outside an inline `<svg>` is not part of the text: it is the section path of text that sits under no heading (the whole page, when it has no headings), while an SVG's `<title>` stays in the text. Keyword search matches the chunk text, not layer or file names. Chunks stored by earlier versions keep their bracketed prefix until their file is ingested again.

Each file's extracted text is stored once as a document, and every chunk records where in that text it sits (a character range, from which the page and the line range follow), so a passage can be located and checked against its source. Lines count from the start of the page for PDFs and from the start of the document for other formats. When a file's text changes, the next ingest stores it as a new version; versions that no chunk and no receipt uses any more are removed. Chunks ingested before this was added have no position until their file changes or the layer is rebuilt (`replace_all=True`).

Chunks follow the document's structure. Each document gets a structure map: its headings and numbered clauses, each with a path such as `7 Termination > 7.2 Termination for cause` and the range of the text it covers, found without any configuration. The map recognises Markdown, reStructuredText and AsciiDoc headings (and so the headings the other parsers write, such as one per XSD type), numbered headings (`4.3.2`, `7.`, `7(2)(b)`) and keyword headings (`Article 12`, `Section 3`, `Annex II`). A numbered line counts as a clause only when it continues the numbering (`7.3` or `8` after `7.2`) and follows a blank line, a heading or the end of a sentence, so a numbered list inside a clause, or a wrapped line that starts with `30 days` or `Article 6`, is not one. In Markdown, a run of `1.` `2.` lines with no subclause (`1.1`) among them is an ordered list, not clauses, and adds nothing to the map. Front matter (a leading `---` block of `key: value` lines) is never read as a heading, in any format. Table-of-contents entries are skipped, whichever pattern they match, so the body's clauses keep their numbers. A chunk never crosses a section boundary (or a PDF page): a short section is one chunk, and a section longer than `chunk_size` is split, with every piece keeping the section's path. A heading with no text of its own before its first subsection (`7. Termination` directly followed by `7.1 …`) is not stored as a chunk of its own: it joins the subsection's chunk, which carries the subsection's path (`7 Termination > 7.1 …`). It stays on its own when a page starts at the subsection, so a Markdown `##` heading directly followed by a `###` one still gives a short chunk. The path of the innermost section is stored with each chunk (`clause_path`); text before the first heading has none. A document with no detectable structure is split by size alone, as before. Chunks ingested before this was added have no clause path until their file changes or the layer is rebuilt (`replace_all=True`).

```mermaid
flowchart LR
    t["Document text"] --> d{"Headings or<br/>numbered clauses?"}
    d -->|"yes"| m["Structure map:<br/>path and range per section"] --> s["Cut at every section start;<br/>a bare heading joins its first subsection;<br/>split long sections by size"] --> c["Chunks with<br/>clause_path"]
    d -->|"no"| z["Split by size"] --> c2["Chunks without<br/>clause_path"]
```

**Ask**

```mermaid
flowchart LR
    q["Question"] --> r["Route to layers"] --> rw["Rewrite<br/>(optional)"] --> h["Hybrid search per query<br/>cache, HNSW + full-text, RRF"]
    h --> d["Merge the searches<br/>by rank (RRF)"] --> rr["Cross-encoder<br/>rerank"] --> b["Keep the best passages<br/>that fit ANSWER_TOKEN_BUDGET"]
    r --> x{"Names a clause<br/>or section?"} -->|"in the structure map"| ex["That section's text,<br/>first and verbatim"] --> b
    b --> c["Cite each passage<br/>file, page, lines, headings"] --> o{"Best passage<br/>scores"}
    o -->|"high"| f["found"]
    o -->|"medium"| pf["partly_found<br/>+ next step"]
    o -->|"low or none"| nf["not_found<br/>+ next step"]
    f & pf & nf --> a["Your MCP client<br/>writes the answer"]
```

**Learn**

```mermaid
flowchart LR
    i["Asks, searches<br/>and feedback"] --> s["Signals"] --> p["Promoter<br/>every 50 interactions"]
    p --> l["Rewrites, synonyms,<br/>layer affinities"]
    l -->|"improve"| rt["Routing and rewriting"]
```

## Adding folders

Point `stratakb add` at a folder of documents. It creates one layer for the folder, named after it, and ingests what it finds:

```bash
uv run stratakb add path/to/docs                      # create the layer, or sync it
uv run stratakb add path/to/docs --name team_docs     # pick the layer name
uv run stratakb add path/to/docs --json               # the report as JSON, for scripts
```

Run the same command again after editing the documents: it re-embeds only what changed and removes files that were deleted. It follows `.gitignore` (inside a git work tree as git does; elsewhere only the folder's top-level `.gitignore`, and the report notes any nested ones it did not apply) and a `.strataignore` in the folder (same syntax), skips hidden files and folders and dependency or build folders such as `node_modules` and `dist`, and never opens files that look like secrets, code or build output, files in formats it can't parse, paths over 500 characters, or files over `--max-file-size` megabytes (default 50); the report lists every skipped file with its reason. The description comes from the folder's README unless you pass `--description`.

```mermaid
flowchart LR
    cmd["stratakb add &lt;folder&gt;"] --> plan["Plan: which layer,<br/>which files, what changed"]
    cmd -->|"STRATA_LOCAL_ONLY on and<br/>embedding leaves the machine"| stop
    plan -->|"broad path or<br/>large run"| ask{"Confirmed?<br/>prompt, --allow-broad, --yes"}
    ask -->|"no"| stop["Refuse, nothing written<br/>(exit 2)"]
    ask -->|"yes"| apply
    plan -->|"name taken, folder overlaps<br/>another layer, nothing to add"| stop
    plan -->|"otherwise"| apply["Create or sync the layer,<br/>ingest, audit"]
    apply --> report["Report and exit code"]
```

A drive root, your home folder (or a folder above it) and system folders need `--allow-broad`; a run that adds, updates or removes more than 2,000 files, adds or updates more than 500 MB, or removes at least 10 files that are more than half of the layer's files, needs `--yes`. At a terminal the command asks instead; in a script it refuses and names the flag. Exit codes: `0` done, `1` some files failed (the rest are stored), `2` refused, a usage error, or `STRATA_LOCAL_ONLY` blocks the embedding path (nothing written), `3` the database is unreachable or its schema can't be updated, `130` interrupted (stored files stay; the next run carries on). The first run with the `local` backend downloads the embedding model and says so.

`stratakb` with no command, or `stratakb serve`, starts the MCP server, as `python -m stratakb` does.

From an MCP client without a shell, either write a preset (below) or call `strata_create_knowledge_layer` followed by `strata_ingest_documents` with a folder path. Run `strata_ingest_documents` again after editing the documents: it re-embeds only what changed. Pass `replace_all=True` to rebuild the whole layer, for example after changing its `chunk_config` or the embedding model.

Upgrading: the server adds the file-tracking table to an existing database on startup. A layer ingested by an older version is rebuilt on its next ingestion, and that run keeps only the folder it ingests, so ingest each folder of such a layer once (applying a preset does this).

## Layers and presets

A preset is a YAML file that names a layer, says what it contains, and lists the folders to ingest, relative to the preset file:

```yaml
name: runbooks
display_name: Ops Runbooks
description: Incident runbooks, on-call procedures and failover guides.
chunk_config: {chunk_size: 1200, chunk_overlap: 150}
routing_patterns:            # questions matching any of these search this layer first
  - '\brunbooks?\b'
  - '\b(failover|incident|on-?call)\b'
answer_mode: grounded        # or extractive; grounded for a new layer when left out
sources:
  - name: runbooks
    path: runbooks
  - name: postmortems
    path: postmortems
    download: https://wiki.example.com/postmortems   # shown when the folder is missing
```

Apply it with `scripts/apply_preset.py` or the `strata_apply_preset` tool. Ready-made presets are in [examples/](examples/README.md): Python PEPs (the quickest to try), SEPA payments and ISO 20022 messaging.

Edit the preset and apply it again to change the layer: the layer takes the preset's `display_name`, `description`, `metadata_schema`, `chunk_config`, `routing_patterns` and `answer_mode`, the result lists the settings that changed (`layer_changes`), and the chunks are rebuilt from the sources. There is no need to delete the layer first. A layer with the same name that was not made by this preset (one made with `stratakb add`, `strata_create_knowledge_layer` or another preset) is refused and left unchanged; rename the preset's `name` or delete that layer. `strata_apply_preset` with `dry_run=true` also lists the settings that would change (`layer_check`), or `would_claim` when a layer with no recorded origin already matches the preset and would be marked as made by it, or `unknown` when the database does not answer within 5 seconds, while the script's `--dry-run` only parses the files and needs no database.

```mermaid
flowchart TD
    A[Apply preset] --> B{Layer exists?}
    B -- no --> C[Create layer from the YAML]
    B -- yes --> D{Made by}
    D -- "this preset, or no origin recorded" --> E[Update the settings that differ]
    D -- "stratakb add, manual, another preset" --> F[Refuse: nothing changed]
    C --> G[Clear chunks, ingest every source]
    E --> G
```

The layer's **description** also tells the query rewriter which terminology the layer uses, so a clear description improves `strata_ask`.

The layer's **answer_mode** says how its questions are to be answered: `grounded` (the default) means your client writes the answer from the cited passages; `extractive` is for layers where paraphrase is not acceptable, answered with the document's own sentences. Any other value is refused. `strata_list_knowledge_layers` shows each layer's mode, applying a preset again sets the mode of an existing layer when the preset gives one (a preset without `answer_mode` leaves the layer's mode alone), and layers created before this setting read as `grounded`. The server records the mode but does not act on it yet: extractive answers are on the [roadmap](#roadmap).

### Try an example preset

The repository also ships three example presets. The quickest to try is Python PEPs: git fetches eight PEPs in a few seconds, with no account:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/python/peps.git examples/peps/source
git -C examples/peps/source sparse-checkout set --no-cone /peps/pep-0008.rst /peps/pep-0020.rst /peps/pep-0257.rst /peps/pep-0484.rst /peps/pep-0572.rst /peps/pep-0634.rst /peps/pep-0695.rst /peps/pep-0723.rst
uv run python scripts/apply_preset.py examples/peps/preset.yaml --dry-run   # parse only, no database
uv run python scripts/apply_preset.py examples/peps/preset.yaml             # create the layer and ingest
```

Then ask from your client, for example *"How long should a line of Python be?"*. The SEPA preset includes a small sample of reason codes, so it works before you download anything (ask *"What does reject reason AC04 mean?"*); the ISO 20022 preset needs the downloaded schemas. [examples/README.md](examples/README.md) has the PEPs options (all PEPs, updates, Git Bash on Windows) and lists where to download the SEPA rulebooks and ISO 20022 documents.

## Tools

| Tool | Role | What it does |
| --- | --- | --- |
| `strata_ask` | viewer | Rewrites a question, searches in parallel, reranks, and returns found, partly found or not found with the best passages, each with its file, page, lines and headings ([what it returns](#what-strata_ask-returns)) |
| `strata_verify` | viewer | Checks a drafted answer against the passages it cites: every quote, number, date, amount, code and clause reference must be in them. A verified draft gets a receipt ([checking a drafted answer](#checking-a-drafted-answer)) |
| `strata_search` | viewer | Hybrid search for exact terms and identifiers, with optional layer and metadata filters, and a confidence level; a clause or section the query names (`clause 7.2`) comes first, verbatim |
| `strata_list_knowledge_layers` | viewer | Lists layers with document and chunk counts |
| `strata_get_layer_status` | viewer | One layer's configuration and ingestion history |
| `strata_get_receipt` | viewer | Opens a receipt (`rcpt_…`, also the resource `strata://receipts/{receipt_id}`): the stored quotes of an answer, each in its surrounding document text, marked current, changed (with the newer version) or gone (its file was removed from its folder or its layer was deleted). Open to whoever asked, to admins, and to anyone who may read every cited layer, for `RECEIPT_RETENTION_DAYS` (90 by default). From the command line: `stratakb receipt <id>`. Receipts are issued by `strata_verify`; extractive answers will issue them too ([roadmap](#roadmap)) |
| `strata_feedback` | viewer | Rates a result as helpful or not, per answer or per passage (`S1`) or chunk |
| `strata_learning_stats` | viewer | What the learning system has recorded and learned |
| `strata_whoami` | viewer | Your identity and role |
| `strata_admin_health` | viewer | Database, pgvector and schema status, the local-only switch, the providers that receive data, whether the SQL tools are on, and the embedding model and vector column checks |
| `strata_db_schema` | viewer | Lists tables, or a table's columns and keys |
| `strata_db_vector_stats` | viewer | Chunk counts per layer, vector dimensions and index sizes |
| `strata_db_sample_chunks` | viewer | Sample chunks from a layer |
| `strata_db_query` | operator | Read-only SQL (only with `READONLY_DATABASE_URL`) |
| `strata_db_explain` | operator | Query plan for a SQL statement (only with `READONLY_DATABASE_URL`) |
| `strata_create_knowledge_layer` | admin | Creates a layer |
| `strata_ingest_documents` | admin | Syncs a file or folder into a layer: adds, replaces and removes files; reports added, updated, unchanged, removed, skipped and flagged |
| `strata_apply_preset` | admin | Creates or updates a layer from a preset and ingests its sources |
| `strata_delete_knowledge_layer` | admin | Deletes a layer and its chunks |
| `strata_reset_learning` | admin | Clears learned patterns |
| `strata_admin_audit_log` | admin | Queries the audit log |
| `strata_admin_list_users` | admin | Lists users and roles |
| `strata_admin_update_role` | admin | Changes a user's role |

The two SQL tools never run as the owner role. They are only registered when `READONLY_DATABASE_URL` is set, and at startup the server removes them again when that role is a superuser, a member of a superuser role, a member of `pg_read_server_files`, `pg_write_server_files`, `pg_execute_server_program` or `pg_signal_backend`, or cannot be checked within 5 seconds (for example, the database is unreachable or does not answer; restart the server once it is up). `strata_admin_health` says which applies under `sql_tools`. The `strata_readonly` role can read five tables only: `strata_core.knowledge_layers`, `strata_core.ingestion_runs`, `strata_core.ingested_files`, `strata_vectors.chunks` and `strata_vectors.documents`. It cannot read the audit log, the users table or anything in `strata_learning`, and a table added later needs a `GRANT` by name. Existing databases get these grants when the server starts.

On the stdio transport, `STRATA_USER` and `STRATA_ROLE` set who you are. Whoever starts the server picks the role, so on stdio roles separate duties rather than enforce security; see [SECURITY.md](SECURITY.md).

The `stratakb add` command has no role check: whoever can reach the database with `DATABASE_URL` can add a folder. Each run writes one audit entry with the tool name `cli:add`, the folder, the run id and the counts per outcome (no file names). Its user is `STRATA_USER` when you set it, otherwise `os:` plus your login name, a name nobody verified.

`strata_search` and `strata_ask` each write one audit entry with the first 200 characters of the query or question, with IBANs, BICs and account numbers replaced by placeholders (`[IBAN_REDACTED]`, `[BIC_REDACTED]`, `[ACCT_REDACTED]`), plus the layers searched and whether routing fell back. `AUDIT_STORE_RAW_QUERIES=true` keeps the question as written; `LEARNING_STORE_RAW_QUERIES` has no effect on the audit log.

`strata_feedback` and `strata_reset_learning` each write one audit entry. A feedback entry holds the interaction id and the rating, not the suggested query or layer; a reset entry names the layer, or says it reset everything.

Each `strata_apply_preset` run writes one audit entry with the layer, whether it was created, updated or already up to date, the names of the settings that changed (never their values) and the number of documents and chunks ingested. A refused run writes none.

### What `strata_ask` returns

By default the answer is short: a status, the mode and the passages, each with its own citation, so the client can quote and cite without matching ids to a sources list.

```json
{
  "status": "found",
  "mode": "grounded",
  "passages": [
    {
      "id": "S1",
      "ref": "d42:1830-1905",
      "text": "7.2 Either party may end this agreement\nwith ninety days' written notice.",
      "citation": {
        "file": "contracts/acme-msa.md",
        "page": null,
        "lines": "30-31",
        "clause_path": "Master agreement > 7 Termination",
        "layer": "contracts"
      }
    }
  ],
  "interaction_id": "…"
}
```

- `status` is `found` when the best passage scores high, `partly_found` when it scores medium, and `not_found` when it scores low or nothing came back. The last two add a `next_step`, such as adding a folder with `stratakb add` when no layer covers the question. The thresholds are the provisional ones of [Confidence](#glossary).
- `mode` is `grounded`: the client writes the answer from the passages and cites them by id.
- A question that names a clause or section (`clause 7.2`, `Article 12`, `§ 4`, `Annex II`, or a bare `4.3.2` or `7(2)(b)`) is looked up exactly in the [structure map](#how-it-works) of the layers searched: that section's text comes back first, verbatim, with its citation and `status` `found`, cut at a word when it would pass the budget. When the same number exists in several documents, each one comes back, cited by its file. The search's own passages follow, without the parts already inside the section. A reference that matches no section (a bare `7.2` that is a version or an amount) changes nothing. `strata_search` puts the same sections first among its results, marked `"match": "exact"`. Exact passages have no chunk id, so `strata_feedback` cannot rate them by passage id yet.
- The passages come best first and fill a budget of about 1,500 tokens (`ANSWER_TOKEN_BUDGET`), each passage's id, ref and citation counted with its text; the first passage always comes back, at most 50 come back, and `top_k` (default 10) only bounds the search before that.
- Each passage's `text` is the stored document's own text. `ref` names its span in the stored document (`d<document id>:<start>-<end>`), or the chunk (`c<chunk id>`) for a chunk ingested before documents were stored. The citation has the file's path relative to the ingested folder, the page (PDFs; `page_end` too when the passage spans pages), the line range (from the page's start for PDFs) and the clause path from the structure map (for a chunk without one, such as one stored before the map existed or text before the first heading, the parser's heading path when it has one). A chunk ingested before documents were stored is cited by its file name, without lines.
- `interaction_id` goes to `strata_feedback`, which takes passage ids for every passage returned: `{"S1": "useful", "S2": "wrong"}`.
- `response_format: "detailed"` returns everything as well: `confidence`, the query variants (`queries_used`), `retrieved_chunks` with their ids, scores and metadata, `sources`, with `status` `ok` and the found, partly found or not found in `answer_status`.

### Checking a drafted answer

After your client writes an answer from the passages, it can pass the draft and the passages' `ref`s (in order, so the first is `S1`) to `strata_verify` before showing it:

```json
{
  "draft": "Either party may end the agreement with 90 days' notice [S1]: \"Either party may end this agreement with ninety days' written notice.\" [S1]",
  "passages": ["d42:1830-1905"]
}
```

```mermaid
flowchart LR
    D["Draft + refs"] --> T["Read each span<br/>from the store"]
    T --> C{Quotes and facts<br/>in the passages?}
    C -->|yes| V["verified + receipt"]
    C -->|no| F["flags: the text,<br/>and the passages searched"]
```

- The passages' text is read from the stored documents, never taken from the request. A `ref` must lie inside its document and span at most twice the layer's `chunk_size`, so a citation cannot be widened to a whole document. A `ref` that names nothing stored, or a layer you may not read, is refused with one message for both.
- **Quotes:** every quoted span (`"…" [S2]`) must appear in the passage it cites (in any passage when it cites none), after whitespace, quote marks and dashes are normalised; `...` inside a quote marks an omission.
- **Facts:** every number, date, amount, percentage, code (such as `AC01`, `pacs.008` or an IBAN) and clause reference outside the quotes must appear in at least one passage. Both sides are normalised: `30 days` matches `thirty (30) days`, `1,000.00` matches `1000`, `EUR 1.000,00` matches `1000 EUR`, and `2026-11-04` matches `4 November 2026`. An amount matches only an amount or a plain number, never a day count, clause number or year. A date such as `04/11/2026`, which could be either day or month first, only matches as written. The `1.` and `2)` of a numbered list are not facts.
- **Citations:** an `[S5]` when four passages were given is flagged.
- The result is `verified` with a `receipt_id` (open it with `strata_get_receipt`), or `flagged` with a list of flags, each naming the text, why, and the passages searched, for example `"13 days" not found in S1–S4`. At most 50 flags are listed; `more_flags` counts the rest. The receipt keeps the passages and the draft, with IBANs, BICs and account numbers in the draft replaced by placeholders.
- Both checks run in Python, without a model: nothing leaves the machine. They catch a changed quote, number, date, amount, code or clause, not a paraphrase that changes the meaning without changing a fact. For layers where that is not acceptable, use `answer_mode: extractive` ([roadmap](#roadmap)).
- Each call writes one audit entry (`verify`) with counts and, when verified, the receipt id; never the draft.

To check an answer outside the chat, open its receipt from the command line:

```bash
uv run stratakb receipt rcpt_…          # each quote marked >>> <<< in its context, with current, changed or gone
uv run stratakb receipt rcpt_… --json   # the same data as strata_get_receipt
```

It makes the same access check as `strata_get_receipt`, with `STRATA_USER` and `STRATA_ROLE` as on stdio, and writes the same audit entry with the tool name `cli:receipt`. An unknown receipt, or one you may not open, exits with code `1` and one line on stderr; an unreachable database exits with `3`. The text form shows control characters from documents, file names, questions and drafts as escapes such as `\x1b`, so a document cannot hide or rewrite what your terminal shows.

## Configuration

Settings come from four sources, each overriding the one before: the defaults below, the user-level `config.env`, the file named by `STRATAKB_CONFIG`, then environment variables. The user-level file is `%APPDATA%\stratakb\config.env` on Windows, and `$XDG_CONFIG_HOME/stratakb/config.env`, else `~/.config/stratakb/config.env`, elsewhere. Copy `.env.example` there to start.

`./.env` is not read unless you set `STRATAKB_CONFIG=.env`; when the server finds one it does not read, it logs a warning at startup. A `STRATAKB_CONFIG` that names a missing file is logged and skipped. Keys the server does not know, such as `DB_PASSWORD`, are ignored, so one file can serve both. `docker compose` still reads `./.env` itself for its own variables.

```mermaid
flowchart LR
    d["Defaults"] --> u["User config.env"] --> c["File named by<br/>STRATAKB_CONFIG"] --> e["Environment<br/>variables"]
```

| Variable | Default | Description |
| --- | --- | --- |
| `STRATAKB_CONFIG` | | A settings file read after the user-level `config.env`; environment variables still win |
| `DATABASE_URL` | `postgresql://strata:…@127.0.0.1:5433/strata` | PostgreSQL connection |
| `READONLY_DATABASE_URL` | | Connection for `strata_db_query` and `strata_db_explain`; point it at the `strata_readonly` role from `db/03_readonly_role.sql`. Empty means the two tools are off; a superuser, or a member of a superuser or server-access role (such as the Docker `strata` user) is refused |
| `STRATA_USER`, `STRATA_ROLE` | `developer`, `viewer` | Identity and role on stdio |
| `EMBEDDING_BACKEND` | `local` | `local` (sentence-transformers) or `openai` (OpenAI, or the OpenAI-compatible server in `EMBEDDING_BASE_URL`) |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Embedding model; the `openai` backend uses `text-embedding-3-large` unless set |
| `EMBEDDING_DIMENSIONS` | `768` | Must match the model and the `vector(…)` column; vectors of another length are refused, and `strata_admin_health` reports a column of another size |
| `EMBEDDING_BASE_URL` | | `openai` backend: an OpenAI-compatible server, ending in `/v1` (for example `http://127.0.0.1:11434/v1`); empty means OpenAI |
| `EMBEDDING_API_KEY` | | Key for `EMBEDDING_BASE_URL`, only if the server asks for one; a local server usually needs none (requests then carry no `Authorization` header) |
| `OPENAI_API_KEY` | | Needed for the `openai` backend without `EMBEDDING_BASE_URL`; never sent to that server |
| `QUERY_REWRITE_PROVIDER` | `anthropic` | Who rewrites questions in `strata_ask`: `anthropic`, or `openai_compatible` (the endpoint in `QUERY_REWRITE_BASE_URL`) |
| `ANTHROPIC_API_KEY` | | Turns on query rewriting with the `anthropic` provider |
| `QUERY_REWRITE_BASE_URL` | | `openai_compatible` provider: an OpenAI-compatible chat endpoint, ending in `/v1` (for example `http://127.0.0.1:11434/v1`) |
| `QUERY_REWRITE_API_KEY` | | Key for `QUERY_REWRITE_BASE_URL`, only if the endpoint asks for one; a local endpoint usually needs none (requests then carry no `Authorization` header); `OPENAI_API_KEY` is never sent there |
| `QUERY_REWRITE_MODEL` | `claude-haiku-5-5` | Model used for query rewriting; must be set for `openai_compatible` |
| `HYBRID_SEARCH_ENABLED` | `true` | Vector plus full-text search |
| `HYBRID_SEMANTIC_WEIGHT`, `HYBRID_KEYWORD_WEIGHT` | `0.7`, `0.3` | RRF weights |
| `HYBRID_RRF_K` | `60` | RRF constant |
| `HYBRID_KEYWORD_SLOTS` | `1` | Results kept for the best full-text matches, so an exact term the embedding misses still comes back (at most `top_k` − 1; `0` turns it off) |
| `HNSW_EF_SEARCH` | `40` | Recall versus latency for the vector index |
| `RERANKING_ENABLED` | `true` | Cross-encoder reranking in `strata_ask` |
| `RERANKING_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker |
| `RERANKING_CANDIDATE_MULTIPLIER` | `3` | `strata_ask` reranks `top_k` times this many candidates |
| `ANSWER_TOKEN_BUDGET` | `1500` | How much text `strata_ask` returns: the best passages that fit this many tokens (about four characters each), citations included; the first passage always comes back |
| `CACHE_ENABLED`, `CACHE_TTL_SECONDS` | `true`, `600` | Query result cache, cleared per layer on ingestion |
| `CACHE_MAXSIZE` | `500` | Query results kept in the cache |
| `LEARNING_ENABLED` | `true` | Record interactions and learn from them |
| `LEARNING_STORE_RAW_QUERIES` | `false` | Store queries without scrubbing personal data |
| `AUDIT_STORE_RAW_QUERIES` | `false` | Keep questions in audit entries without scrubbing personal data; independent of `LEARNING_STORE_RAW_QUERIES` |
| `RECEIPT_RETENTION_DAYS` | `90` | Days a receipt is kept; the server deletes older ones at startup and then once a day, with the document versions only they kept, and records each prune in the audit log. `0` keeps receipts forever |
| `LEARNING_SESSION_TIMEOUT_MINUTES` | `30` | Idle minutes that end a session for rephrase detection |
| `LEARNING_PROMOTE_EVERY_N_QUERIES` | `50` | Interactions between promotion runs |
| `LEARNING_MIN_SUCCESS_RATE` | `0.6` | Lowest success rate for a learned rewrite to be used |
| `LEARNING_REWRITE_BYPASS_THRESHOLD`, `LEARNING_REWRITE_BYPASS_MIN_SAMPLES` | `0.85`, `5` | A learned rewrite at or above this success rate and sample count skips the rewrite model |
| `STRATA_LOCAL_ONLY` | `false` | Refuse to start while any provider would send content off this machine, and load models from local files only; see [Where your data goes](#where-your-data-goes) |
| `LOG_LEVEL` | `INFO` | Structured JSON logs on stderr |

To change the embedding model, update `EMBEDDING_BACKEND`, `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` (and `EMBEDDING_BASE_URL` for a model server), then re-embed every layer at once with `vector_store.re_embed_all_layers()`. It re-embeds the stored chunk text without re-parsing files and resizes the vector column once when the dimensions change; if anything fails, every layer and the column stay as they were. It works in two phases. First it reads the chunks a batch at a time, embeds them and keeps the new vectors aside, while searches and ingests carry on as usual. When more than one batch of chunks was added in the meantime, it embeds those too, still without a lock (up to three extra passes). Then one transaction locks the chunks table, resizes the column if needed, writes the new vectors and embeds the few chunks added since. Ingests wait for that transaction, and searches wait too when it resizes the column. The model runs under the lock for at most about one batch of chunks, usually none, but writing the vectors and rebuilding the index still takes longer the more chunks there are, so plan a resize of a large knowledge base for a quiet moment. Each statement of that transaction may take up to 10 minutes (`locked_timeout=600`, in seconds); one that takes longer rolls everything back, and the error says so. Pass a larger `locked_timeout` for a very large knowledge base.

```mermaid
flowchart LR
    A["Phase 1: read a batch of chunks"] --> B["Embed it with the new model"]
    B --> C["Keep the vectors aside"]
    C -->|more chunks| A
    C -->|all done| N{"More than a batch added meanwhile?"}
    N -->|"yes, up to 3 passes"| A
    N -->|no| D["Phase 2: lock the chunks table"]
    D --> E{"Dimensions changed?"}
    E -->|yes| F["Resize the vector column"]
    E -->|no| G["Write the new vectors"]
    F --> G
    G --> H["Embed the few chunks added since"]
    H --> I["Commit"]
```

```bash
uv run python -c "import asyncio; from stratakb.config import get_settings; from stratakb.knowledge.vector_store import re_embed_all_layers; print(asyncio.run(re_embed_all_layers(get_settings().database_url)))"
```

Every layer shares the one vector column, so `vector_store.re_embed_layer()` re-embeds a single layer at the column's current size only, and refuses any other size. When `re_embed_all_layers()` resizes the column while `re_embed_layer()` is running, the single-layer run stops before its next write and says which size to set.

### Database and models

`stratakb db up` runs the container `stratakb-db` from the image `pgvector/pgvector:pg16`, with its data in the Docker volume `stratakb-data`. It binds to `127.0.0.1` only, connects as the user `strata` with the default password, and brings the schema up to date before it returns. `--port` picks another host port when the container is created; an existing container keeps its port. `stratakb db down` stops the container and keeps the volume. The container starts again with Docker until `stratakb db down` stops it. It creates no `strata_readonly` role, so the SQL tools stay off until `db/03_readonly_role.sql` is applied (see `READONLY_DATABASE_URL`). With no `docker` on `PATH`, or the Docker daemon not running, it exits with code 3 and says which.

Any empty PostgreSQL 16 database with pgvector works as well: `stratakb add` and the server create the schema on first use, as long as the user in `DATABASE_URL` may create the `vector` and `pg_trgm` extensions. Use pgvector 0.8 or later for complete results: older versions still work, but a search can return fewer results than it asks for, and the server logs `hnsw_iterative_scan_unavailable`. After upgrading the image of an existing database, run `ALTER EXTENSION vector UPDATE;`.

A schema update waits at most 30 seconds for a lock that another process holds (another StrataKB process updating the schema, or a session using a table the update changes), then stops and says which lock it waited on. A user that is neither the database owner nor a superuser can use a schema that is current; when an update is due, they are told which file to apply as the owner (`psql -f`). A user without `USAGE` on the StrataKB schemas is told which privilege is missing.

The server answers its client at once and, with the default `local` backend, loads the embedding model in the background; a search sent before the model is ready waits for it. The first start downloads the embedding model (about 420 MB) and the first `strata_ask` downloads the reranker (about 90 MB), both into your Hugging Face cache. After that, both load from the cache without contacting Hugging Face. With `STRATA_LOCAL_ONLY=true` nothing is downloaded; see [Where your data goes](#where-your-data-goes).

### Embeddings from a model server

Instead of the in-process model, StrataKB can embed through any server that speaks the OpenAI embeddings API, such as Ollama, LM Studio or vLLM. Your documents and questions then go to that server and nowhere else; no OpenAI key is needed, and a local server such as a default Ollama, llama.cpp or vLLM needs no key at all: leave `EMBEDDING_API_KEY` empty and requests carry no `Authorization` header. With `STRATA_LOCAL_ONLY=true` only a server on this machine (`127.0.0.1`, `::1`, `localhost`) is allowed. For a local Ollama:

```bash
ollama pull nomic-embed-text   # 768 dimensions, matches the default column
```

```bash
EMBEDDING_BACKEND=openai
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_MODEL=nomic-embed-text
# EMBEDDING_API_KEY=           # only if the server asks for a key
```

The URL ends in `/v1`. A model with other dimensions, for example `mxbai-embed-large` (1024), needs `EMBEDDING_DIMENSIONS` changed to match; `vector_store.re_embed_all_layers()` then resizes the `vector(…)` column (see [Configuration](#configuration)). Switching models means re-embedding every layer with it, even when the dimensions match: vectors from two models cannot be compared.

Once at startup, in the background, the server asks the server in `EMBEDDING_BASE_URL` for its model list and checks the vector column's size against `EMBEDDING_DIMENSIONS`. `strata_admin_health` reports both under `embedding` (`model` and `vector_column`), and `overall` turns `degraded` when the model is not listed or the sizes differ; the message names the setting to change. `nomic-embed-text` matches `nomic-embed-text:latest`. The check compares names only: a server that serves any requested name under one listed file name (llama.cpp's server does this) can show the model as missing while embedding works, and that report can be ignored. A server that cannot list its models, or is not up yet, shows `unchecked` and does not degrade health. When the server answers 401, the report says so and names `EMBEDDING_API_KEY` (not set, or refused). When the model is missing, the report lists at most 20 of the server's model names, each cut to 100 printable characters. After pulling a model or resizing the column, restart StrataKB to check again.

### Query rewriting through a model server

`strata_ask` can also rewrite questions through any endpoint that speaks the OpenAI chat API, such as Ollama, LM Studio, vLLM, Azure OpenAI or Groq, instead of Anthropic. Your questions and the layer descriptions then go to that endpoint; no Anthropic key is needed, and a local endpoint needs no key at all: leave `QUERY_REWRITE_API_KEY` empty and requests carry no `Authorization` header. With `STRATA_LOCAL_ONLY=true` only an endpoint on this machine (`127.0.0.1`, `::1`, `localhost`) is allowed. For a local Ollama:

```bash
ollama pull llama3.2
```

```bash
QUERY_REWRITE_PROVIDER=openai_compatible
QUERY_REWRITE_BASE_URL=http://127.0.0.1:11434/v1
QUERY_REWRITE_MODEL=llama3.2
# QUERY_REWRITE_API_KEY=       # only if the endpoint asks for a key
```

The URL ends in `/v1`, and `QUERY_REWRITE_MODEL` must name a model the endpoint serves. Small models often wrap their answer in prose or a code fence; StrataKB takes the first JSON list of strings from the reply. When a reply has none, the question is searched as written and the server logs `query_rewrite_json_error` with a running count. When the endpoint answers 401, the question is searched as written and the `query_rewrite_failed` log carries a `hint` that names `QUERY_REWRITE_API_KEY`.

## Where your data goes

By default nothing leaves this machine except what you choose to send: your documents stay in PostgreSQL, and the embedding and reranking models run in-process. Each other path is off until a setting turns it on.

```mermaid
flowchart LR
    subgraph local["This machine"]
        files["Your files"] --> server["StrataKB server"]
        models["Local embedding<br/>and reranking models"] --- server
    end
    server -->|"documents, chunks, learning, audit<br/>DATABASE_URL"| pg[("PostgreSQL")]
    server -->|"retrieved text"| client["Your MCP client<br/>and its language model"]
    server -.->|"questions and layer descriptions<br/>ANTHROPIC_API_KEY / QUERY_REWRITE_BASE_URL"| anthropic["Anthropic or a remote<br/>rewrite endpoint<br/>(query rewriting)"]
    server -.->|"documents and questions<br/>EMBEDDING_BACKEND / EMBEDDING_BASE_URL"| emb["OpenAI or a remote<br/>embedding server"]
    hf["Hugging Face"] -.->|"model files on first use,<br/>nothing about your documents"| models
```

Dotted paths are off by default and blocked by `STRATA_LOCAL_ONLY=true`.

| Path | What leaves | Turned on by | Blocked by `STRATA_LOCAL_ONLY` |
| --- | --- | --- | --- |
| PostgreSQL | Documents, chunks, stored queries, audit entries | `DATABASE_URL` (always) | No |
| Your MCP client and its language model | Retrieved text, sources | Every search and ask | No |
| Anthropic query rewriting | Questions and layer descriptions | `ANTHROPIC_API_KEY` | Yes |
| Rewrite endpoint on another host | Questions and layer descriptions | `QUERY_REWRITE_PROVIDER=openai_compatible` with `QUERY_REWRITE_BASE_URL` pointing off this machine and `QUERY_REWRITE_MODEL` set | Yes |
| Rewrite endpoint on this machine | Questions and layer descriptions, to `127.0.0.1`, `::1` or `localhost` | `QUERY_REWRITE_PROVIDER=openai_compatible` with `QUERY_REWRITE_BASE_URL` and `QUERY_REWRITE_MODEL` set | No |
| OpenAI embeddings | Documents and questions | `EMBEDDING_BACKEND=openai` without `EMBEDDING_BASE_URL` | Yes |
| Embedding server on another host | Documents and questions | `EMBEDDING_BASE_URL` pointing off this machine | Yes |
| Embedding server on this machine | Documents and questions, to `127.0.0.1`, `::1` or `localhost` | `EMBEDDING_BASE_URL` | No |
| Hugging Face downloads | Nothing about your documents; model files come in | The first start, the first `stratakb add` and the first `strata_ask` | Yes |

With `STRATA_LOCAL_ONLY=true` the server refuses to start while any blocked path is configured, and the error names each setting that caused it. The embedding calls and query rewriting check the switch again on every call, so a key added later is refused too. The Hugging Face libraries run offline and load the embedding and reranking models from local files only; a model that is not installed fails with a message that says how to install it (`uv run hf download <model>` on a machine that may download, or `EMBEDDING_MODEL` / `RERANKING_MODEL` set to a folder that holds it). The switch does not cover the database: `DATABASE_URL` may point at a team server on purpose. `stratakb add` checks the embedding path before it starts: with a blocked one configured it refuses with exit `2` (`reason` `local_only` in `--json`), names the setting and writes nothing. It does not check query rewriting, which it never uses. The OpenAI and Anthropic SDKs also read `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL`; StrataKB treats them as the destination of those paths, and `STRATA_LOCAL_ONLY` checks them too. Requests to `EMBEDDING_BASE_URL` and `QUERY_REWRITE_BASE_URL` never carry `OPENAI_API_KEY`, `OPENAI_ADMIN_KEY`, `OPENAI_ORG_ID` or `OPENAI_PROJECT_ID`.

`strata_admin_health` always lists the switch (`local_only`) and the active outbound providers (`outbound`: provider, purpose, what it sends, destination host and the setting that turns it on), whether the switch is on or not. It never shows keys, credentials, paths or ports.

With `EMBEDDING_BASE_URL` set (and `EMBEDDING_BACKEND` not `local`), the server also asks that server for its model list once at startup. The request carries `EMBEDDING_API_KEY` (no `Authorization` header when it is empty) and nothing about your documents, and with `STRATA_LOCAL_ONLY=true` it is only sent to a server on this machine.

StrataKB cannot control what your MCP client's language model does with the retrieved text. If the client uses a cloud model, the retrieved text goes to that provider. Teams that need the whole path to stay on the machine should embed through an OpenAI-compatible server on this machine (see [Embeddings from a model server](#embeddings-from-a-model-server)), rewrite questions through one too if they want rewriting (see [Query rewriting through a model server](#query-rewriting-through-a-model-server)), and use a client with a local model.

## Glossary

- **Chunk**: a passage of a document, cut along its headings or structure. Search returns chunks, each with its file and section.
- **Confidence**: `high`, `medium` or `low`, returned with every `strata_search` result and with `strata_ask` in `response_format: "detailed"`; a low one comes with a warning. `strata_ask`'s detailed `confidence`, and the one the learning loop stores, is judged on all reranked chunks before the budget trims them; its `found`, `partly_found` or `not_found` is read from the best returned passage's score on the same thresholds. It is judged on the reranker's relevance probability (0 to 1) or, with reranking off, on each chunk's `similarity` (how similar the question and the chunk are by meaning); the result says which (`basis`: `rerank` or `similarity`). Each detailed `strata_ask` chunk carries `score` (its merged rank, best 1.0), `similarity`, `rerank_score` (null with reranking off), `source_file` and `section`, and each source's `relevance_score` is the score confidence was judged on. `strata_search` results carry each chunk's `similarity` (with hybrid search on, the default, its `score` is the fused rank, where the best result is always 1.0), and `strata_search` judges its confidence on that similarity (`basis` `similarity`), never on the fused `score`; a chunk without an embedding counts as similarity 0. The learning loop stores the confidence the tool returned, and each stored result keeps its `similarity`. The thresholds are provisional until the evaluation suite measures them.
- **Embedding**: a list of numbers that stands for a chunk's meaning, so similar meanings get similar numbers. The embedding model (`all-mpnet-base-v2` by default) makes them; the vector column stores them (`EMBEDDING_DIMENSIONS`, 768 by default).
- **HNSW**: the pgvector index that finds similar embeddings quickly; `HNSW_EF_SEARCH` trades recall for speed.
- **Hybrid search**: a search by meaning (embeddings) and a search by words (PostgreSQL full-text search), merged into one list with RRF.
- **interaction_id**: the id on each search and ask result while learning is on; pass it to `strata_feedback` to rate the result.
- **Layer**: a named collection of documents, usually one folder or one domain, with its own chunking and routing rules.
- **Query rewriting**: optional; a language model turns a question into several searches in the layer's own terms. Learned rewrites are reused without the model.
- **Reranking**: in `strata_ask`, a second model (a cross-encoder, `RERANKING_MODEL`) reads the question with each candidate chunk and puts the best first.
- **Routing patterns**: a layer's regular expressions. When a question names no layers, those whose patterns match are searched first.
- **RRF** (Reciprocal Rank Fusion): how hybrid search merges its two lists by rank, weighted by `HYBRID_SEMANTIC_WEIGHT` and `HYBRID_KEYWORD_WEIGHT`, with the constant `HYBRID_RRF_K`. The best full-text matches keep `HYBRID_KEYWORD_SLOTS` places in the results, because a chunk found only by words would otherwise rank below every chunk found by meaning. `strata_ask` merges the results of its searches (one per rewritten query) the same way, by rank with `HYBRID_RRF_K`: a chunk several searches find comes first, and ties go to the more similar chunk.
- **top_k**: how many chunks a tool returns: 5 by default for `strata_search`; for `strata_ask` (10 by default) how many each search returns before `ANSWER_TOKEN_BUDGET` trims them.

## Project structure

```
db/                   03_readonly_role.sql (SQL tools role, applied with psql)
examples/             Example layer presets (Python PEPs, SEPA, ISO 20022)
scripts/              apply_preset.py, backup_restore.py
src/stratakb/
  cli.py              The stratakb command: add a folder, start a local database, or start the server
  mcp_server.py       MCP server: admin and user tools
  core/               Database pool, roles, audit log, logging, schema migrations, local database in Docker
    sql/              Schema, applied in version order by the server: 01_init, 02_learning, then the 04–12 upgrades
  knowledge/          Layers, presets, ingestion pipeline, structure map, parsers, embeddings, vector store
    add_folder.py     Turns one folder into one layer and keeps it in sync
  rag/                Retriever (hybrid + RRF), reranker, cache, routing, query rewriter, cited passages, receipts, verify, tools
  learning/           Interaction tracking, signals, synonyms, promotion, feedback tools
  database_access/    Read-only SQL and introspection tools
tests/                pytest suite
docs/IMPROVEMENTS.md  Product goal and order of work (issues hold the items)
```

## Development

```bash
docker compose up -d           # the development database (container strata-postgres, port 5433, with the read-only role)
uv sync --extra dev
uv run pytest                  # full suite; integration tests need the database running
uv run ruff check src tests    # lint
uv run python -m stratakb      # run the server on stdio (same as: uv run stratakb)
uv run stratakb add docs --name strata_docs --json   # add a folder as a layer
uv run stratakb --help         # the commands and their flags
```

Without a running database, the integration tests fail with a connection error and the rest pass. `docker compose` and `stratakb db up` both use port 5433, so run one at a time, or pass `--port` to `stratakb db up`. Both containers (`strata-postgres` and `stratakb-db`) start again with Docker unless they were stopped, so on a machine with both on 5433 they race for the port at every Docker start. Stop the one you don't use (`stratakb db down` or `docker compose stop`) and it stays stopped, or give `stratakb db up --port` another port when it creates the container.

`scripts/backup_restore.py` backs up the whole database or exports and imports a single layer, through `docker exec` in the `strata-postgres` container; `DB_CONTAINER=stratakb-db` points it at the `stratakb db up` database. A layer export holds its configuration and chunks but not its file tracking, so the first ingestion after an import rebuilds the layer.

## Roadmap

The target: ask your own documents (contracts, rulebooks, specs, runbooks) and get answers you can **prove**, without learning how retrieval works and without sending private data anywhere you did not choose. Work is grouped into [milestones](https://github.com/boudou/StrataKB/milestones):

1. **Five-minute start**: add a folder in one step, a one-command local mode, any OpenAI-compatible model provider, and a clear statement of where data goes.
2. **Answers you can prove** (planned): every claim can be checked against its source, so you can be sure the model did not make it up.
   - Each passage comes with its exact location: file, page, line range and clause (`7 Termination > 7.2`), and documents are chunked along their clauses and sections.
   - **Grounded mode** (available): your client writes the answer, and `strata_verify` checks that every quote, number, date, amount, code and clause reference in it appears in the cited passages.
   - **Extractive mode:** for layers where paraphrase is not acceptable, the server answers with the document's own sentences, chosen without a language model.
   - **Receipts:** every verified or extractive answer gets a receipt you can open outside the chat, showing the quotes in context and whether the document changed since.
   - An honest "not found" instead of a confident guess, and smaller responses that cost fewer tokens.
3. **Team server**: Streamable HTTP, real identities, per-layer access.
4. **Daily use and sync**: a command line for daily use, connectors (git first), scanned PDFs.
5. **Depth**: exact-lookup catalogs, cross-document references, source code.

```mermaid
flowchart LR
    Q[Question] --> A[strata_ask]
    A --> P["Passages with<br/>exact locations"]
    P --> G{Layer mode}
    G -->|grounded| W[Client writes answer] --> V[strata_verify] --> R[Receipt]
    G -->|extractive| X["Server returns the<br/>document's own sentences"] --> R
```

The product goal, the goals every change is checked against and the order of work are in [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

## Contributing

Bug reports, ideas and pull requests are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), and please follow the [code of conduct](CODE_OF_CONDUCT.md). Known issues and planned work are in [GitHub issues](https://github.com/boudou/StrataKB/issues), grouped by [milestone](https://github.com/boudou/StrataKB/milestones).

- **Found a bug?** [Open a bug report](https://github.com/boudou/StrataKB/issues/new?template=bug_report.md).
- **Have an idea?** [Suggest a feature](https://github.com/boudou/StrataKB/issues/new?template=feature_request.md).
- **Security issue?** Report it privately, as described in [SECURITY.md](SECURITY.md).

Changes are listed in the [changelog](CHANGELOG.md).

## Tech stack

Python 3.12, the MCP Python SDK, PostgreSQL 16 with pgvector, asyncpg, sentence-transformers, LangChain text splitters, pypdf, python-docx, lxml, BeautifulSoup, structlog, Pydantic.

## License

[MIT](LICENSE) © [boudou](https://github.com/boudou)
