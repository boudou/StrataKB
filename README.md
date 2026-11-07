<div align="center">

# StrataKB

**Answers from your own documents that you can check, with the exact place in the source.**
Point it at a folder and ask from any MCP client.

[![CI](https://github.com/boudou/StrataKB/actions/workflows/ci.yml/badge.svg)](https://github.com/boudou/StrataKB/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-informational)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick start](#quick-start) · [Answers you can prove](#answers-you-can-prove) · [Where your data goes](#where-your-data-goes) · [Layers and presets](#layers-and-presets) · [Documentation](#documentation) · [Roadmap](#roadmap)

</div>

StrataKB turns folders of your documents into a knowledge base that your AI tools can ask through the Model Context Protocol (MCP). Each folder becomes a **layer**: a searchable collection named after the folder.

StrataKB finds the passages that answer a question and names the files they come from; your MCP client writes the answer from them, or, on a layer set to extractive, the server answers with the document's own sentences.

**Today it is for one developer or a small trusted group.** The server runs on your machine over stdio, and each person says who they are in their own settings, so roles separate duties and are not a security boundary. A team server with real identities and per-layer access is planned ([roadmap](#roadmap)).

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

Ask your client a question about the documents, for example *"How do we fail over the database?"*. The client calls StrataKB's `strata_ask` tool, gets back the passages that match and the files they come from, and writes the answer. The server tells the client to answer only from those passages, cite them and check the answer, and a checked answer comes with a receipt you can open with `uv run stratakb receipt <id>`.

No documents at hand? Try the [Python PEPs example](#try-an-example-preset).

How it works, every tool and every setting are in the [wiki](https://github.com/boudou/StrataKB/wiki); the terms it uses are in its [Glossary](https://github.com/boudou/StrataKB/wiki/Glossary).

## What works today

- **Add a folder**: one command reads PDF, DOCX, Markdown, HTML, plain text, reStructuredText, AsciiDoc, CSV (up to 10,000 rows per file), JSON, XML/XSD and OpenAPI/AsyncAPI YAML, and later runs read only what changed.
- **Cited passages**: every passage comes with its file, page, lines, headings and date.
- **Answers you can check**: `strata_verify` checks the quotes and facts in a drafted answer, extractive answers use the document's own sentences, and both give a receipt you can open outside the chat. See [Answers you can prove](#answers-you-can-prove).
- **Layers, routing and presets**: a layer per folder or domain, questions routed to the right layers by each layer's patterns, and one YAML file to build or rebuild a layer. See [Layers and presets](#layers-and-presets).
- **Local embeddings and a local-only switch**: `all-mpnet-base-v2` runs on CPU with no API key, and `STRATA_LOCAL_ONLY=true` refuses to start while any setting would send content off this machine. See [Where your data goes](#where-your-data-goes).
- **Privacy guards**: IBANs and card numbers are masked in logs, IBANs, BICs and account numbers are scrubbed from stored queries and audited questions, secrets are redacted from audit entries, and files whose names look like secrets are never read.

```mermaid
flowchart LR
    client["MCP client"] <-->|"stdio"| server["StrataKB server"]
    server <--> pg[("PostgreSQL + pgvector<br/>layers, chunks, learning, audit")]
```

## Answers you can prove

Every claim in an answer can be checked against the document it came from. How the answer is written depends on the mode, which the layer's `answer_mode` chooses:

- **Grounded** (the default): your client writes the answer from the passages, cites them by id and checks the draft with `strata_verify`. The response is [the cited passages](https://github.com/boudou/StrataKB/wiki/Asking-and-Searching#what-strata_ask-returns), and [the check](https://github.com/boudou/StrataKB/wiki/Asking-and-Searching#checking-a-drafted-answer) issues a receipt when it passes.
- **Extractive**: for text that must not be paraphrased, the server [answers with the document's own sentences](https://github.com/boudou/StrataKB/wiki/Asking-and-Searching#extractive-answers) and a receipt, and your client shows them as they are.

```mermaid
flowchart TD
    Q[Question] --> A[strata_ask]
    A --> S{status}
    S -->|not_found| N["Say: not in the documents"]
    S -->|"found, partly_found"| M{mode}
    M -->|grounded| W["Client writes the answer<br/>from the passages, cites [S1]"] --> V[strata_verify]
    V -->|flagged| W
    V -->|verified| R[Receipt]
    V -->|nothing_checked| U["Quote a passage and verify again,<br/>or say: unverified"]
    M -->|extractive| X["Server returns the document's<br/>own sentences, shown as they are"] --> R
```

**The answer contract** is what the server tells the client's model to do, in the instructions it sends when a client connects:

- Answer only from the passages `strata_ask` returns, never from its own knowledge.
- Cite every claim by passage id in the sentence that makes it, since a sentence's facts are checked against only the passages it cites, and copy numbers, dates, amounts, codes and clause numbers verbatim.
- In grounded mode, call `strata_verify` before showing the answer, and fix or drop what it flags. A flag of kind `passage` (a span too wide to check, for example) cannot be fixed by rewriting, so the client says the answer is unverified. When the result is `nothing_checked`, the draft has no quote or fact to check and no receipt: the client quotes the passage that supports it and verifies again, or says the answer is unverified.
- In extractive mode, show `answer` verbatim, with each sentence's citation and any note or warning.
- Show the receipt id when there is one. A `not_found`, a flagged draft, a `nothing_checked` draft and a grounded answer not yet verified have none.
- Use each citation's date to tell which document is newer, and `date_source` to tell whether it came from the document's front matter or the file's modification time, which can be a checkout or a copy. A date does not show that documents conflict. The client gives it beside the answer, not in the draft it verifies, because `strata_verify` checks only passage text.
- When a citation has `replaced_by`, tell the reader which file says it replaces this one. The client gives it beside the answer, not in the draft it verifies. It is only what a document declares: no label does not mean a passage is current.
- When a citation has `archived`, tell the reader the file is marked as archived. The client gives it beside the answer, not in the draft it verifies. It means a layer's patterns matched the path, says nothing of what replaces the file, and no label does not mean a passage is current.
- When the status is `not_found`, say "not in the documents" and give `next_step`, even if passages came back. When it is `partly_found`, answer what the passages support and say what they leave open. When it is `error`, report the error.

Clients that show MCP prompts also offer the same steps as the prompt `answer_from_documents`, with a `question` and optional comma-separated `layers`. The prompt reads no data and needs no database; the tools it names make their usual role checks.

**The limits.** The instructions are a request to the client, not something the server can enforce: a client can ignore them, skip the check or retype a quote. What you can rely on is the receipt, which holds what the server itself read from the documents and, for a checked answer, the draft it checked. `strata_verify` checks quotes, numbers, dates, amounts, codes and clause references. A draft with citations but no quotes or facts returns `nothing_checked`, because the check had nothing to compare. The check does not catch a paraphrase that changes the meaning without changing a fact; for layers where that is not acceptable, use `answer_mode: extractive`.

**What a date means.** Each citation has a `date`, as `YYYY-MM-DD`, and a `date_source` that says where it came from. With `front_matter` it is the document's own `updated` key or, without one, its `date` key. With `file` it is the file's modification time as a UTC date, which can show a checkout or a copy rather than an edit. A date shows which document is newer, not that two documents conflict. A document stored before dates were recorded shows `null` until its folder is ingested again, and one stored before document text was kept shows `null` until the file changes or the layer is rebuilt.

**What `replaced_by` means.** A document says it replaces another with a `supersedes` key in its front matter, naming one file relative to its own folder, like a Markdown link: `supersedes: failover-2024.md`. A citation of the replaced file then has `replaced_by`, the path of the file that says so; other citations do not have the key. The label changes no ranking, so the replaced file's passages still come back. It is only a declaration: anyone who can add a file to the folder can label another file. PDF and Word files can be named by the key but cannot declare it. One key names one file, and when several files name the same one the citation gives the first by path. A chain names only the next file. The path must match exactly, letter case too, and within one layer and source root, so it does not resolve across a preset's source groups, which have separate roots. A file that held the key before it was recorded takes effect when the file changes or the layer is rebuilt. `strata_search` has no citations, and a chunk stored before document text was kept cites no document, so neither is labelled. A grounded passage's label is not counted against the token budget, so an answer can pass it by the label's length; an extractive answer counts it.

**What `archived` means.** A citation has `"archived": true` when the file's path matches its layer's archived patterns; other citations do not have the key. It is worked out on every ask, so a change to the patterns needs no new ingest. The label changes no ranking, so the file's passages still come back. It only means the patterns matched: it does not say what replaces the file, and a file without it may still be out of date. A citation can have both `archived` and `replaced_by`. Viewers see the label but not the patterns. A passage stored before document text was kept has no label until the file changes or the layer is rebuilt.

**Opening a receipt.** Use `strata_get_receipt`, the resource `strata://receipts/{receipt_id}` or the command line, `uv run stratakb receipt <id>` ([what it prints](https://github.com/boudou/StrataKB/wiki/Asking-and-Searching#checking-a-drafted-answer)). Each quote appears in its surrounding text, marked current, changed or gone. Receipts are kept for `RECEIPT_RETENTION_DAYS` (90 by default).

## Where your data goes

By default nothing leaves this machine except what you choose to send: your documents stay in PostgreSQL, and the embedding and reranking models run in-process. Each other path is off until a setting turns it on, except the model download from Hugging Face on first use, which sends nothing about your documents.

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

Dotted paths are blocked by `STRATA_LOCAL_ONLY=true`. The rewrite and embedding paths are off by default; the Hugging Face download runs by default when a model is not in the cache.

| Path | What leaves | Turned on by | Blocked by `STRATA_LOCAL_ONLY` |
| --- | --- | --- | --- |
| PostgreSQL | Documents, chunks, stored queries, audit entries | `DATABASE_URL` (always) | No |
| Your MCP client and its language model | Retrieved text, sources | Every search and ask | No |
| Anthropic query rewriting | Questions and layer descriptions | `ANTHROPIC_API_KEY` | Yes |
| Rewrite endpoint on another host | Questions and layer descriptions | `QUERY_REWRITE_PROVIDER=openai_compatible` with `QUERY_REWRITE_BASE_URL` pointing off this machine and `QUERY_REWRITE_MODEL` set | Yes |
| Rewrite endpoint on this machine | Questions and layer descriptions, to `127.0.0.1`, `::1` or `localhost` | `QUERY_REWRITE_PROVIDER=openai_compatible` with `QUERY_REWRITE_BASE_URL` and `QUERY_REWRITE_MODEL` set | No |
| OpenAI embeddings | Documents and questions | `EMBEDDING_BACKEND=openai` without `EMBEDDING_BASE_URL` | Yes |
| Embedding server on another host | Documents and questions | `EMBEDDING_BACKEND=openai` with `EMBEDDING_BASE_URL` pointing off this machine | Yes |
| Embedding server on this machine | Documents and questions, to `127.0.0.1`, `::1` or `localhost` | `EMBEDDING_BACKEND=openai` with `EMBEDDING_BASE_URL` on this machine | No |
| Hugging Face downloads | Nothing about your documents; model files come in | The first start, the first `stratakb add` and the first `strata_ask` | Yes |

With `STRATA_LOCAL_ONLY=true` the server refuses to start while any blocked path is configured, and the error names each setting that caused it. `strata_search`, `strata_ask` and query rewriting use the embedding, search, rewrite and answer settings the server started with: editing those in a config file while the server runs changes nothing for them until you restart it. Three things are read from the config files later: `STRATA_USER` and `STRATA_ROLE` on every tool call, the cache settings when the cache is first built, and the switch itself when a model has to be loaded and is not in the cache. `strata_ingest_documents` and `strata_apply_preset` still read the config files on each call and check the switch there, so a key added later is refused. The Hugging Face libraries run offline and load the embedding and reranking models from local files only; a model that is not installed fails with a message that says how to install it (`uv run hf download <model>` on a machine that may download, or `EMBEDDING_MODEL` / `RERANKING_MODEL` set to a folder that holds it). The switch does not cover the database: `DATABASE_URL` may point at a shared database on purpose. `stratakb add` checks the embedding path before it starts: with a blocked one configured it refuses with exit `2` (`reason` `local_only` in `--json`), names the setting and writes nothing. It does not check query rewriting, which it never uses. The OpenAI and Anthropic SDKs also read `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL`; StrataKB treats them as the destination of those paths, and `STRATA_LOCAL_ONLY` checks them too. Requests to `EMBEDDING_BASE_URL` and `QUERY_REWRITE_BASE_URL` never carry `OPENAI_API_KEY`, `OPENAI_ADMIN_KEY`, `OPENAI_ORG_ID` or `OPENAI_PROJECT_ID`.

`strata_admin_health` always lists the switch (`local_only`) and the active outbound providers (`outbound`: provider, purpose, what it sends, destination host and the setting that turns it on), whether the switch is on or not. It never shows keys, credentials, paths or ports. The report shows the settings the server started with.

With `EMBEDDING_BASE_URL` set (and `EMBEDDING_BACKEND` not `local`), the server also asks that server for its model list once at startup. The request carries `EMBEDDING_API_KEY` (no `Authorization` header when it is empty) and nothing about your documents, and with `STRATA_LOCAL_ONLY=true` it is only sent to a server on this machine.

StrataKB cannot control what your MCP client's language model does with the retrieved text. If the client uses a cloud model, the retrieved text goes to that provider. To keep the whole path on the machine, embed through an OpenAI-compatible server on this machine (see [Embeddings from a model server](https://github.com/boudou/StrataKB/wiki/Connecting-a-Client#embeddings-from-a-model-server)), rewrite questions through one too if you want rewriting (see [Query rewriting](https://github.com/boudou/StrataKB/wiki/Asking-and-Searching#query-rewriting)), and use a client with a local model.

## Layers and presets

Make a layer per folder or domain. A preset is a YAML file that names a layer and lists its folders; apply it with `scripts/apply_preset.py` or the `strata_apply_preset` tool. Ready-made presets are in [examples/](examples/README.md), and every key is described in the [wiki](https://github.com/boudou/StrataKB/wiki/Layers-and-Presets#presets).

**Archived files.** A layer can hold a list of path patterns saying which of its files are archived, and citations of those files then say so ([what the label means](#answers-you-can-prove)). `strata_set_archived_patterns` (admin) sets them, in `.gitignore` style, at most three wildcards each; an empty list clears them. A preset sets them with `archived_patterns: ['archive/', '*-2024.md', '!keep.md']`, quoted because YAML reads a leading `*` or `!` as its own syntax. A preset without the key leaves the layer's patterns alone, and `[]` clears them. Each pattern is matched against the file's path relative to the folder ingested, letter case included, so in a preset with several source groups `rulebooks/archive/` matches nothing: write `archive/`. This is not the layer's `archived` status, which covers the whole layer.

### Try an example preset

The repository also ships three example presets. The quickest to try is Python PEPs: git fetches a handful of PEPs in a few seconds, with no account:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/python/peps.git examples/peps/source
git -C examples/peps/source sparse-checkout set --no-cone /peps/pep-0008.rst /peps/pep-0020.rst /peps/pep-0257.rst /peps/pep-0484.rst /peps/pep-0572.rst /peps/pep-0634.rst /peps/pep-0695.rst /peps/pep-0723.rst
uv run python scripts/apply_preset.py examples/peps/preset.yaml --dry-run   # parse only, no database
uv run python scripts/apply_preset.py examples/peps/preset.yaml             # create the layer and ingest
```

Then ask from your client, for example *"How long should a line of Python be?"*. The SEPA preset includes a small sample of reason codes, so it works before you download anything (ask *"What does reject reason AC04 mean?"*); the ISO 20022 preset needs the downloaded schemas. [examples/README.md](examples/README.md) has the PEPs options (all PEPs, updates, Git Bash on Windows) and lists where to download the SEPA rulebooks and ISO 20022 documents.

## Roles and audit

StrataKB has viewer, operator and admin roles. `STRATA_USER` and `STRATA_ROLE` are self-declared, so whoever starts the server picks the role: the Quick start sets `admin` because that one person runs everything. Roles separate duties and do not enforce security; see [SECURITY.md](SECURITY.md).

The audit log records searches and changes under that declared name. The `strata_readonly` role can read five tables only: `strata_core.knowledge_layers`, `strata_core.ingestion_runs`, `strata_core.ingested_files`, `strata_vectors.chunks` and `strata_vectors.documents`. It cannot read the audit log, the users table or anything in `strata_learning`.

## Documentation

The reference is in the [wiki](https://github.com/boudou/StrataKB/wiki):

| Page | What it holds |
| --- | --- |
| [Home](https://github.com/boudou/StrataKB/wiki/Home) | Features and how the parts fit together |
| [Installation](https://github.com/boudou/StrataKB/wiki/Installation) | Install, other PostgreSQL servers, the schema, development setup |
| [Connecting a Client](https://github.com/boudou/StrataKB/wiki/Connecting-a-Client) | Client configuration, embeddings from a model server |
| [Configuration](https://github.com/boudou/StrataKB/wiki/Configuration) | Every setting and which source wins |
| [Tools](https://github.com/boudou/StrataKB/wiki/Tools) | Every tool, its role and what it does |
| [Layers and Presets](https://github.com/boudou/StrataKB/wiki/Layers-and-Presets) | Adding folders, supported formats, document structure, every preset key, moving a layer |
| [Asking and Searching](https://github.com/boudou/StrataKB/wiki/Asking-and-Searching) | What `strata_ask` returns, extractive answers, checking a draft, query rewriting, feedback and learning |
| [Roles and Audit](https://github.com/boudou/StrataKB/wiki/Roles-and-Audit) | Roles, the audit log, read-only SQL |
| [Troubleshooting](https://github.com/boudou/StrataKB/wiki/Troubleshooting) | Common problems and what to do |
| [Glossary](https://github.com/boudou/StrataKB/wiki/Glossary) | The terms these pages use |

## Roadmap

The target: ask your own documents (contracts, rulebooks, specs, runbooks) and get answers you can **prove**, without learning how retrieval works and without sending private data anywhere you did not choose. Work is grouped into [milestones](https://github.com/boudou/StrataKB/milestones):

1. **Five-minute start**: add a folder in one step, a one-command local mode, any OpenAI-compatible model provider, and a clear statement of where data goes.
2. **[Answers you can prove](#answers-you-can-prove)** (available): every claim can be checked against its source. Still to come: an evaluation suite that measures the provisional not-found thresholds, and structured tool output with failures reported as real tool errors.
3. **Team server** (planned): Streamable HTTP, real identities, per-layer access.
4. **Daily use and sync**: a command line for daily use, connectors (git first), scanned PDFs.
5. **Depth**: exact-lookup catalogs, cross-document references, source code.

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
