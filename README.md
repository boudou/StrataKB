<div align="center">

# StrataKB

**A layered, self-learning knowledge base served over the Model Context Protocol.**
Point it at your documentation, organise it in layers, and query it from any MCP client.

[![CI](https://github.com/boudou/StrataKB/actions/workflows/ci.yml/badge.svg)](https://github.com/boudou/StrataKB/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-informational)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick start](#quick-start) · [Layers and presets](#layers-and-presets) · [Tools](#tools) · [Configuration](#configuration) · [Roadmap](#roadmap)

</div>

StrataKB is a retrieval server for teams. It keeps documents in **knowledge layers**, one per domain (a standard, a product's docs, a team's runbooks), and serves them to MCP clients with hybrid search, reranking and source citations. It runs on PostgreSQL with pgvector, so one instance can serve a whole team, with roles and an audit log.

The server retrieves; it does not write answers. Your MCP client reads the returned context and composes the answer itself.

## Features

- **Knowledge layers**: create a layer per domain at runtime, each with its own chunking and routing rules; no code changes to add a new domain
- **Layer presets**: describe a layer and its source folders in one YAML file, then build or rebuild it in one call
- **Hybrid retrieval**: pgvector HNSW similarity plus PostgreSQL full-text search, fused with Reciprocal Rank Fusion, then reranked by a cross-encoder
- **Query routing**: when no layers are given, questions are routed to the right layers by each layer's patterns
- **Query rewriting** (optional): with an Anthropic API key, `strata_ask` expands a question into several searches in the knowledge base's own terminology
- **Learns from use**: records searches, detects rephrased questions and layer switches, and takes explicit feedback on each result's `interaction_id`. It then learns synonyms, query rewrites that worked (a proven rewrite skips the LLM call) and which layers a question belongs to
- **Many formats**: PDF, DOCX, Markdown, HTML, plain text, reStructuredText, AsciiDoc, CSV, JSON, XML/XSD and OpenAPI/AsyncAPI YAML, each split along its natural structure
- **Local embeddings by default**: `all-mpnet-base-v2` runs on CPU with no API key; OpenAI or any OpenAI-compatible server (Ollama, LM Studio, vLLM) is optional
- **Roles and audit**: viewer, operator and admin roles on every tool, and an append-only audit log of searches and changes
- **Local-only switch**: `STRATA_LOCAL_ONLY=true` refuses to start while any setting would send content off this machine, and keeps model downloads off; see [Where your data goes](#where-your-data-goes)
- **Privacy guards**: IBANs and card numbers are masked in logs, and IBANs, BICs and account numbers are scrubbed from stored queries; secrets are redacted from audit entries; files whose names look like secrets (`.env`, keys, certificates, credentials) are never read or ingested
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
    start["python -m stratakb"] --> db["Connect to PostgreSQL,<br/>update the schema"] --> ro{"READONLY_DATABASE_URL<br/>set, and a safe role?"}
    ro -->|"yes"| ready["Answer the client"]
    ro -->|"no"| off["Drop strata_db_query<br/>and strata_db_explain"] --> ready
    db -.->|"background"| m["Load the local embedding model<br/>(local backend only)"] -.-> s["First search<br/>(waits if still loading)"]
```

**Ingest**

```mermaid
flowchart LR
    f["Files"] --> n{"Secret-looking<br/>name?"}
    n -->|"yes"| k["Skip, never opened"]
    n -->|"no"| h{"Content<br/>changed?"}
    h -->|"no"| s["Skip"]
    h -->|"new or changed"| p["Parse<br/>per format"]
    p -->|"empty"| t["No text:<br/>skip or remove"]
    p --> c["Chunk"] --> x["Prefix<br/>[layer > file > section]"] --> e["Embed"] --> v[("pgvector<br/>replace that file's chunks")]
```

Ingesting a folder again is a sync: unchanged files are skipped, changed files are replaced, new files are added, and files deleted from that folder are removed. Other folders in the same layer are left alone. Files whose names look like secrets (`.env`, `client_secret.json`, `id_rsa`, `*.pem`) are skipped without being opened, and one ingested earlier is removed. Empty files are reported as no text rather than as failures. A file that fails to parse is reported and the rest of the run carries on.

**Ask**

```mermaid
flowchart LR
    q["Question"] --> r["Route to layers"] --> rw["Rewrite<br/>(optional)"] --> h["Hybrid search per query<br/>cache, HNSW + full-text, RRF"]
    h --> d["Deduplicate"] --> rr["Cross-encoder<br/>rerank"] --> o["Chunks, sources,<br/>confidence"]
    o --> a["Your MCP client<br/>writes the answer"]
```

**Learn**

```mermaid
flowchart LR
    i["Asks, searches<br/>and feedback"] --> s["Signals"] --> p["Promoter<br/>every 50 interactions"]
    p --> l["Rewrites, synonyms,<br/>layer affinities"]
    l -->|"improve"| rt["Routing and rewriting"]
```

## Quick start

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/) and [Git](https://git-scm.com/). uv installs Python 3.12 if you don't have it.

```bash
git clone https://github.com/boudou/StrataKB.git
cd StrataKB
docker compose up -d        # PostgreSQL 16 + pgvector on port 5433, schemas created on first start
uv sync
```

The server answers its client at once and, with the default `local` backend, loads the embedding model in the background; a search sent before the model is ready waits for it. The first start downloads the embedding model (about 420 MB) and the first `strata_ask` downloads the reranker (about 90 MB), both into your Hugging Face cache. With `STRATA_LOCAL_ONLY=true` nothing is downloaded; see [Where your data goes](#where-your-data-goes).

### Connect an MCP client

Add the server to your client's MCP configuration (for example a project `.mcp.json`):

```json
{
  "mcpServers": {
    "stratakb": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/StrataKB", "python", "-m", "stratakb"],
      "env": {
        "DATABASE_URL": "postgresql://strata:strata_dev_password@localhost:5433/strata",
        "STRATA_USER": "developer",
        "STRATA_ROLE": "admin"
      }
    }
  }
}
```

### Build your first layer

The repository ships two example presets, SEPA payments and ISO 20022 messaging. The SEPA preset includes a small sample of reason codes, so it works before you download anything:

```bash
uv run python scripts/apply_preset.py examples/sepa/preset.yaml --dry-run   # parse only, no database
uv run python scripts/apply_preset.py examples/sepa/preset.yaml             # create the layer and ingest
```

Then ask from your client, for example *"What does reject reason AC04 mean?"*. [examples/README.md](examples/README.md) lists where to download the full rulebooks and ISO 20022 schemas.

For your own documents, either write a preset (below) or call `strata_create_knowledge_layer` followed by `strata_ingest_documents` with a folder path. Run `strata_ingest_documents` again after editing the documents: it re-embeds only what changed. Pass `replace_all=True` to rebuild the whole layer, for example after changing its `chunk_config` or the embedding model.

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
sources:
  - name: runbooks
    path: runbooks
  - name: postmortems
    path: postmortems
    download: https://wiki.example.com/postmortems   # shown when the folder is missing
```

Apply it with `scripts/apply_preset.py` or the `strata_apply_preset` tool. Applying a preset again rebuilds the layer from scratch.

The layer's **description** also tells the query rewriter which terminology the layer uses, so a clear description improves `strata_ask`.

## Tools

| Tool | Role | What it does |
| --- | --- | --- |
| `strata_ask` | viewer | Rewrites a question, searches in parallel, reranks, returns context with citations and a confidence level |
| `strata_search` | viewer | Hybrid search for exact terms and identifiers, with optional layer and metadata filters |
| `strata_list_knowledge_layers` | viewer | Lists layers with document and chunk counts |
| `strata_get_layer_status` | viewer | One layer's configuration and ingestion history |
| `strata_feedback` | viewer | Rates a result as helpful or not, per answer or per chunk |
| `strata_learning_stats` | viewer | What the learning system has recorded and learned |
| `strata_whoami` | viewer | Your identity and role |
| `strata_admin_health` | viewer | Database, pgvector and schema status, the local-only switch, the providers that receive data, and whether the SQL tools are on |
| `strata_db_schema` | viewer | Lists tables, or a table's columns and keys |
| `strata_db_vector_stats` | viewer | Chunk counts per layer, vector dimensions and index sizes |
| `strata_db_sample_chunks` | viewer | Sample chunks from a layer |
| `strata_db_query` | operator | Read-only SQL (only with `READONLY_DATABASE_URL`) |
| `strata_db_explain` | operator | Query plan for a SQL statement (only with `READONLY_DATABASE_URL`) |
| `strata_create_knowledge_layer` | admin | Creates a layer |
| `strata_ingest_documents` | admin | Syncs a file or folder into a layer: adds, replaces and removes files; reports added, updated, unchanged, removed and skipped |
| `strata_apply_preset` | admin | Creates a layer from a preset and ingests its sources |
| `strata_delete_knowledge_layer` | admin | Deletes a layer and its chunks |
| `strata_reset_learning` | admin | Clears learned patterns |
| `strata_admin_audit_log` | admin | Queries the audit log |
| `strata_admin_list_users` | admin | Lists users and roles |
| `strata_admin_update_role` | admin | Changes a user's role |

The two SQL tools never run as the owner role. They are only registered when `READONLY_DATABASE_URL` is set, and at startup the server removes them again when that role is a superuser, a member of a superuser role, a member of `pg_read_server_files`, `pg_write_server_files`, `pg_execute_server_program` or `pg_signal_backend`, or cannot be checked (for example, the database is unreachable; restart the server once it is up). `strata_admin_health` says which applies under `sql_tools`. The `strata_readonly` role can read four tables only: `strata_core.knowledge_layers`, `strata_core.ingestion_runs`, `strata_core.ingested_files` and `strata_vectors.chunks`. It cannot read the audit log, the users table or anything in `strata_learning`, and a table added later needs a `GRANT` by name. Existing databases get these grants when the server starts.

On the stdio transport, `STRATA_USER` and `STRATA_ROLE` set who you are. Whoever starts the server picks the role, so on stdio roles separate duties rather than enforce security; see [SECURITY.md](SECURITY.md).

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
| `DATABASE_URL` | `postgresql://strata:…@localhost:5433/strata` | PostgreSQL connection |
| `READONLY_DATABASE_URL` | | Connection for `strata_db_query` and `strata_db_explain`; point it at the `strata_readonly` role from `db/03_readonly_role.sql`. Empty means the two tools are off; a superuser, or a member of a superuser or server-access role (such as the Docker `strata` user) is refused |
| `STRATA_USER`, `STRATA_ROLE` | `developer`, `viewer` | Identity and role on stdio |
| `EMBEDDING_BACKEND` | `local` | `local` (sentence-transformers) or `openai` (OpenAI, or the OpenAI-compatible server in `EMBEDDING_BASE_URL`) |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Embedding model; the `openai` backend uses `text-embedding-3-large` unless set |
| `EMBEDDING_DIMENSIONS` | `768` | Must match the model and the `vector(…)` column; vectors of another length are refused |
| `EMBEDDING_BASE_URL` | | `openai` backend: an OpenAI-compatible server, ending in `/v1` (for example `http://127.0.0.1:11434/v1`); empty means OpenAI |
| `EMBEDDING_API_KEY` | | Key for `EMBEDDING_BASE_URL`, only if the server asks for one |
| `OPENAI_API_KEY` | | Needed for the `openai` backend without `EMBEDDING_BASE_URL`; never sent to that server |
| `ANTHROPIC_API_KEY` | | Turns on query rewriting in `strata_ask` |
| `QUERY_REWRITE_MODEL` | `claude-haiku-5-5` | Model used for query rewriting |
| `HYBRID_SEARCH_ENABLED` | `true` | Vector plus full-text search |
| `HYBRID_SEMANTIC_WEIGHT`, `HYBRID_KEYWORD_WEIGHT` | `0.7`, `0.3` | RRF weights |
| `HYBRID_RRF_K` | `60` | RRF constant |
| `HNSW_EF_SEARCH` | `40` | Recall versus latency for the vector index |
| `RERANKING_ENABLED` | `true` | Cross-encoder reranking in `strata_ask` |
| `RERANKING_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker |
| `CACHE_ENABLED`, `CACHE_TTL_SECONDS` | `true`, `600` | Query result cache, cleared per layer on ingestion |
| `LEARNING_ENABLED` | `true` | Record interactions and learn from them |
| `LEARNING_STORE_RAW_QUERIES` | `false` | Store queries without scrubbing personal data |
| `STRATA_LOCAL_ONLY` | `false` | Refuse to start while any provider would send content off this machine, and load models from local files only; see [Where your data goes](#where-your-data-goes) |
| `LOG_LEVEL` | `INFO` | Structured JSON logs on stderr |

To change the embedding model, update the three `EMBEDDING_*` settings and re-embed each layer with `vector_store.re_embed_layer()`, which re-embeds the stored chunk text in place without re-parsing files.

### Embeddings from a model server

Instead of the in-process model, StrataKB can embed through any server that speaks the OpenAI embeddings API, such as Ollama, LM Studio or vLLM. Your documents and questions then go to that server and nowhere else; no OpenAI key is needed. With `STRATA_LOCAL_ONLY=true` only a server on this machine (`127.0.0.1`, `::1`, `localhost`) is allowed. For a local Ollama:

```bash
ollama pull nomic-embed-text   # 768 dimensions, matches the default column
```

```bash
EMBEDDING_BACKEND=openai
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_MODEL=nomic-embed-text
# EMBEDDING_API_KEY=           # only if the server asks for a key
```

The URL ends in `/v1`. A model with other dimensions, for example `mxbai-embed-large` (1024), needs `EMBEDDING_DIMENSIONS` and the `vector(…)` column changed to match. Switching models means re-embedding every layer with `vector_store.re_embed_layer()`, even when the dimensions match: vectors from two models cannot be compared.

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
    server -.->|"questions and layer descriptions<br/>ANTHROPIC_API_KEY"| anthropic["Anthropic<br/>(query rewriting)"]
    server -.->|"documents and questions<br/>EMBEDDING_BACKEND / EMBEDDING_BASE_URL"| emb["OpenAI or a remote<br/>embedding server"]
    hf["Hugging Face"] -.->|"model files on first use,<br/>nothing about your documents"| models
```

Dotted paths are off by default and blocked by `STRATA_LOCAL_ONLY=true`.

| Path | What leaves | Turned on by | Blocked by `STRATA_LOCAL_ONLY` |
| --- | --- | --- | --- |
| PostgreSQL | Documents, chunks, stored queries, audit entries | `DATABASE_URL` (always) | No |
| Your MCP client and its language model | Retrieved text, sources | Every search and ask | No |
| Anthropic query rewriting | Questions and layer descriptions | `ANTHROPIC_API_KEY` | Yes |
| OpenAI embeddings | Documents and questions | `EMBEDDING_BACKEND=openai` without `EMBEDDING_BASE_URL` | Yes |
| Embedding server on another host | Documents and questions | `EMBEDDING_BASE_URL` pointing off this machine | Yes |
| Embedding server on this machine | Documents and questions, to `127.0.0.1`, `::1` or `localhost` | `EMBEDDING_BASE_URL` | No |
| Hugging Face downloads | Nothing about your documents; model files come in | The first start and the first `strata_ask` | Yes |

With `STRATA_LOCAL_ONLY=true` the server refuses to start while any blocked path is configured, and the error names each setting that caused it. The embedding calls and query rewriting check the switch again on every call, so a key added later is refused too. The Hugging Face libraries run offline and load the embedding and reranking models from local files only; a model that is not installed fails with a message that says how to install it (`uv run hf download <model>` on a machine that may download, or `EMBEDDING_MODEL` / `RERANKING_MODEL` set to a folder that holds it). The switch does not cover the database: `DATABASE_URL` may point at a team server on purpose.

`strata_admin_health` always lists the switch (`local_only`) and the active outbound providers (`outbound`: provider, purpose, what it sends, destination host and the setting that turns it on), whether the switch is on or not. It never shows keys, credentials, paths or ports.

StrataKB cannot control what your MCP client's language model does with the retrieved text. If the client uses a cloud model, the retrieved text goes to that provider. Teams that need the whole path to stay on the machine should embed through an OpenAI-compatible server on this machine (see [Embeddings from a model server](#embeddings-from-a-model-server)) and use a client with a local model.

## Project structure

```
db/                   Schema: 01_init.sql (core, vectors), 02_learning.sql (learning), 03_readonly_role.sql (SQL tools role)
examples/             Example layer presets (SEPA, ISO 20022)
scripts/              apply_preset.py, backup_restore.py
src/stratakb/
  mcp_server.py       Entry point: server, admin and user tools
  core/               Database pool, roles, audit log, logging
  knowledge/          Layers, presets, ingestion pipeline, parsers, embeddings, vector store
  rag/                Retriever (hybrid + RRF), reranker, cache, routing, query rewriter, tools
  learning/           Interaction tracking, signals, synonyms, promotion, feedback tools
  database_access/    Read-only SQL and introspection tools
tests/                pytest suite
docs/IMPROVEMENTS.md  Product goal and order of work (issues hold the items)
```

## Development

```bash
uv sync --extra dev
uv run pytest                  # full suite; integration tests need the database running
uv run ruff check src tests    # lint
uv run python -m stratakb      # run the server on stdio
```

Without a running database, the integration tests fail with a connection error and the rest pass.

`scripts/backup_restore.py` backs up the whole database or exports and imports a single layer.

## Roadmap

The target: ask your own documents (contracts, rulebooks, specs, runbooks) and get answers you can **prove**, without learning how retrieval works and without sending private data anywhere you did not choose. Work is grouped into [milestones](https://github.com/boudou/StrataKB/milestones):

1. **Five-minute start**: add a folder in one step, a one-command local mode, any OpenAI-compatible model provider, and a clear statement of where data goes.
2. **Answers you can prove** (planned): every claim can be checked against its source, so you can be sure the model did not make it up.
   - Each passage comes with its exact location: file, page, line range and clause (`7 Termination > 7.2`), and documents are chunked along their clauses and sections.
   - **Grounded mode:** your client writes the answer, and `strata_verify` checks that every quote, number, date, amount, code and clause reference in it appears in the cited passages.
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
