# Changelog

Notable changes to StrataKB. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Knowledge layers with per-layer chunking, metadata schema and routing patterns.
- Layer presets: a YAML file describes a layer and its source folders; `strata_apply_preset` and `scripts/apply_preset.py` build or rebuild the layer, with a dry-run mode that needs no database.
- Example presets for SEPA payments and ISO 20022 messaging, with a sample reason code list and download instructions.
- Ingestion of PDF, DOCX, Markdown, HTML, CSV, JSON, XML/XSD and OpenAPI/AsyncAPI YAML.
- Hybrid retrieval (pgvector HNSW plus full-text search, fused with RRF), cross-encoder reranking and a per-layer-invalidated query cache.
- `strata_ask`: optional query rewriting, built from the layers' own descriptions, with parallel multi-query search and source citations.
- Learning system: interaction tracking, implicit signals (rephrasing, layer switches, follow-ups), synonym mining and explicit feedback.
- Viewer, operator and admin roles on every tool, an append-only audit log, and read-only SQL tools.
- Masking of IBANs and card numbers in logs, and scrubbing of IBANs, BICs and account numbers from stored queries.
- Local embeddings (`all-mpnet-base-v2`) by default, OpenAI embeddings optional, and in-place re-embedding of a layer.
- Docker Compose setup that creates all schemas, including the learning schema, on first start.
- `strata_readonly` database role (`db/03_readonly_role.sql`) and the `READONLY_DATABASE_URL` setting for the SQL tools.
- `strata_search` and `strata_ask` return an `interaction_id` for `strata_feedback`; `strata_ask` also reports `auto_detected_layers`.
- The promoter learns query rewrites from asks marked helpful and from rephrased questions that then got a high-confidence result, and layer affinities from `correct_layer` feedback and layer switches.
- `stratakb add <folder>` turns a folder into a layer in one step: it picks the files (git-aware, honouring `.gitignore` and a root `.strataignore`), skips secrets, build files, code and unsupported formats, asks before a broad path or a large run (`--allow-broad`, `--yes`), shows progress and prints a summary, or a JSON report with `--json`. Running it again syncs the folder. `stratakb` alone or `stratakb serve` still starts the server.
- Layers record their origin (`manual`, `preset:<name>` or `folder`), owner and root folder; one folder belongs to at most one layer.
- Plain-text documents: `.txt`, `.rst` and `.adoc`.
- A Python PEPs example preset (`examples/peps/`), fetched with a sparse git clone and never committed.
- `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY`: embed through any OpenAI-compatible server, such as Ollama, LM Studio or vLLM. Vectors whose length differs from `EMBEDDING_DIMENSIONS` are refused with a message naming the setting.
- `QUERY_REWRITE_PROVIDER=openai_compatible` with `QUERY_REWRITE_BASE_URL`, `QUERY_REWRITE_MODEL` and `QUERY_REWRITE_API_KEY`: rewrite questions through a local or self-hosted chat model. The reply parser accepts the first JSON array of strings anywhere in the reply.
- `STRATA_LOCAL_ONLY`: the server refuses to start while any setting would send content to a host that is not loopback, keeps Hugging Face offline, and `strata_admin_health` lists the active outbound providers by host and setting name. The README gains a "Where your data goes" section.
- At startup the server checks, in the background, that the embedding server lists the configured model and that the vector column matches `EMBEDDING_DIMENSIONS`; `strata_admin_health` reports both under `embedding`.
- Ingestion flags PDFs that look scanned (`files_flagged`, reason `probably_scanned`).
- Numbered schema migrations, recorded in `strata_core.schema_migrations`. The server and `stratakb add` build an empty database from the schema files shipped in the package, and bring an existing one up to date.

### Changed

- Ingesting a folder again now syncs it instead of rebuilding the layer: files with unchanged content are skipped, changed files are replaced, new files are added and files that are gone from that folder are removed, without touching other folders in the layer. `strata_ingest_documents` reports `files_added`, `files_updated`, `files_unchanged` and `files_removed`; pass `replace_all=True` for a full rebuild, for example after changing the layer's chunking or the embedding model.
- A chunk's `source_file` is now the file's path relative to the ingested folder (for example `guides/readme.md`), not its bare name.
- Ingested files are tracked in a new `strata_core.ingested_files` table. The server adds it to an existing database at startup, or on the first ingestion; a layer ingested before this change is rebuilt once on its next ingestion. That first run keeps only the folder it ingests, so ingest every folder of such a layer again (a preset run does this).
- **Breaking:** settings are no longer read from `./.env` in the folder the server starts in. They come from the defaults, then the user-level `config.env` (`%APPDATA%\stratakb\config.env` on Windows, `~/.config/stratakb/config.env` elsewhere), then the file named by `STRATAKB_CONFIG`, then environment variables. The server logs `dotenv_not_read` when it finds a `./.env` it no longer reads. Move your settings to `config.env` or point `STRATAKB_CONFIG` at your file.
- The server answers `initialize` and `tools/list` at once and loads the embedding model in the background (a cold start took 13–20 s before).
- Ingestion no longer fails a run on an empty file or one parser error: secret-named files (such as `client_secret.json` or `secrets.yaml`) are skipped, or removed when they were ingested before; empty files and files with no text are skipped; a file whose parser fails keeps its old chunks while the rest of the run goes on. Every file gets an outcome, and `strata_ingest_documents` returns `files_skipped`. Documents named after secrets but in a document format (for example `credentials.txt` or `secrets.md`) are still ingested, so check what you add.
- The Markdown, HTML and CSV parsers treat empty input as no text instead of a parse error.
- The base schema files `01_init.sql` and `02_learning.sql` moved from `db/` to `src/stratakb/core/sql/`; `db/` keeps only `03_readonly_role.sql`. Pointed at an empty database, the server now creates the StrataKB schemas and the `vector` and `pg_trgm` extensions there instead of failing.
- The README opens with a four-step Quick start in plain words, followed by the reference and a Glossary.
- `stratakb add` and the server share one config source and one entry point; `python -m stratakb` runs the same command.

### Fixed

- A layer's document and chunk counts are recounted from the database after every ingestion, instead of showing only the last run's numbers.
- Resuming an ingestion no longer skips a file because another file with the same name, in a different folder, was already ingested.
- The keyword leg of hybrid search found almost nothing for natural-language questions, because it required every word of the question. It now matches any word of the question (stemmed, with stopwords dropped by PostgreSQL), keeps the best-ranked matches when more match than it returns, and keeps the strict web-search syntax for queries with quoted phrases, `OR` or `-exclusions`.
- Implicit signals (rephrasing, layer switches) were lost whenever the previous query had results, because stored results came back as JSON text.
- Layer switches are compared with the layers routing picked for the previous query, not with the layers it searched.
- Synonym mining pairs a rephrased question with the question just before it, not with every earlier question in the session.
- Interactions record their real response time instead of 0, and the promoter reports how many rewrites it decayed.
- `strata_feedback` rejects an unknown `interaction_id` instead of reporting success.
- The query and routing caches are cleared when any process changes a layer (checked at most every 2 s); before, searches could serve results up to 10 minutes old.
- The CI and release workflows pin `astral-sh/setup-uv` to a tag that exists.

### Security

- `strata_db_query` and `strata_db_explain` run each statement in a read-only transaction that is always rolled back, so `SELECT … INTO`, writing functions and `set_config()` no longer take effect. The statement timeout is now transaction-local instead of sticking to the pooled connection, and rows are fetched through a cursor up to the 1,000-row cap instead of in full.
- The SQL tools are registered only when `READONLY_DATABASE_URL` is set, and are removed at startup when that role is a superuser, a member of one, or holds a server-file or signal role. They no longer fall back to the owner role.
- The `strata_readonly` role may read only `knowledge_layers`, `ingestion_runs`, `ingested_files` and `chunks`, not the audit log, users or the learning schema. Migration `06_readonly_grants.sql` narrows existing installs at startup.
- Locked dependencies upgraded to patched releases for all 124 open advisories (among them certifi, PyJWT, AnyIO, pypdf, transformers, urllib3, cryptography, starlette and python-multipart). Some transitive packages move a major version: filelock 4, cryptography 50, soupsieve 4 and huggingface-hub 2.
- File listing through git passes `core.fsmonitor=false`, so a crafted `.git/config` in an added folder cannot run a command.
