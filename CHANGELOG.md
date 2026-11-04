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

### Changed

- Ingesting a folder again now syncs it instead of rebuilding the layer: files with unchanged content are skipped, changed files are replaced, new files are added and files that are gone from that folder are removed, without touching other folders in the layer. `strata_ingest_documents` reports `files_added`, `files_updated`, `files_unchanged` and `files_removed`; pass `replace_all=True` for a full rebuild, for example after changing the layer's chunking or the embedding model.
- A chunk's `source_file` is now the file's path relative to the ingested folder (for example `guides/readme.md`), not its bare name.
- Ingested files are tracked in a new `strata_core.ingested_files` table. The server adds it to an existing database at startup, or on the first ingestion; a layer ingested before this change is rebuilt once on its next ingestion. That first run keeps only the folder it ingests, so ingest every folder of such a layer again (a preset run does this).

### Fixed

- A layer's document and chunk counts are recounted from the database after every ingestion, instead of showing only the last run's numbers.
- Resuming an ingestion no longer skips a file because another file with the same name, in a different folder, was already ingested.
- The keyword leg of hybrid search found almost nothing for natural-language questions, because it required every word of the question. It now matches any word of the question (stemmed, with stopwords dropped by PostgreSQL), keeps the best-ranked matches when more match than it returns, and keeps the strict web-search syntax for queries with quoted phrases, `OR` or `-exclusions`.
- Implicit signals (rephrasing, layer switches) were lost whenever the previous query had results, because stored results came back as JSON text.
- Layer switches are compared with the layers routing picked for the previous query, not with the layers it searched.
- Synonym mining pairs a rephrased question with the question just before it, not with every earlier question in the session.
- Interactions record their real response time instead of 0, and the promoter reports how many rewrites it decayed.
- `strata_feedback` rejects an unknown `interaction_id` instead of reporting success.

### Security

- `strata_db_query` and `strata_db_explain` run each statement in a read-only transaction that is always rolled back, so `SELECT … INTO`, writing functions and `set_config()` no longer take effect. The statement timeout is now transaction-local instead of sticking to the pooled connection, and rows are fetched through a cursor up to the 1,000-row cap instead of in full.
