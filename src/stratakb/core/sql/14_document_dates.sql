-- StrataKB document dates
-- doc_date: the document's date as YYYY-MM-DD; NULL when none is known.
-- doc_date_source: where it came from, the document's front matter or the
-- file's modification time. Both are read at ingest.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

ALTER TABLE strata_vectors.documents
    ADD COLUMN IF NOT EXISTS doc_date TEXT;

ALTER TABLE strata_vectors.documents
    ADD COLUMN IF NOT EXISTS doc_date_source TEXT
    CONSTRAINT documents_doc_date_source_check
    CHECK (doc_date_source IN ('front_matter', 'file'));
