-- StrataKB supersedes declaration
-- supersedes: the file a document version says it replaces, from its front
-- matter, as a path below the same source root; NULL when it declares nothing
-- usable. Read at ingest. The target may not exist.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

ALTER TABLE strata_vectors.documents
    ADD COLUMN IF NOT EXISTS supersedes TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_supersedes
    ON strata_vectors.documents (layer_id, source_root, supersedes)
    WHERE supersedes IS NOT NULL;
