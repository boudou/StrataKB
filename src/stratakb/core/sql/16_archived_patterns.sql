-- StrataKB per-layer archived files patterns
-- gitignore-style patterns, matched against each file's relative path, saying
-- which of a layer's files are archived. This is not the layer's 'archived'
-- status, which covers the whole layer.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

ALTER TABLE strata_core.knowledge_layers
    ADD COLUMN IF NOT EXISTS archived_patterns JSONB NOT NULL DEFAULT '[]';
