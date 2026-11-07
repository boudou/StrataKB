-- StrataKB per-layer structure patterns
-- Regular expressions, usually set by a layer preset, that mark a section
-- start at the beginning of a line (a named group 'label' gives the label).
-- Ingestion uses them next to the generic heading and clause detectors.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

ALTER TABLE strata_core.knowledge_layers
    ADD COLUMN IF NOT EXISTS structure_patterns JSONB NOT NULL DEFAULT '[]';
