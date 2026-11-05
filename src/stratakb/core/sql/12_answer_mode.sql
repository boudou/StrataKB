-- StrataKB per-layer answer mode
-- grounded: the client writes the answer from the cited passages (default).
-- extractive: the server answers with the document's own sentences.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

ALTER TABLE strata_core.knowledge_layers
    ADD COLUMN IF NOT EXISTS answer_mode TEXT NOT NULL DEFAULT 'grounded'
    CONSTRAINT knowledge_layers_answer_mode_check
    CHECK (answer_mode IN ('grounded', 'extractive'));
