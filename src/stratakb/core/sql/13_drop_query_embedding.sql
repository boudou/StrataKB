-- StrataKB: remove the unused query embedding from the learning tables
-- The column was never written and kept a fixed vector(768) when a re-embed
-- resized strata_vectors.chunks. Whatever needs it later adds it back, sized
-- with the chunks column.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

ALTER TABLE strata_learning.interaction_events
    DROP COLUMN IF EXISTS query_embedding;
