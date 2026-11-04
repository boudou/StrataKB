-- StrataKB layer origin and folder root, plus the migration record
-- A layer remembers where it came from, who owns it and the folder it was
-- added from (at most one layer per folder). schema_migrations records which
-- numbered files a database has. Idempotent: applied by the server on startup
-- (stratakb.core.migrations) and by psql as the database owner.

CREATE TABLE IF NOT EXISTS strata_core.schema_migrations (
    version     INT PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE strata_core.knowledge_layers ADD COLUMN IF NOT EXISTS origin VARCHAR(100);
ALTER TABLE strata_core.knowledge_layers ADD COLUMN IF NOT EXISTS owner VARCHAR(200);
ALTER TABLE strata_core.knowledge_layers ADD COLUMN IF NOT EXISTS root_path TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS knowledge_layers_root_path_key
    ON strata_core.knowledge_layers (root_path)
    WHERE root_path IS NOT NULL;

-- 04 always runs before this file, so both are recorded here
INSERT INTO strata_core.schema_migrations (version, name)
VALUES (4, '04_ingested_files.sql'), (5, '05_add_folder.sql')
ON CONFLICT DO NOTHING;
