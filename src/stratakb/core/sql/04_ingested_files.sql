-- StrataKB per-file ingestion tracking
-- One row per ingested file, so a re-run can skip unchanged files, replace
-- changed ones and remove files that are gone. Idempotent: applied by the
-- server on startup (stratakb.core.migrations) and by docker-compose on a
-- fresh volume.

CREATE TABLE IF NOT EXISTS strata_core.ingested_files (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer_id        UUID NOT NULL REFERENCES strata_core.knowledge_layers(id) ON DELETE CASCADE,
    -- Resolved absolute directory that was ingested
    source_root     TEXT NOT NULL,
    -- Path below source_root, with POSIX separators
    relative_path   TEXT NOT NULL,
    -- SHA-256 of the file's bytes
    content_hash    CHAR(64) NOT NULL,
    size_bytes      BIGINT,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (layer_id, source_root, relative_path)
);

-- Chunks belong to the file they came from; two roots can hold the same
-- relative path, so chunks are replaced by file_id, never by source_file
ALTER TABLE strata_vectors.chunks
    ADD COLUMN IF NOT EXISTS file_id UUID REFERENCES strata_core.ingested_files(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON strata_vectors.chunks(file_id);
