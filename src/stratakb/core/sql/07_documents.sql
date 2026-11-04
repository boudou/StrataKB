-- StrataKB document text, versions and chunk offsets
-- Each ingested file's extracted text is stored once per version, and every
-- chunk records the span of that text it holds, so
-- text[char_start:char_end] = content. Portable types only (text, integers,
-- JSON as text), so a local SQLite mode can store the same shape.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

CREATE TABLE IF NOT EXISTS strata_vectors.documents (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    layer_id        UUID NOT NULL REFERENCES strata_core.knowledge_layers(id) ON DELETE CASCADE,
    file_id         UUID REFERENCES strata_core.ingested_files(id) ON DELETE SET NULL,
    -- The ingested root and the path below it, with POSIX separators: the
    -- file's identity, as in ingested_files. Kept when the file row goes,
    -- so a cleared layer continues each file's numbering
    source_root     TEXT NOT NULL,
    relative_path   TEXT NOT NULL,
    -- Starts at 1; increments when the extracted text changes
    version         INT NOT NULL,
    -- The parser's output, pages joined by a blank line
    text            TEXT NOT NULL,
    -- SHA-256 of text (not of the file's bytes)
    sha256          CHAR(64) NOT NULL,
    -- JSON [[char_start, page_number], ...] for PDFs; NULL otherwise
    page_map        TEXT,
    text_origin     TEXT NOT NULL DEFAULT 'text'
        CONSTRAINT documents_text_origin_check CHECK (text_origin IN ('text', 'ocr')),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (layer_id, source_root, relative_path, version)
);

-- No cascade: a document is pruned only when no chunk references it.
-- Chunks stored before this file keep NULL offsets.
ALTER TABLE strata_vectors.chunks
    ADD COLUMN IF NOT EXISTS document_id BIGINT REFERENCES strata_vectors.documents(id);
ALTER TABLE strata_vectors.chunks ADD COLUMN IF NOT EXISTS char_start INT;
ALTER TABLE strata_vectors.chunks ADD COLUMN IF NOT EXISTS char_end INT;

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON strata_vectors.chunks(document_id);
