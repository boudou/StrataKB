-- StrataKB document structure map and chunk clause paths
-- Each stored document version gets its headings, numbered clauses and
-- sections, with the span of the document text each one covers, so
-- text[char_start:char_end] is the section with its subsections. Every chunk
-- records the path of the innermost section it lies in.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

CREATE TABLE IF NOT EXISTS strata_vectors.document_sections (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id     BIGINT NOT NULL REFERENCES strata_vectors.documents(id) ON DELETE CASCADE,
    -- The normalised number or id ('7.2', 'article 12', 'annex ii'), or the
    -- heading text in lower case when the heading has no number
    label           TEXT NOT NULL,
    -- The heading text after the number; NULL when the line has none
    title           TEXT,
    -- The names of the enclosing sections and this one: '7 Termination > 7.2 Notice'
    path            TEXT NOT NULL,
    -- 1 for a top-level section
    depth           INT NOT NULL,
    char_start      INT NOT NULL,
    char_end        INT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_sections_label
    ON strata_vectors.document_sections(label);
CREATE INDEX IF NOT EXISTS idx_document_sections_document_id
    ON strata_vectors.document_sections(document_id);

-- Chunks stored before this file, and chunks outside any section, keep NULL
ALTER TABLE strata_vectors.chunks ADD COLUMN IF NOT EXISTS clause_path TEXT;
