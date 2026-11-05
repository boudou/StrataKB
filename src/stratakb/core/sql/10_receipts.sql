-- StrataKB receipts: a stored record of exactly what the server returned or
-- verified, which a person can open outside the chat. Portable types only
-- (text, integers, JSON as text), so a local SQLite mode can store the same
-- shape. Neither table is granted to strata_readonly.
-- Idempotent: applied by the server on startup (stratakb.core.migrations)
-- and by psql as the database owner.

CREATE TABLE IF NOT EXISTS strata_core.receipts (
    -- 'rcpt_' and a random URL-safe token
    id          TEXT PRIMARY KEY,
    issued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    issued_by   TEXT NOT NULL,
    kind        TEXT NOT NULL
        CONSTRAINT receipts_kind_check CHECK (kind IN ('extractive', 'verify')),
    mode        TEXT NOT NULL,
    -- Always PII-scrubbed
    question    TEXT,
    -- JSON list of the cited layer names
    layers      TEXT NOT NULL,
    -- JSON: the quotes (with their text, so a receipt outlives its sources),
    -- the draft and the verification result when present
    body        TEXT NOT NULL
);

-- The document versions a receipt cites; pruning keeps them. A document
-- deleted with its layer takes its rows with it, and the receipt still
-- shows its stored quote.
CREATE TABLE IF NOT EXISTS strata_core.receipt_documents (
    receipt_id  TEXT NOT NULL REFERENCES strata_core.receipts(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES strata_vectors.documents(id) ON DELETE CASCADE,
    PRIMARY KEY (receipt_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_receipt_documents_document_id
    ON strata_core.receipt_documents(document_id);
