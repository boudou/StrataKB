-- StrataKB database initialization
-- Creates the core and vector schemas. The learning schema is in 02_learning.sql.

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- ==========================================
-- SCHEMA: strata_core (Application Data)
-- ==========================================
CREATE SCHEMA IF NOT EXISTS strata_core;

-- Knowledge Layer Configuration
CREATE TABLE IF NOT EXISTS strata_core.knowledge_layers (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             VARCHAR(100) UNIQUE NOT NULL,
    display_name     VARCHAR(200) NOT NULL,
    description      TEXT,
    metadata_schema  JSONB NOT NULL DEFAULT '{}',
    chunk_config     JSONB NOT NULL DEFAULT '{"chunk_size": 1000, "chunk_overlap": 200, "splitting_strategy": "recursive"}',
    -- Regular expressions that route a query to this layer when no layers are given
    routing_patterns JSONB NOT NULL DEFAULT '[]',
    status           VARCHAR(20) DEFAULT 'active',
    doc_count        INTEGER DEFAULT 0,
    chunk_count      INTEGER DEFAULT 0,
    last_ingested    TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Ingestion History
CREATE TABLE IF NOT EXISTS strata_core.ingestion_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer_id        UUID REFERENCES strata_core.knowledge_layers(id) ON DELETE CASCADE,
    source_path     TEXT NOT NULL,
    status          VARCHAR(20) NOT NULL,
    docs_processed  INTEGER DEFAULT 0,
    docs_failed     INTEGER DEFAULT 0,
    chunks_created  INTEGER DEFAULT 0,
    error_log       JSONB,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

-- Users & Roles
CREATE TABLE IF NOT EXISTS strata_core.users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id     VARCHAR(200) UNIQUE NOT NULL,
    display_name    VARCHAR(200),
    email           VARCHAR(200),
    role            VARCHAR(20) NOT NULL DEFAULT 'viewer',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_login      TIMESTAMPTZ
);

-- Audit Log (append-only)
CREATE TABLE IF NOT EXISTS strata_core.audit_log (
    id              BIGSERIAL PRIMARY KEY,
    user_id         VARCHAR(200) NOT NULL,
    action          VARCHAR(100) NOT NULL,
    tool_name       VARCHAR(200) NOT NULL,
    parameters      JSONB,
    result_summary  TEXT,
    ip_address      INET,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ==========================================
-- SCHEMA: strata_vectors (Vector Store)
-- ==========================================
CREATE SCHEMA IF NOT EXISTS strata_vectors;

-- Document chunks with embeddings
CREATE TABLE IF NOT EXISTS strata_vectors.chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer_name      VARCHAR(100) NOT NULL,
    content         TEXT NOT NULL,
    embedding       vector(768),
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata        JSONB NOT NULL DEFAULT '{}',
    source_file     VARCHAR(500),
    source_section  VARCHAR(500),
    doc_type        VARCHAR(100),
    version         VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_layer ON strata_vectors.chunks(layer_name);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON strata_vectors.chunks USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON strata_vectors.chunks(doc_type);

-- HNSW index for vector similarity search.
-- HNSW does not require pre-existing data (unlike IVFFlat).
-- For very large datasets (>1M chunks), tune m and ef_construction.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON strata_vectors.chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON strata_vectors.chunks USING gin(tsv);
