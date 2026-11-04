-- StrataKB learning schema
-- Self-improving retrieval: captures interactions, feedback, and learned patterns

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS strata_learning;

-- Sessions group related queries
CREATE TABLE IF NOT EXISTS strata_learning.query_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         VARCHAR(100),  -- external_id from get_current_user()
    session_start   TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_end     TIMESTAMPTZ,
    query_count     INT NOT NULL DEFAULT 0,
    avg_confidence  FLOAT,
    resolution      VARCHAR(20) DEFAULT 'unknown'
                    CHECK (resolution IN ('succeeded', 'partial', 'failed', 'unknown'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON strata_learning.query_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_resolution ON strata_learning.query_sessions(resolution);

-- Every query and its results
-- A plain table with a created_at index is enough for moderate volume; switch to
-- monthly PARTITION BY RANGE (created_at) once it grows past ~100k rows.
CREATE TABLE IF NOT EXISTS strata_learning.interaction_events (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id        UUID REFERENCES strata_learning.query_sessions(id) ON DELETE CASCADE,
    query_text        TEXT NOT NULL,
    query_embedding   vector(768),
    rewritten_queries JSONB,
    layers_detected   TEXT[],
    layers_searched   TEXT[],
    results           JSONB NOT NULL DEFAULT '[]',
    result_count      INT NOT NULL DEFAULT 0,
    confidence_level  VARCHAR(10),
    confidence_score  FLOAT,
    tool_used         VARCHAR(20),
    response_time_ms  INT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_interactions_session ON strata_learning.interaction_events(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON strata_learning.interaction_events(created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_confidence ON strata_learning.interaction_events(confidence_level);

-- Feedback signals (implicit + explicit)
CREATE TABLE IF NOT EXISTS strata_learning.feedback_signals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id  UUID REFERENCES strata_learning.interaction_events(id) ON DELETE CASCADE NOT NULL,
    signal_type     VARCHAR(10) NOT NULL CHECK (signal_type IN ('implicit', 'explicit')),
    signal_name     VARCHAR(50) NOT NULL,
    signal_value    FLOAT NOT NULL,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_interaction ON strata_learning.feedback_signals(interaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_signal ON strata_learning.feedback_signals(signal_name);

-- Learned query rewrites
CREATE TABLE IF NOT EXISTS strata_learning.learned_rewrites (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern          TEXT NOT NULL,
    pattern_type     VARCHAR(10) NOT NULL CHECK (pattern_type IN ('exact', 'keyword', 'regex')),
    optimized_rewrite TEXT NOT NULL,
    layers           TEXT[],
    success_count    INT NOT NULL DEFAULT 0,
    total_count      INT NOT NULL DEFAULT 0,
    success_rate     FLOAT NOT NULL DEFAULT 0.0,
    last_used        TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(pattern, pattern_type)
);

CREATE INDEX IF NOT EXISTS idx_rewrites_pattern_trgm ON strata_learning.learned_rewrites USING gin (pattern gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_rewrites_success ON strata_learning.learned_rewrites(success_rate DESC);

-- Term equivalences
CREATE TABLE IF NOT EXISTS strata_learning.query_synonyms (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    term        TEXT NOT NULL,
    synonym     TEXT NOT NULL,
    layer       TEXT,
    confidence  FLOAT NOT NULL DEFAULT 0.5,
    source      VARCHAR(20) NOT NULL CHECK (source IN ('user_feedback', 'pattern_mining', 'manual')),
    usage_count INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(term, synonym, layer)
);

CREATE INDEX IF NOT EXISTS idx_synonyms_term ON strata_learning.query_synonyms(term);

-- Layer routing affinities
CREATE TABLE IF NOT EXISTS strata_learning.layer_affinity_scores (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_pattern  TEXT NOT NULL,
    layer_name     TEXT NOT NULL,
    affinity_score FLOAT NOT NULL DEFAULT 0.5,
    sample_count   INT NOT NULL DEFAULT 0,
    last_updated   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(query_pattern, layer_name)
);

CREATE INDEX IF NOT EXISTS idx_affinity_pattern ON strata_learning.layer_affinity_scores(query_pattern);
