"""StrataKB configuration management."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://strata:strata_dev_password@localhost:5433/strata"
    # Connection for strata_db_query and strata_db_explain; empty = database_url.
    # Point it at the strata_readonly role from db/03_readonly_role.sql.
    readonly_database_url: str = ""

    # LLM APIs
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Query Rewriting (used only when ANTHROPIC_API_KEY is set)
    query_rewrite_model: str = "claude-haiku-5-5"

    # Embeddings
    embedding_backend: str = "local"  # "local" or "openai"
    embedding_model: str = "all-mpnet-base-v2"  # local: sentence-transformers model
    embedding_dimensions: int = 768  # must match model output

    # HNSW Search Tuning
    hnsw_ef_search: int = 40  # recall vs latency tradeoff (higher = more recall, slower)

    # Cache
    cache_enabled: bool = True
    cache_maxsize: int = 500
    cache_ttl_seconds: int = 600  # 10 minutes

    # Hybrid Search
    hybrid_search_enabled: bool = True
    hybrid_semantic_weight: float = 0.7  # RRF weight for vector results
    hybrid_keyword_weight: float = 0.3   # RRF weight for full-text results
    hybrid_rrf_k: int = 60              # RRF constant (standard = 60)

    # Reranking
    reranking_enabled: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranking_candidate_multiplier: int = 3  # over-retrieve factor (top_k * this)

    # Learning System
    learning_enabled: bool = True
    learning_store_raw_queries: bool = False  # if True, stores queries without PII scrubbing
    learning_session_timeout_minutes: int = 30
    learning_promote_every_n_queries: int = 50
    learning_min_success_rate: float = 0.6
    learning_rewrite_bypass_threshold: float = 0.85
    learning_rewrite_bypass_min_samples: int = 5

    # MCP Server
    strata_user: str = "developer"
    strata_role: str = "viewer"

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
