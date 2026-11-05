"""StrataKB configuration management."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings

DOTENV_WARNING = "./.env is not read; set STRATAKB_CONFIG=.env to use it."

# The local database: the default DATABASE_URL and the stratakb db up container
# (core/local_db.py) both use these
LOCAL_DB_USER = "strata"
LOCAL_DB_PASSWORD = "strata_dev_password"
LOCAL_DB_NAME = "strata"
LOCAL_DB_PORT = 5433


def local_database_url(port: int = LOCAL_DB_PORT) -> str:
    return f"postgresql://{LOCAL_DB_USER}:{LOCAL_DB_PASSWORD}@127.0.0.1:{port}/{LOCAL_DB_NAME}"


class Settings(BaseSettings):
    """Application settings.

    Read, each source overriding the one before: the defaults below, the
    user-level config.env, the file named by STRATAKB_CONFIG, then
    environment variables. ./.env is not read unless STRATAKB_CONFIG names it.
    """

    # Database (127.0.0.1, not localhost: on Windows localhost resolves to ::1
    # first, and Docker in WSL forwards IPv4 only)
    database_url: str = local_database_url()
    # Connection for strata_db_query and strata_db_explain; empty = the SQL tools
    # are not registered. A superuser, a member of a superuser or server-access
    # role, or a role that can't be checked is refused at startup.
    # Point it at the strata_readonly role from db/03_readonly_role.sql.
    readonly_database_url: str = ""

    # LLM APIs
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Query Rewriting
    # "anthropic" (needs ANTHROPIC_API_KEY), or "openai_compatible" (needs
    # QUERY_REWRITE_BASE_URL, e.g. http://127.0.0.1:11434/v1, and a QUERY_REWRITE_MODEL)
    query_rewrite_provider: str = "anthropic"
    query_rewrite_model: str = "claude-haiku-5-5"
    query_rewrite_base_url: str = ""
    # key for QUERY_REWRITE_BASE_URL; OPENAI_API_KEY is never sent there
    query_rewrite_api_key: str = ""

    # Embeddings
    # "local", or "openai" (OpenAI or an OpenAI-compatible server)
    embedding_backend: str = "local"
    # local: sentence-transformers model; openai: text-embedding-3-large unless set
    embedding_model: str = "all-mpnet-base-v2"
    embedding_dimensions: int = 768  # must match model output
    # openai backend: an OpenAI-compatible server, e.g. http://127.0.0.1:11434/v1;
    # empty = OpenAI
    embedding_base_url: str = ""
    embedding_api_key: str = ""  # key for EMBEDDING_BASE_URL; OPENAI_API_KEY is never sent there

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
    hybrid_keyword_slots: int = 1       # results kept for the best full-text hits

    # Reranking
    reranking_enabled: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranking_candidate_multiplier: int = 3  # over-retrieve factor (top_k * this)

    # Answers: strata_ask returns the best passages that fit this many tokens
    # (about four characters each); the first passage always comes back
    answer_token_budget: int = 1500

    # Audit: if True, search and ask entries keep questions without PII scrubbing
    audit_store_raw_queries: bool = False

    # Learning System
    learning_enabled: bool = True
    learning_store_raw_queries: bool = False  # if True, stores queries without PII scrubbing
    learning_session_timeout_minutes: int = 30
    learning_promote_every_n_queries: int = 50
    learning_min_success_rate: float = 0.6
    learning_rewrite_bypass_threshold: float = 0.85
    learning_rewrite_bypass_min_samples: int = 5

    # Data egress
    # True: refuse to start with any provider that sends content off this
    # machine, and load Hugging Face models from local files only
    strata_local_only: bool = False

    # MCP Server
    strata_user: str = "developer"
    strata_role: str = "viewer"

    # Logging
    log_level: str = "INFO"

    # utf-8-sig: a file saved with a BOM keeps its first key. Unknown keys,
    # such as the compose-only ones in .env.example, are ignored.
    model_config = {"env_file_encoding": "utf-8-sig", "extra": "ignore"}


def user_config_file() -> Path:
    """The user-level config.env, whether or not it exists."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "stratakb" / "config.env"


def _named_config_file() -> Path | None:
    """The path in STRATAKB_CONFIG, or None when it is unset or empty."""
    value = os.environ.get("STRATAKB_CONFIG")
    return Path(value).expanduser() if value else None


def config_files() -> list[Path]:
    """The config files that exist, lowest priority first.

    The user-level config.env, then the file named by STRATAKB_CONFIG.
    A file listed twice is read once.
    """
    candidates = [user_config_file()]
    named = _named_config_file()
    if named is not None:
        candidates.append(named)

    files: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            files.append(path)
    return files


def config_file_missing() -> bool:
    """True when STRATAKB_CONFIG names a path that is not a file."""
    named = _named_config_file()
    return named is not None and not named.is_file()


def dotenv_warning() -> str | None:
    """A hint when ./.env exists but is not among the files read, else None."""
    dotenv = Path.cwd() / ".env"
    if not dotenv.is_file():
        return None
    if dotenv.resolve() in {path.resolve() for path in config_files()}:
        return None
    return DOTENV_WARNING


def get_settings() -> Settings:
    """Settings from the defaults, the config files and the environment."""
    return Settings(_env_file=config_files())
