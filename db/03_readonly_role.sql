-- StrataKB read-only role for the SQL tools (strata_db_query, strata_db_explain).
-- Point READONLY_DATABASE_URL at it. The tools already run each statement in a
-- read-only transaction; this role also keeps them away from superuser-only
-- functions such as pg_read_file().
--
-- The role may SELECT from four tables only: strata_core.knowledge_layers,
-- strata_core.ingestion_runs, strata_core.ingested_files and
-- strata_vectors.chunks. Not the audit log, users or anything in
-- strata_learning. A table added later is not readable until granted by name.
--
-- The password comes from the READONLY_DB_PASSWORD environment variable (psql 15+):
--   READONLY_DB_PASSWORD=... psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f db/03_readonly_role.sql
-- Run it after 01_init.sql and 02_learning.sql, as the owner of the schemas.

\getenv readonly_password READONLY_DB_PASSWORD

CREATE ROLE strata_readonly LOGIN PASSWORD :'readonly_password';

ALTER ROLE strata_readonly SET default_transaction_read_only = on;
ALTER ROLE strata_readonly SET statement_timeout = '30s';

GRANT USAGE ON SCHEMA strata_core, strata_vectors TO strata_readonly;
GRANT SELECT ON strata_core.knowledge_layers, strata_core.ingestion_runs, strata_vectors.chunks
    TO strata_readonly;

-- 04_ingested_files.sql runs after this file on a fresh Docker volume only;
-- the server's migration 06 (06_readonly_grants.sql) grants it then. In CI the
-- server's migrations run first, so the block below grants it
DO $$
BEGIN
    IF to_regclass('strata_core.ingested_files') IS NOT NULL THEN
        GRANT SELECT ON strata_core.ingested_files TO strata_readonly;
    END IF;
END $$;
