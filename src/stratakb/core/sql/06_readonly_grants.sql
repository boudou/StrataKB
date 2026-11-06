-- StrataKB read-only role grants, by table name
-- strata_readonly (the role behind READONLY_DATABASE_URL) may SELECT from the
-- layer, ingestion-run, ingested-file, chunk and document tables only: not the
-- audit log, users, schema_migrations or anything in strata_learning. On a new
-- database this file runs before 07 creates the document table, so 08 grants
-- that one; a later run grants it here too. Older installs got SELECT on every
-- table plus default privileges for future tables; this file removes them.
-- Idempotent, run after 04: applied by the server on startup
-- (stratakb.core.migrations) and by psql as the database owner. Does nothing
-- when the role does not exist.

DO $$
DECLARE
    acl RECORD;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'strata_readonly') THEN
        RETURN;
    END IF;

    -- Default privileges belong to the role that set them
    FOR acl IN
        SELECT DISTINCT pg_get_userbyid(d.defaclrole) AS owner_role, n.nspname
        FROM pg_default_acl d
        JOIN pg_namespace n ON n.oid = d.defaclnamespace
        CROSS JOIN LATERAL aclexplode(d.defaclacl) a
        WHERE d.defaclobjtype = 'r'
          AND n.nspname IN ('strata_core', 'strata_vectors', 'strata_learning')
          AND a.grantee = 'strata_readonly'::regrole
    LOOP
        EXECUTE format(
            'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA %I REVOKE ALL ON TABLES FROM strata_readonly',
            acl.owner_role, acl.nspname
        );
    END LOOP;

    REVOKE ALL ON ALL TABLES IN SCHEMA strata_core, strata_vectors, strata_learning
        FROM strata_readonly;
    REVOKE ALL ON SCHEMA strata_learning FROM strata_readonly;

    GRANT USAGE ON SCHEMA strata_core, strata_vectors TO strata_readonly;
    GRANT SELECT ON strata_core.knowledge_layers, strata_core.ingestion_runs,
        strata_core.ingested_files, strata_vectors.chunks TO strata_readonly;
    -- A re-run must not drop 08's grant
    IF to_regclass('strata_vectors.documents') IS NOT NULL THEN
        GRANT SELECT ON strata_vectors.documents TO strata_readonly;
    END IF;
END $$;
