-- StrataKB read-only role grant on the document table
-- strata_readonly may SELECT from strata_vectors.documents (each document's
-- extracted text, the same content its chunks already expose). 07 creates the
-- table after 06 has run, so this file grants it. Idempotent, run after 07:
-- applied by the server on startup (stratakb.core.migrations) and by psql as
-- the database owner. Does nothing when the role does not exist.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'strata_readonly') THEN
        GRANT SELECT ON strata_vectors.documents TO strata_readonly;
    END IF;
END $$;
