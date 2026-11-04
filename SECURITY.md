# Security policy

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/boudou/StrataKB/security/advisories/new)
rather than a public issue. Include steps to reproduce and the affected version.

Reports about these areas are especially welcome:

- **Read-only SQL** (`strata_db_query`, `strata_db_explain`): any way to change data or
  escape the read-only checks.
- **Retrieved content**: ingested documents are untrusted input, and text in them reaches
  the MCP client. Prompt injection through ingested files is in scope.
- **Roles**: any way for a role to run a tool above its level.
- **Personal data**: values that slip past log masking or query scrubbing.

## Deployment notes

- On the stdio transport, whoever starts the server sets `STRATA_ROLE`. Roles separate
  duties between team members; they are not a security boundary on a machine the user
  controls. Restrict what the database user can do with PostgreSQL grants.
- Give the server a database user with only the privileges it needs, and keep
  `DATABASE_URL` and API keys out of version control.
- The SQL tools run each statement in a read-only transaction that is rolled back.
  A read-only transaction still allows reading functions such as `pg_read_file()` when
  the connection is a superuser, so the SQL tools are only registered when
  `READONLY_DATABASE_URL` is set; point it at the `strata_readonly` role from
  `db/03_readonly_role.sql`, which can only `SELECT` from the StrataKB schemas. At startup
  the server removes the SQL tools when that role is a superuser, a member of a
  superuser role, or a member of `pg_read_server_files`, `pg_write_server_files`, `pg_execute_server_program` or
  `pg_signal_backend`, or when the check cannot run.
