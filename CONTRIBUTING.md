# Contributing to StrataKB

Thanks for your interest! Bug reports, ideas and pull requests are welcome. Good first stops are the open [enhancement issues](https://github.com/boudou/StrataKB/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement), the [milestones](https://github.com/boudou/StrataKB/milestones), and the product goal in [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

## Setup

```bash
uv sync --extra dev
cp .env.example .env
docker compose up -d       # PostgreSQL + pgvector for the integration tests
```

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

## Before opening a pull request

- Run `uv run pytest`. With the database running, every test must pass.
- Run `uv run ruff check` on the files you changed and leave them cleaner than you found them.
- Keep changes focused; one topic per pull request.
- Match the surrounding code: naming, comment density, idiom.
- Keep the code domain-neutral. Anything specific to one documentation domain belongs in a layer preset under `examples/`, not in `src/`.
- Name new MCP tools `strata_<verb>_<noun>`, guard them with `@require_role`, and write an audit entry for anything that changes data.
- Log with structlog only. Logs go to stderr; stdout carries the MCP protocol.
- Never log secrets or personal data; the masking in `core/logging.py` is a safety net, not a licence.

## Reporting bugs

Open an issue with the steps to reproduce, what you expected, what happened, your OS, and your MCP client. For ingestion problems, a minimal sample file helps most.

## License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).
