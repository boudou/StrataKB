"""Create a knowledge layer from a preset file and ingest its sources.

Same as the strata_apply_preset MCP tool, for ingestions too long to run
inside a tool call.

Usage:
    uv run python scripts/apply_preset.py examples/sepa/preset.yaml
    uv run python scripts/apply_preset.py examples/iso20022/preset.yaml --docs-root D:/iso20022
    uv run python scripts/apply_preset.py examples/sepa/preset.yaml --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from stratakb.config import get_settings
from stratakb.core.database import close_pool
from stratakb.core.logging import configure_logging
from stratakb.knowledge.presets import (
    create_preset_layer,
    dry_run_parse,
    ingest_preset,
    load_preset,
)


async def _apply(preset_path: Path, docs_root: Path | None, dry_run: bool) -> dict:
    settings = get_settings()
    preset = load_preset(preset_path)

    if dry_run:
        result = dry_run_parse(preset, docs_root)
        return {
            "layer": preset.name,
            "total_files": result.total_files,
            "parsed_files": result.parsed_files,
            "failed_files": result.failed_files,
            "total_chunks": result.total_chunks,
            "errors": result.errors,
        }

    try:
        layer = await create_preset_layer(settings.database_url, preset)
        ingestion = await ingest_preset(
            settings.database_url, settings.openai_api_key, preset, docs_root
        )
    finally:
        await close_pool()
    return {"layer": layer["status"], "ingestion": ingestion}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("preset", type=Path, help="Path to the preset YAML file")
    parser.add_argument(
        "--docs-root", type=Path, default=None,
        help="Directory the preset's source paths are relative to (default: the preset's folder)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only parse the files and report counts; no database or embeddings needed",
    )
    args = parser.parse_args()

    configure_logging(get_settings().log_level)
    result = asyncio.run(_apply(args.preset, args.docs_root, args.dry_run))
    json.dump(result, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
