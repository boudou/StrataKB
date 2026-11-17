"""Create or update a knowledge layer from a preset file and ingest its sources.

Same as the strata_apply_preset MCP tool, for ingestions too long to run
inside a tool call. It writes the same audit entries as the tool, under the
tool name script:apply_preset, with STRATA_USER as the user.

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

from stratakb.config import get_settings
from stratakb.core.database import close_pool
from stratakb.core.logging import configure_logging
from stratakb.knowledge.presets import (
    PresetInputError,
    apply_preset,
    dry_run_parse,
    dry_run_report,
    open_preset,
)


async def _apply(preset_path: str, docs_root: str | None, dry_run: bool) -> dict:
    settings = get_settings()
    # The same checks and refusal texts as the strata_apply_preset tool
    try:
        preset, root = open_preset(preset_path, docs_root)
    except PresetInputError as e:
        return {"layer": "refused", "error": str(e)}

    if dry_run:
        return dry_run_report(preset, dry_run_parse(preset, root))

    try:
        result = await apply_preset(
            settings.database_url, settings.openai_api_key, preset, root,
            settings=settings,
            tool_name="script:apply_preset",
            parameters={
                "preset_path": preset_path,
                "docs_root": docs_root or None,
            },
        )
    finally:
        await close_pool()
    if result["status"] == "error":
        return {"layer": "refused", "error": result["error"]}
    return {
        "layer": result["layer"],
        "layer_changes": result["layer_changes"],
        "ingestion": result["ingestion"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("preset", type=str, help="Path to the preset YAML file")
    parser.add_argument(
        "--docs-root", type=str, default=None,
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
    # Not layer == "refused": a dry run of a preset named "refused" succeeded
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
