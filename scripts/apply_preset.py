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
from pathlib import Path

from stratakb.config import get_settings
from stratakb.core.database import close_pool
from stratakb.core.logging import configure_logging
from stratakb.knowledge.layers import LayerError
from stratakb.knowledge.presets import apply_preset, dry_run_parse, load_preset


async def _apply(preset_path: Path, docs_root: Path | None, dry_run: bool) -> dict:
    settings = get_settings()
    try:
        preset = load_preset(preset_path)
    except FileNotFoundError:
        return {"layer": "refused", "error": f"Preset not found: {preset_path}"}
    except (LayerError, ValueError) as e:
        return {"layer": "refused", "error": str(e)}

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
        result = await apply_preset(
            settings.database_url, settings.openai_api_key, preset, docs_root,
            settings=settings,
            tool_name="script:apply_preset",
            parameters={
                "preset_path": str(preset_path),
                "docs_root": str(docs_root) if docs_root else None,
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
    if result.get("layer") == "refused":
        sys.exit(1)


if __name__ == "__main__":
    main()
