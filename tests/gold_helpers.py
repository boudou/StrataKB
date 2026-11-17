"""The gold set scorer run on a stub server, shared by its test modules."""

from __future__ import annotations

import asyncio
import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

from tests.helpers import REPO_DIR

EVALS = REPO_DIR / "evals"
SCORER = EVALS / "score_gold.py"

# A stub strata_ask call that never returns
HANG = object()


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _last_run(runs=None):
    """The layer's latest ingestion run stubbed: `runs` (default: none recorded)."""
    return patch("stratakb.knowledge.layers.last_ingestion_run", AsyncMock(return_value=runs))


def _score(
    folder, layer, out, *extra, failing=None, layer_status=None, asked=None, runs=None,
    seen=None, reply=None,
):
    """Run the scorer on a stub server; returns the questions it asked and the layers.

    `failing` maps a question to a reply to return, an exception to raise, or HANG
    for a call that never returns.
    `layer_status` replaces the check of the layer, which otherwise passes.
    `asked` is a list to collect the questions in, for a run that stops.
    `runs` is the layer's latest ingestion run (default: none recorded).
    `seen` is a list to collect each call's arguments in.
    `reply` makes the reply to the arguments of every other call.
    """

    async def call_tool(name, arguments):
        assert name == "strata_ask"
        calls.append((arguments["question"], arguments["layers"]))
        if seen is not None:
            seen.append(arguments)
        if reply is not None:
            return reply(arguments)
        failure = (failing or {}).get(arguments["question"])
        if failure is HANG:
            await asyncio.Event().wait()
        if isinstance(failure, Exception):
            raise failure
        if failure:
            return failure
        found = arguments["question"] != "refuse?"
        return {
            "status": "ok",
            "mode": "grounded",
            "answer_status": "found" if found else "not_found",
            "passages": [{"text": "a  needle", "citation": {"file": "a.rst"}}] if found else [],
        }

    calls = [] if asked is None else asked
    server = MagicMock(call_tool=AsyncMock(side_effect=call_tool))
    with (
        patch("stratakb.mcp_server.build_server", return_value=server),
        patch("stratakb.core.logging.configure_logging"),
        patch(
            "stratakb.knowledge.layers.get_layer_status",
            layer_status or AsyncMock(return_value={"id": "layer-id", "chunk_count": 3}),
        ),
        _last_run(runs),
    ):
        _load(SCORER).main(
            ["--gold", str(folder / "gold.jsonl"), "--layer", layer, "--out", str(out), *extra]
        )
    return calls
