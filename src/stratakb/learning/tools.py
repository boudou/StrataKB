"""MCP tool definitions for the StrataKB Learning System.

Registers feedback, stats, and reset tools.
"""

from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP

from stratakb.core.audit import write_audit_log
from stratakb.core.auth import get_current_user, require_role
from stratakb.core.logging import log_tool_call

_PASSAGE_ID = re.compile(r"[Ss]([1-9][0-9]*)")


async def _resolve_passage_ids(
    store, interaction_id: str, chunk_feedback: dict
) -> tuple[dict, str | None]:
    """Replace S<n> keys with the chunk id of the interaction's n-th result.

    Other keys are chunk ids and stay. Returns the resolved feedback and an
    error message, or None.
    """
    if not any(_PASSAGE_ID.fullmatch(str(key)) for key in chunk_feedback):
        return chunk_feedback, None
    results = await store.get_interaction_results(interaction_id)
    if results is None:
        return chunk_feedback, (
            f"Unknown interaction_id '{interaction_id}'. Pass the interaction_id "
            "from a strata_search or strata_ask response."
        )
    resolved: dict = {}
    unknown: list[str] = []
    for key, rating in chunk_feedback.items():
        match = _PASSAGE_ID.fullmatch(str(key))
        if match is None:
            resolved[key] = rating
            continue
        n = int(match.group(1))
        chunk_id = results[n - 1].get("chunk_id") if n <= len(results) else None
        if chunk_id is None:
            unknown.append(str(key))
        else:
            resolved[chunk_id] = rating
    if unknown:
        return chunk_feedback, (
            f"Unknown passage id(s) {', '.join(unknown)} for this interaction: "
            f"it returned {len(results)} passage(s)."
        )
    return resolved, None


def register_learning_tools(mcp: FastMCP, database_url: str) -> None:
    """Register learning system tools on the MCP server."""

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_feedback(
        interaction_id: str,
        helpful: bool,
        correct_layer: str | None = None,
        better_query: str | None = None,
        chunk_feedback: dict | None = None,
    ) -> dict:
        """Submit feedback on a knowledge base query result.

        Use this after strata_ask or strata_search to indicate whether
        the results were helpful. This feedback trains the system to
        return better results for similar future queries.

        Args:
            interaction_id: The interaction_id from the strata_search or strata_ask response.
            helpful: Whether the results answered the question.
            correct_layer: If results were from wrong layer, specify the correct one.
            better_query: If you found better search terms, share them here.
            chunk_feedback: Per-passage feedback: {"S1": "useful"|"wrong"|"outdated"},
                using the passage ids of the strata_ask response, or chunk ids.
        """
        from stratakb.learning.store import LearningStore

        store = LearningStore(database_url)

        # Passage ids (S1, S2, ...) name the interaction's results in order
        if chunk_feedback:
            chunk_feedback, error = await _resolve_passage_ids(
                store, interaction_id, chunk_feedback
            )
            if error:
                return {"status": "error", "error": error}

        # Store explicit helpful/not-helpful signal
        stored = await store.store_feedback(
            interaction_id=interaction_id,
            signal_type="explicit",
            signal_name="helpful" if helpful else "wrong",
            signal_value=1.0 if helpful else -1.0,
        )
        if not stored:
            return {
                "status": "error",
                "error": (
                    f"Unknown interaction_id '{interaction_id}'. Pass the interaction_id "
                    "from a strata_search or strata_ask response."
                ),
            }

        # The audit summary names only the extras that were stored. The
        # suggested query and layer are caller free text: name them only
        extras = []

        # Store layer correction if provided
        if correct_layer and await store.store_feedback(
            interaction_id=interaction_id,
            signal_type="explicit",
            signal_name="correct_layer",
            signal_value=-0.3,
            metadata={"correct_layer": correct_layer},
        ):
            extras.append("correct_layer")

        # Store better query if provided
        if better_query and await store.store_feedback(
            interaction_id=interaction_id,
            signal_type="explicit",
            signal_name="better_query",
            signal_value=-0.2,
            metadata={"better_query": better_query},
        ):
            extras.append("better_query")

        # Store chunk-level feedback
        if chunk_feedback:
            rated = 0
            for chunk_id, rating in chunk_feedback.items():
                value = {"useful": 0.7, "wrong": -1.0, "outdated": -0.5}.get(rating, 0)
                if await store.store_feedback(
                    interaction_id=interaction_id,
                    signal_type="explicit",
                    signal_name=f"chunk_{rating}",
                    signal_value=value,
                    metadata={"chunk_id": chunk_id},
                ):
                    rated += 1
            if rated:
                extras.append(f"{rated} chunk ratings")
        summary = "Rated helpful" if helpful else "Rated not helpful"
        if extras:
            summary += ", with " + ", ".join(extras)
        user_id, _ = get_current_user()
        await write_audit_log(
            database_url,
            user_id=user_id,
            action="feedback",
            tool_name="strata_feedback",
            parameters={"interaction_id": interaction_id, "helpful": helpful},
            result_summary=summary,
        )

        return {"status": "ok", "message": "Feedback recorded. Thank you!"}

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_learning_stats(
        layer: str | None = None,
    ) -> dict:
        """View learning system statistics.

        Shows how many patterns the system has learned, top synonyms,
        success rates, and query volume.

        Args:
            layer: Optional layer name to filter stats.
        """
        from stratakb.learning.store import LearningStore

        store = LearningStore(database_url)
        stats = await store.get_stats(layer=layer)
        return {"status": "ok", **stats}

    @mcp.tool()
    @require_role("admin")
    @log_tool_call
    async def strata_reset_learning(
        layer: str | None = None,
        confirm: bool = False,
    ) -> dict:
        """Reset learning data.

        Clears learned rewrites, synonyms, and affinities. Admin only.
        Interaction history is preserved unless doing a global reset.

        Args:
            layer: Reset only this layer's learned data. Omit for global reset;
                an empty layer is refused.
            confirm: Must be true to proceed. Safety check.
        """
        if layer is not None and not layer.strip():
            return {
                "status": "error",
                "error": "layer is empty. Name a layer, or omit layer for a global reset.",
            }
        if not confirm:
            return {
                "status": "error",
                "error": "Set confirm=true to proceed. This will delete learned patterns.",
            }

        from stratakb.learning.store import LearningStore

        store = LearningStore(database_url)
        result = await store.reset(layer=layer)
        # A reset that failed in the database deleted nothing: report, don't audit
        if result is None:
            return {
                "status": "error",
                "error": "Reset failed in the database; nothing was deleted. "
                "Check the server log and retry.",
            }
        user_id, _ = get_current_user()
        await write_audit_log(
            database_url,
            user_id=user_id,
            action="reset_learning",
            tool_name="strata_reset_learning",
            parameters={"scope": "layer", "layer": layer} if layer else {"scope": "all"},
            result_summary=(
                f"Reset learned data for layer '{layer}'"
                if layer
                else "Reset all learning data, including interaction history"
            ),
        )
        return {"status": "ok", **result}
