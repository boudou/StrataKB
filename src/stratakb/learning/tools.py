"""MCP tool definitions for the StrataKB Learning System.

Registers feedback, stats, and reset tools.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from stratakb.core.auth import require_role
from stratakb.core.logging import log_tool_call


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
            chunk_feedback: Per-chunk feedback: {chunk_id: "useful"|"wrong"|"outdated"}.
        """
        from stratakb.learning.store import LearningStore

        store = LearningStore(database_url)

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

        # Store layer correction if provided
        if correct_layer:
            await store.store_feedback(
                interaction_id=interaction_id,
                signal_type="explicit",
                signal_name="correct_layer",
                signal_value=-0.3,
                metadata={"correct_layer": correct_layer},
            )

        # Store better query if provided
        if better_query:
            await store.store_feedback(
                interaction_id=interaction_id,
                signal_type="explicit",
                signal_name="better_query",
                signal_value=-0.2,
                metadata={"better_query": better_query},
            )

        # Store chunk-level feedback
        if chunk_feedback:
            for chunk_id, rating in chunk_feedback.items():
                value = {"useful": 0.7, "wrong": -1.0, "outdated": -0.5}.get(rating, 0)
                await store.store_feedback(
                    interaction_id=interaction_id,
                    signal_type="explicit",
                    signal_name=f"chunk_{rating}",
                    signal_value=value,
                    metadata={"chunk_id": chunk_id},
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
            layer: Reset only this layer's learned data. Omit for global reset.
            confirm: Must be true to proceed. Safety check.
        """
        if not confirm:
            return {
                "status": "error",
                "error": "Set confirm=true to proceed. This will delete learned patterns.",
            }

        from stratakb.learning.store import LearningStore

        store = LearningStore(database_url)
        result = await store.reset(layer=layer)
        return {"status": "ok", **result}
