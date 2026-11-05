"""Receipt readers: the strata_get_receipt tool and the strata://receipts resource."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from stratakb.core.auth import PermissionDeniedError, check_role, get_current_user, require_role
from stratakb.core.logging import log_tool_call
from stratakb.rag.receipts import ReceiptAccessError, open_receipt

RECEIPT_URI = "strata://receipts/{receipt_id}"


def register_receipt_tools(mcp: FastMCP, database_url: str) -> None:
    """Register the receipt tool and resource on the MCP server."""

    @mcp.tool()
    @require_role("viewer")
    @log_tool_call
    async def strata_get_receipt(receipt_id: str) -> dict:
        """Open a receipt: the stored record of what StrataKB returned or verified.

        Shows each quote in its surrounding document text, with the document
        version, SHA-256, offsets and clause path it came from. Each quote has a
        status: current; changed, with a note naming the newer version ("v1 →
        v2"); or gone, when its file was removed from its folder or its layer
        was deleted. A changed or gone quote still shows the original text. Open
        to the person who asked, to admins and to anyone who may read every
        cited layer.

        Args:
            receipt_id: The receipt id: rcpt_ and 22 URL-safe characters.
        """
        user_id, role = get_current_user()
        try:
            receipt = await open_receipt(
                database_url, receipt_id, user_id, role, tool_name="strata_get_receipt"
            )
        except ReceiptAccessError as e:
            return {"status": "error", "error": str(e)}
        return {"status": "ok", "receipt": receipt}

    # Resources are not wrapped like tools, so the role check is made here
    @mcp.resource(RECEIPT_URI, mime_type="application/json")
    async def strata_receipt(receipt_id: str) -> str:
        """A receipt as JSON, each quote shown in its surrounding document text."""
        user_id, role = get_current_user()
        if not check_role(role, "viewer"):
            raise PermissionDeniedError(user_id, role, "viewer", "strata_receipt")
        receipt = await open_receipt(
            database_url, receipt_id, user_id, role, tool_name="resource:strata_receipt"
        )
        return json.dumps(receipt, ensure_ascii=False)
