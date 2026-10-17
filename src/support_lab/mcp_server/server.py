"""MCP server exposing the support backend as tools.

Run over stdio:   uv run support-mcp
Inspect it:       npx @modelcontextprotocol/inspector uv run support-mcp

Tool errors are raised as ToolError with a "<code>: <message>" string. The
client sees `isError: true` and the agent can branch on the code (e.g. switch
to escalate_ticket on `requires_escalation`) instead of the call crashing.
"""

from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from support_lab.backend import PolicyViolation, SupportStore, UnknownUser
from support_lab.backend.store import REFUND_AUTO_APPROVE_LIMIT_CENTS

UserId = Annotated[str, Field(description="Customer id, e.g. 'u_alice'.")]


class Balance(BaseModel):
    user_id: str
    status: str
    balance: str = Field(description="Account balance as a decimal string, e.g. '120.00'.")


class RefundResult(BaseModel):
    refund_id: int
    new_balance: str
    replayed: bool = Field(description="True if this idempotency_key was already processed.")


class TicketResult(BaseModel):
    ticket_id: int
    status: str


def _fmt(cents: int) -> str:
    return f"{Decimal(cents) / 100:.2f}"


def _to_cents(amount: float | str) -> int:
    try:
        value = Decimal(str(amount))
    except InvalidOperation:
        raise ToolError(f"invalid_amount: {amount!r} is not a number") from None
    if value != value.quantize(Decimal("0.01")):
        raise ToolError("invalid_amount: at most 2 decimal places")
    return int(value * 100)


def build_server(store: SupportStore) -> MCPServer:
    server = MCPServer(
        "support-backend",
        instructions=(
            "Customer support backend. Check the balance before refunding. "
            f"Refunds over {_fmt(REFUND_AUTO_APPROVE_LIMIT_CENTS)} cannot be processed "
            "and must be escalated with escalate_ticket."
        ),
    )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def check_balance(user_id: UserId) -> Balance:
        """Return the account status and current balance for a customer."""
        try:
            acct = store.get_account(user_id)
        except UnknownUser as exc:
            raise ToolError(f"unknown_user: {exc}") from None
        return Balance(user_id=acct.user_id, status=acct.status, balance=_fmt(acct.balance_cents))

    @server.tool(annotations=ToolAnnotations(destructive_hint=True, idempotent_hint=True))
    def process_refund(
        user_id: UserId,
        amount: Annotated[float | str, Field(description="Refund amount in dollars, e.g. 25.50.")],
        idempotency_key: Annotated[
            str | None,
            Field(description="Optional. Retrying with the same key will not refund twice."),
        ] = None,
    ) -> RefundResult:
        """Refund `amount` from the customer's balance.

        Fails with requires_escalation above the auto-approve limit, and with
        insufficient_balance or account_not_active when policy forbids it.
        """
        try:
            r = store.process_refund(user_id, _to_cents(amount), idempotency_key)
        except PolicyViolation as exc:
            raise ToolError(f"{exc.code}: {exc}") from None
        except UnknownUser as exc:
            raise ToolError(f"unknown_user: {exc}") from None
        return RefundResult(
            refund_id=r["refund_id"],
            new_balance=_fmt(r["new_balance_cents"]),
            replayed=r["replayed"],
        )

    @server.tool()
    def escalate_ticket(
        user_id: UserId,
        reason: Annotated[str, Field(description="Why a human needs to handle this.")],
    ) -> TicketResult:
        """Open a ticket for a human agent.

        Use when policy blocks an action or the request is unclear.
        """
        try:
            r = store.escalate_ticket(user_id, reason)
        except PolicyViolation as exc:
            raise ToolError(f"{exc.code}: {exc}") from None
        except UnknownUser as exc:
            raise ToolError(f"unknown_user: {exc}") from None
        return TicketResult(**r)

    return server


def main() -> None:
    store = SupportStore.seeded(os.environ.get("SUPPORT_DB", ":memory:"))
    build_server(store).run("stdio")
