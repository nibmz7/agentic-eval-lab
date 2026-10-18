"""MCP server exposing the support backend as tools, resources and prompts.

Run over stdio:   uv run support-mcp
Run over HTTP:    SUPPORT_TRANSPORT=http uv run support-mcp   (bearer auth required)
Inspect it:       npx @modelcontextprotocol/inspector uv run support-mcp

Tool errors are raised as ToolError with a "<code>: <message>" string. The
client sees `isError: true` and the agent can branch on the code (e.g. switch
to escalate_ticket on `requires_escalation`) instead of the call crashing.
"""

from __future__ import annotations

import dataclasses
import json
import os
from decimal import Decimal, InvalidOperation
from typing import Annotated

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from support_lab.backend import PolicyViolation, SupportStore, UnknownUser
from support_lab.backend.store import REFUND_AUTO_APPROVE_LIMIT_CENTS
from support_lab.mcp_server.auth import SCOPE_REFUNDS_WRITE, DevTokenVerifier, dev_auth_settings

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


def _require_scope(scope: str) -> None:
    token = get_access_token()
    if token is None or scope not in token.scopes:
        raise ToolError(f"forbidden: requires scope {scope!r}")


def build_server(store: SupportStore, *, auth_resource_url: str | None = None) -> MCPServer:
    """Build the server. Pass `auth_resource_url` to require bearer tokens (HTTP only).

    Without it (stdio, in-process tests) there is no caller identity and no
    scope checks. That gap is deliberate for now: see docs/decisions/0002.
    """
    require_auth = auth_resource_url is not None
    server = MCPServer(
        "support-backend",
        instructions=(
            "Customer support backend. Check the balance before refunding. "
            f"Refunds over {_fmt(REFUND_AUTO_APPROVE_LIMIT_CENTS)} cannot be processed "
            "and must be escalated with escalate_ticket. The full rules are in the "
            "policy://refunds resource."
        ),
        token_verifier=DevTokenVerifier(auth_resource_url) if auth_resource_url else None,
        auth=dev_auth_settings(auth_resource_url) if auth_resource_url else None,
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
        if require_auth:
            _require_scope(SCOPE_REFUNDS_WRITE)
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

    # -- resources: read-only context the client app or user chooses to attach --

    @server.resource("policy://refunds", mime_type="text/markdown")
    def refund_policy() -> str:
        """The refund rules the backend enforces."""
        limit = _fmt(REFUND_AUTO_APPROVE_LIMIT_CENTS)
        return (
            "# Refund policy\n\n"
            "- The account must be `active`. Suspended accounts cannot be refunded.\n"
            "- The amount must be positive, with at most 2 decimal places.\n"
            "- The amount must not exceed the current balance.\n"
            f"- Refunds over {limit} are never processed automatically: "
            "escalate them with `escalate_ticket`.\n"
            "- Pass an `idempotency_key` when retrying so a refund is not issued twice.\n"
        )

    @server.resource("account://{user_id}", mime_type="application/json")
    def account(user_id: str) -> str:
        """A customer's account record."""
        try:
            acct = store.get_account(user_id)
        except UnknownUser:
            raise ResourceNotFoundError(f"unknown user_id: {user_id!r}") from None
        return json.dumps(dataclasses.asdict(acct))

    # -- prompts: user-selected templates that start a conversation --

    @server.prompt()
    def handle_refund_request(user_id: str, amount: str) -> str:
        """Start handling a customer's refund request."""
        return (
            f"Customer {user_id} is asking for a refund of {amount}.\n"
            "1. Check their balance and account status with check_balance.\n"
            "2. Follow the rules in policy://refunds.\n"
            "3. If policy allows it, process the refund. Otherwise escalate_ticket "
            "with a clear reason.\n"
            "4. Tell the customer what happened."
        )

    return server


def main() -> None:
    store = SupportStore.seeded(os.environ.get("SUPPORT_DB", ":memory:"))
    transport = os.environ.get("SUPPORT_TRANSPORT", "stdio")
    if transport == "stdio":
        build_server(store).run("stdio")
    elif transport == "http":
        host = os.environ.get("SUPPORT_HOST", "127.0.0.1")
        port = int(os.environ.get("SUPPORT_PORT", "8000"))
        server = build_server(store, auth_resource_url=f"http://{host}:{port}/mcp")
        server.run("streamable-http", host=host, port=port)
    else:
        raise SystemExit(f"SUPPORT_TRANSPORT must be 'stdio' or 'http', got {transport!r}")
