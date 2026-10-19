"""The agent's view of the MCP server: tool schemas, the policy resource, and
tool calls classified into success, policy error, or transient failure."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from mcp import Client
from mcp.types import TextContent, TextResourceContents

from .model import Block

# Error codes the server uses for deliberate refusals. Retrying these can't help.
POLICY_CODES = frozenset(
    {
        "account_not_active",
        "forbidden",
        "idempotency_conflict",
        "insufficient_balance",
        "invalid_amount",
        "invalid_reason",
        "requires_escalation",
        "unknown_user",
    }
)

_SDK_PREFIX = re.compile(r"^Error executing tool [^\s:]+(: )?")
_CODE = re.compile(r"^([a-z]+(?:_[a-z]+)*): ")


@dataclass(frozen=True)
class ToolOutcome:
    text: str
    is_error: bool = False
    code: str | None = None

    @property
    def transient(self) -> bool:
        """An error with no known policy code: a crash, timeout, or lost connection."""
        return self.is_error and self.code not in POLICY_CODES


def classify_error(message: str) -> ToolOutcome:
    detail = _SDK_PREFIX.sub("", message, count=1)
    match = _CODE.match(detail)
    return ToolOutcome(text=message, is_error=True, code=match.group(1) if match else None)


class McpToolbox:
    def __init__(self, client: Client):
        self._client = client

    async def tool_specs(self) -> list[Block]:
        """MCP tool definitions converted to Messages API tool definitions."""
        tools = (await self._client.list_tools()).tools
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in tools
        ]

    async def read_policy(self) -> str:
        contents = (await self._client.read_resource("policy://refunds")).contents
        return "\n".join(c.text for c in contents if isinstance(c, TextResourceContents))

    async def call(self, name: str, args: dict) -> ToolOutcome:
        try:
            result = await self._client.call_tool(name, args)
        except Exception as exc:  # anything between us and the server is transient
            return ToolOutcome(text=f"tool call failed: {exc}", is_error=True)
        message = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        if result.is_error:
            return classify_error(message)
        if result.structured_content is not None:
            message = json.dumps(result.structured_content)
        return ToolOutcome(text=message)
