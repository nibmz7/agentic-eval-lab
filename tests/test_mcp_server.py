"""Exercise the tools through a real MCP client, in-process.

This tests the contract an agent actually sees: tool names, JSON schemas,
structured output, and isError results.
"""

import pytest
from mcp import Client

from support_lab.mcp_server import build_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(store):
    async with Client(build_server(store)) as c:
        yield c


async def test_lists_expected_tools_with_schemas(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"check_balance", "process_refund", "escalate_ticket"}
    refund_schema = tools["process_refund"].input_schema
    assert refund_schema["required"] == ["user_id", "amount"]
    assert tools["check_balance"].annotations.read_only_hint is True


async def test_check_balance_returns_structured_content(client):
    result = await client.call_tool("check_balance", {"user_id": "u_alice"})
    assert not result.is_error
    assert result.structured_content == {
        "user_id": "u_alice",
        "status": "active",
        "balance": "120.00",
    }


async def test_refund_accepts_dollars_and_mutates_backend(client, store):
    result = await client.call_tool("process_refund", {"user_id": "u_alice", "amount": 25.5})
    assert result.structured_content["new_balance"] == "94.50"
    assert store.get_account("u_alice").balance_cents == 94_50


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"user_id": "u_bob", "amount": 900}, "requires_escalation"),
        ({"user_id": "u_alice", "amount": 1000}, "requires_escalation"),
        ({"user_id": "u_alice", "amount": 200}, "insufficient_balance"),
        ({"user_id": "u_alice", "amount": "1.005"}, "invalid_amount"),
        ({"user_id": "u_alice", "amount": "ten"}, "invalid_amount"),
        ({"user_id": "u_ghost", "amount": 1}, "unknown_user"),
    ],
)
async def test_policy_failures_surface_as_tool_errors(client, store, args, code):
    before = store.snapshot()
    result = await client.call_tool("process_refund", args)
    assert result.is_error
    assert code in result.content[0].text
    assert store.snapshot() == before


async def test_escalate_ticket(client, store):
    result = await client.call_tool(
        "escalate_ticket", {"user_id": "u_bob", "reason": "refund of 900"}
    )
    assert result.structured_content == {"ticket_id": 1, "status": "open"}
