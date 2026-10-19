"""Exercise the tools through a real MCP client, in-process.

This tests the contract an agent actually sees: tool names, JSON schemas,
structured output, and isError results.
"""

import dataclasses
import json

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from support_lab.mcp_server import build_server

pytestmark = pytest.mark.anyio


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


# -- resources --------------------------------------------------------------


async def test_policy_resource_is_listed_and_matches_backend_limit(client):
    resources = (await client.list_resources()).resources
    assert [(r.uri, r.mime_type) for r in resources] == [("policy://refunds", "text/markdown")]
    text = (await client.read_resource("policy://refunds")).contents[0].text
    assert "500.00" in text


async def test_account_template_returns_backend_record(client, store):
    templates = (await client.list_resource_templates()).resource_templates
    assert [t.uri_template for t in templates] == ["account://{user_id}"]
    contents = (await client.read_resource("account://u_alice")).contents[0]
    assert json.loads(contents.text) == dataclasses.asdict(store.get_account("u_alice"))


async def test_account_template_unknown_user_is_a_protocol_error(client):
    with pytest.raises(MCPError, match="unknown user_id"):
        await client.read_resource("account://u_ghost")


async def test_reading_resources_does_not_change_state(client, store):
    before = store.snapshot()
    await client.read_resource("policy://refunds")
    await client.read_resource("account://u_bob")
    assert store.snapshot() == before


# -- prompts ----------------------------------------------------------------


async def test_refund_prompt_declares_required_arguments(client):
    (prompt,) = (await client.list_prompts()).prompts
    assert prompt.name == "handle_refund_request"
    assert {(a.name, a.required) for a in prompt.arguments or []} == {
        ("user_id", True),
        ("amount", True),
    }


async def test_refund_prompt_renders_a_user_message(client):
    result = await client.get_prompt("handle_refund_request", {"user_id": "u_bob", "amount": "900"})
    (message,) = result.messages
    assert message.role == "user"
    assert "u_bob" in message.content.text and "900" in message.content.text
