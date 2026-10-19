"""The agent CLI launches the MCP server without leaking secrets to it."""

import pytest
from mcp import Client
from mcp.client.stdio import get_default_environment

from support_lab.agent.cli import server_params

pytestmark = pytest.mark.anyio


def test_server_subprocess_does_not_inherit_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    params = server_params(":memory:")
    # This is the environment the MCP SDK gives the subprocess.
    child_env = get_default_environment() | (params.env or {})

    assert "ANTHROPIC_API_KEY" not in child_env
    assert "AWS_SECRET_ACCESS_KEY" not in child_env
    assert child_env["SUPPORT_DB"] == ":memory:"
    assert "PATH" in child_env


async def test_server_starts_with_the_minimal_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    async with Client(server_params(":memory:")) as client:
        result = await client.call_tool("check_balance", {"user_id": "u_alice"})
    assert result.structured_content == {
        "user_id": "u_alice",
        "status": "active",
        "balance": "120.00",
    }
