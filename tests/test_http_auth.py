"""Bearer-token auth on the streamable HTTP transport.

Auth lives in HTTP middleware, so the in-process client used elsewhere never
touches it. These tests run a real server on a random local port.
"""

import socket
import threading
import time

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

from support_lab.backend import SupportStore
from support_lab.mcp_server import build_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_server(store: SupportStore):
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    app = build_server(store, auth_resource_url=url).streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    yield url
    server.should_exit = True
    thread.join(timeout=5)


async def _call(url: str, token: str, tool: str, args: dict):
    async with (
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http,
        Client(streamable_http_client(url, http_client=http)) as client,
    ):
        return await client.call_tool(tool, args)


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-real-token"}])
async def test_missing_or_unknown_token_is_rejected(http_server, headers):
    async with httpx2.AsyncClient(headers=headers) as http:
        response = await http.post(http_server, json={})
    assert response.status_code == 401
    assert "oauth-protected-resource" in response.headers["www-authenticate"]


async def test_readonly_token_can_read_but_not_refund(http_server, store):
    balance = await _call(http_server, "dev-readonly", "check_balance", {"user_id": "u_alice"})
    assert not balance.is_error

    before = store.snapshot()
    refund = await _call(
        http_server, "dev-readonly", "process_refund", {"user_id": "u_alice", "amount": 10}
    )
    assert refund.is_error
    (content,) = refund.content
    assert isinstance(content, TextContent)
    assert "forbidden" in content.text
    assert store.snapshot() == before


async def test_agent_token_can_refund(http_server, store):
    refund = await _call(
        http_server, "dev-agent", "process_refund", {"user_id": "u_alice", "amount": 10}
    )
    assert not refund.is_error
    assert store.get_account("u_alice").balance_cents == 110_00
