"""Sign-in through the demo bank, end to end, against real running servers.

`bank` starts the authorization server (`support-auth`) and the MCP server in
HTTP mode. `sign_in` does what a browser would: register an app, start the
authorization request with PKCE, submit the login form, and exchange the code
for tokens.
"""

import base64
import hashlib
import secrets
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import httpx2
import pytest
from conftest import free_port
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp.types import TextContent

from support_lab.agent.login import MemoryTokenStorage, oauth_provider
from support_lab.auth_server import build_auth_app
from support_lab.backend import SupportStore
from support_lab.mcp_server import build_server
from support_lab.mcp_server.auth import OAuthConfig
from support_lab.mcp_server.server import http_app

pytestmark = pytest.mark.anyio

SECRET = "test-introspection-secret"
REDIRECT_URI = "http://127.0.0.1:1/callback"


@dataclass
class Bank:
    issuer: str
    mcp_url: str
    store: SupportStore


@pytest.fixture
def bank(run_app, store) -> Bank:
    as_port, mcp_port = free_port(), free_port()
    issuer, mcp_url = f"http://127.0.0.1:{as_port}", f"http://127.0.0.1:{mcp_port}/mcp"
    run_app(build_auth_app(issuer, SECRET), as_port)
    oauth = OAuthConfig(resource_url=mcp_url, issuer_url=issuer, introspection_secret=SECRET)
    run_app(http_app(build_server(store, oauth=oauth), oauth), mcp_port)
    return Bank(issuer, mcp_url, store)


def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


@dataclass
class Authorization:
    client_id: str
    verifier: str
    redirect: str  # where the login form sent the browser

    @property
    def params(self) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(urlparse(self.redirect).query).items()}


async def authorize(
    bank: Bank,
    username: str,
    *,
    action: str = "approve",
    scopes: list[str] | None = None,
    resource: str | None = None,
) -> Authorization:
    """Everything up to the redirect back to the app: register, /authorize, login form."""
    async with httpx2.AsyncClient(base_url=bank.issuer) as http:
        registration = await http.post(
            "/register",
            json={
                "client_name": "Test app",
                "redirect_uris": [REDIRECT_URI],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        client_id = registration.json()["client_id"]
        verifier, challenge = pkce()
        start = await http.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "state-123",
                "scope": "accounts:read refunds:write tickets:write",
                "resource": resource or bank.mcp_url,
            },
        )
        assert start.status_code == 302
        login_url = start.headers["location"]
        page = await http.get(login_url)
        assert page.status_code == 200 and "Demo Bank" in page.text
        txn = parse_qs(urlparse(login_url).query)["txn"][0]
        chosen = scopes if scopes is not None else ["refunds:write", "tickets:write"]
        submit = await http.post(
            "/login",
            data={"txn": txn, "username": username, "action": action, "scope": chosen},
        )
        assert submit.status_code == 302, submit.text
        return Authorization(client_id, verifier, submit.headers["location"])


async def exchange(bank: Bank, auth: Authorization, *, verifier: str | None = None):
    async with httpx2.AsyncClient(base_url=bank.issuer) as http:
        return await http.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": auth.params["code"],
                "redirect_uri": REDIRECT_URI,
                "client_id": auth.client_id,
                "code_verifier": verifier or auth.verifier,
                "resource": bank.mcp_url,
            },
        )


async def sign_in(bank: Bank, username: str, **kwargs) -> dict:
    auth = await authorize(bank, username, **kwargs)
    response = await exchange(bank, auth)
    assert response.status_code == 200, response.text
    return response.json()


async def call(bank: Bank, access_token: str, tool: str, args: dict):
    async with (
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {access_token}"}) as http,
        Client(streamable_http_client(bank.mcp_url, http_client=http)) as client,
    ):
        return await client.call_tool(tool, args)


def text_of(result) -> str:
    (content,) = result.content
    assert isinstance(content, TextContent)
    return content.text


# -- discovery ----------------------------------------------------------------


async def test_unauthenticated_request_points_to_the_bank(bank):
    async with httpx2.AsyncClient() as http:
        response = await http.post(bank.mcp_url, json={})
        assert response.status_code == 401
        metadata_url = response.headers["www-authenticate"].split('resource_metadata="')[1][:-1]
        resource = (await http.get(metadata_url)).json()
        server = (await http.get(f"{bank.issuer}/.well-known/oauth-authorization-server")).json()
    assert resource["resource"] == bank.mcp_url
    assert resource["scopes_supported"] == ["accounts:read", "refunds:write", "tickets:write"]
    assert resource["authorization_servers"] == [f"{bank.issuer}/"]
    assert server["code_challenge_methods_supported"] == ["S256"]
    assert server["registration_endpoint"].endswith("/register")


# -- the sign-in flow ------------------------------------------------------------


async def test_sign_in_and_use_the_token(bank):
    auth = await authorize(bank, "u_alice")
    assert auth.params["state"] == "state-123"

    tokens = (await exchange(bank, auth)).json()

    assert set(tokens["scope"].split()) == {"accounts:read", "refunds:write", "tickets:write"}
    result = await call(bank, tokens["access_token"], "check_balance", {"user_id": "u_alice"})
    assert result.structured_content == {
        "user_id": "u_alice",
        "status": "active",
        "balance": "120.00",
    }


async def test_wrong_pkce_verifier_is_rejected(bank):
    auth = await authorize(bank, "u_alice")
    response = await exchange(bank, auth, verifier=secrets.token_urlsafe(48))
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_grant"


async def test_code_can_only_be_used_once(bank):
    auth = await authorize(bank, "u_alice")
    assert (await exchange(bank, auth)).status_code == 200
    second = await exchange(bank, auth)
    assert second.status_code == 400


async def test_deny_returns_access_denied_to_the_app(bank):
    auth = await authorize(bank, "u_alice", action="deny")
    assert auth.params == {"error": "access_denied", "state": "state-123",
                           "error_description": "The user declined access"}  # fmt: skip


async def test_unknown_username_stays_on_the_login_page(bank):
    with pytest.raises(AssertionError, match="No account named"):
        await authorize(bank, "u_mallory")


async def test_refresh_rotates_tokens(bank):
    tokens = await sign_in(bank, "u_alice")
    client_id = None
    async with httpx2.AsyncClient(base_url=bank.issuer) as http:
        # The refresh grant needs the same client; find it from introspection.
        info = await http.post(
            "/introspect",
            data={"token": tokens["access_token"]},
            headers={"Authorization": f"Bearer {SECRET}"},
        )
        client_id = info.json()["client_id"]
        refresh = {"grant_type": "refresh_token", "client_id": client_id,
                   "refresh_token": tokens["refresh_token"]}  # fmt: skip
        first = await http.post("/token", data=refresh)
        reused = await http.post("/token", data=refresh)
    assert first.status_code == 200
    assert reused.status_code == 400  # the old refresh token was rotated out
    old = await call_status(bank, tokens["access_token"])
    assert old == 401  # and so was the old access token
    result = await call(bank, first.json()["access_token"], "check_balance", {"user_id": "u_alice"})
    assert not result.is_error


async def call_status(bank: Bank, access_token: str) -> int:
    async with httpx2.AsyncClient() as http:
        response = await http.post(
            bank.mcp_url, json={}, headers={"Authorization": f"Bearer {access_token}"}
        )
    return response.status_code


async def test_introspection_requires_the_shared_secret(bank):
    tokens = await sign_in(bank, "u_alice")
    async with httpx2.AsyncClient() as http:
        response = await http.post(
            f"{bank.issuer}/introspect", data={"token": tokens["access_token"]}
        )
    assert response.status_code == 401


async def test_token_for_another_server_is_rejected(bank):
    # Audience binding: a token issued for a different MCP server must not work here.
    auth = await authorize(bank, "u_alice", resource="http://127.0.0.1:1/other-mcp")
    async with httpx2.AsyncClient(base_url=bank.issuer) as http:
        response = await http.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": auth.params["code"],
                "redirect_uri": REDIRECT_URI,
                "client_id": auth.client_id,
                "code_verifier": auth.verifier,
            },
        )
    assert response.status_code == 200
    assert await call_status(bank, response.json()["access_token"]) == 401


# -- what the token allows ----------------------------------------------------------


async def test_signed_in_customer_cannot_touch_another_account(bank):
    tokens = await sign_in(bank, "u_alice")
    before = bank.store.snapshot()

    balance = await call(bank, tokens["access_token"], "check_balance", {"user_id": "u_bob"})
    refund = await call(
        bank, tokens["access_token"], "process_refund", {"user_id": "u_bob", "amount": 10}
    )

    assert balance.is_error and "signed in as u_alice, cannot act on u_bob" in text_of(balance)
    assert refund.is_error and "forbidden" in text_of(refund)
    assert bank.store.snapshot() == before


async def test_scopes_unticked_on_consent_are_not_granted(bank):
    tokens = await sign_in(bank, "u_alice", scopes=[])  # only the required accounts:read
    assert tokens["scope"] == "accounts:read"
    before = bank.store.snapshot()

    refund = await call(
        bank, tokens["access_token"], "process_refund", {"user_id": "u_alice", "amount": 10}
    )
    ticket = await call(
        bank, tokens["access_token"], "escalate_ticket", {"user_id": "u_alice", "reason": "x"}
    )

    assert "requires scope 'refunds:write'" in text_of(refund)
    assert "requires scope 'tickets:write'" in text_of(ticket)
    assert bank.store.snapshot() == before


async def test_account_resource_and_session_follow_the_token(bank):
    tokens = await sign_in(bank, "u_bob")
    async with (
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {tokens['access_token']}"}) as http,
        Client(streamable_http_client(bank.mcp_url, http_client=http)) as client,
    ):
        session = await client.read_resource("session://customer")
        own = await client.read_resource("account://u_bob")
        with pytest.raises(MCPError, match="cannot act on u_alice"):
            await client.read_resource("account://u_alice")
    assert '"customer_id": "u_bob"' in session.contents[0].text  # type: ignore[union-attr]
    assert '"user_id": "u_bob"' in own.contents[0].text  # type: ignore[union-attr]


# -- the CLI's client: the SDK's OAuth provider with a scripted "browser" -----------


async def test_cli_login_flow_with_the_sdk_client(bank):
    async def browser(authorization_url: str) -> None:
        # What the user does in the browser: open the link, sign in as u_bob, allow.
        async with httpx2.AsyncClient(follow_redirects=False) as http:
            start = await http.get(authorization_url)
            login_url = start.headers["location"]
            txn = parse_qs(urlparse(login_url).query)["txn"][0]
            submit = await http.post(
                f"{bank.issuer}/login",
                data={
                    "txn": txn,
                    "username": "u_bob",
                    "action": "approve",
                    "scope": ["refunds:write", "tickets:write"],
                },  # fmt: skip
            )
            await http.get(submit.headers["location"])  # the redirect back to the CLI

    storage = MemoryTokenStorage()
    auth = oauth_provider(bank.mcp_url, callback_port=free_port(), open_url=browser,
                          storage=storage)  # fmt: skip
    async with (
        httpx2.AsyncClient(auth=auth) as http,
        Client(streamable_http_client(bank.mcp_url, http_client=http)) as client,
    ):
        session = await client.read_resource("session://customer")
        result = await client.call_tool("process_refund", {"user_id": "u_bob", "amount": 25})

    assert '"customer_id": "u_bob"' in session.contents[0].text  # type: ignore[union-attr]
    assert not result.is_error
    assert bank.store.get_account("u_bob").balance_cents == 1_475_00
    assert storage.tokens is not None and storage.tokens.refresh_token
