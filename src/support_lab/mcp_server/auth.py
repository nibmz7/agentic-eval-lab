"""Dev-only bearer-token auth for the HTTP transport.

In a real deployment the MCP server is an OAuth resource server: an external
authorization server issues tokens and the verifier validates them (signature,
expiry, audience). Here a fixed table of tokens stands in for that, which is
enough to exercise 401s and per-tool scope checks.
"""

from __future__ import annotations

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl

SCOPE_ACCOUNTS_READ = "accounts:read"
SCOPE_REFUNDS_WRITE = "refunds:write"
SCOPE_TICKETS_WRITE = "tickets:write"

DEV_TOKENS: dict[str, tuple[str, list[str]]] = {
    # token: (client_id, scopes)
    "dev-agent": (
        "support-agent",
        [SCOPE_ACCOUNTS_READ, SCOPE_REFUNDS_WRITE, SCOPE_TICKETS_WRITE],
    ),
    "dev-readonly": ("dashboard", [SCOPE_ACCOUNTS_READ]),
}


class DevTokenVerifier:
    def __init__(self, resource_url: str):
        self._resource_url = resource_url

    async def verify_token(self, token: str) -> AccessToken | None:
        entry = DEV_TOKENS.get(token)
        if entry is None:
            return None
        client_id, scopes = entry
        return AccessToken(
            token=token, client_id=client_id, scopes=scopes, resource=self._resource_url
        )


def dev_auth_settings(resource_url: str) -> AuthSettings:
    return AuthSettings(
        # Nothing runs here; a real deployment points at its authorization server.
        issuer_url=AnyHttpUrl("http://localhost:9000"),
        resource_server_url=AnyHttpUrl(resource_url),
        required_scopes=[SCOPE_ACCOUNTS_READ],
        validate_token_resource=True,
    )
