"""OAuth for the HTTP transport: the MCP server as a resource server.

The demo bank's authorization server (`support-auth`) signs users in and
issues tokens. This server never sees a password. For each request it asks
the authorization server whether the bearer token is valid (RFC 7662 token
introspection) and learns who the token belongs to (`sub`), what it allows
(`scope`) and which server it was issued for (`aud`). The SDK's middleware
rejects tokens issued for a different server.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx2
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl

SCOPE_ACCOUNTS_READ = "accounts:read"
SCOPE_REFUNDS_WRITE = "refunds:write"
SCOPE_TICKETS_WRITE = "tickets:write"
ALL_SCOPES = [SCOPE_ACCOUNTS_READ, SCOPE_REFUNDS_WRITE, SCOPE_TICKETS_WRITE]


@dataclass(frozen=True)
class OAuthConfig:
    resource_url: str  # this MCP server, e.g. http://127.0.0.1:8000/mcp
    issuer_url: str  # the authorization server, e.g. http://127.0.0.1:9000
    introspection_secret: str


class IntrospectionTokenVerifier:
    def __init__(self, config: OAuthConfig):
        self._url = f"{config.issuer_url.rstrip('/')}/introspect"
        self._secret = config.introspection_secret

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            async with httpx2.AsyncClient(timeout=5) as http:
                response = await http.post(
                    self._url,
                    data={"token": token},
                    headers={"Authorization": f"Bearer {self._secret}"},
                )
        except httpx2.HTTPError:
            return None  # can't confirm the token, so treat it as invalid
        if response.status_code != 200:
            return None
        data = response.json()
        if not data.get("active"):
            return None
        return AccessToken(
            token=token,
            client_id=data["client_id"],
            scopes=data.get("scope", "").split(),
            expires_at=data.get("exp"),
            resource=data.get("aud"),
            subject=data.get("sub"),
        )


def auth_settings(config: OAuthConfig) -> AuthSettings:
    return AuthSettings(
        issuer_url=AnyHttpUrl(config.issuer_url),
        resource_server_url=AnyHttpUrl(config.resource_url),
        required_scopes=[SCOPE_ACCOUNTS_READ],
        validate_token_resource=True,
    )
