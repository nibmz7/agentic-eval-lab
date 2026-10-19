"""A pretend bank's OAuth authorization server.

This plays the part of the bank's login system. The MCP server never sees a
password: the user signs in here, approves what the app may do, and the app
receives a short-lived access token naming the user (`subject`) and the MCP
server it may be used with (`resource`).

Sign-in is just a username from DEMO_USERS. Everything else follows the real
flow: dynamic client registration, authorization code + PKCE (checked by the
SDK's token handler), one-time codes, refresh-token rotation, revocation.
State is in memory, so restarting the server signs everyone out.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

DEMO_USERS = {
    "u_alice": "Alice Tan",
    "u_bob": "Bob Lim",
    "u_carol": "Carol Ng",
    "u_dave": "Dave Koh",
}

# What each scope lets the app do, as shown on the consent screen.
SCOPES = {
    "accounts:read": "See your account status and balance",
    "refunds:write": "Issue refunds from your balance",
    "tickets:write": "Open support tickets on your behalf",
}
REQUIRED_SCOPE = "accounts:read"

ACCESS_TOKEN_TTL = 15 * 60
REFRESH_TOKEN_TTL = 24 * 60 * 60
CODE_TTL = 5 * 60
LOGIN_TTL = 10 * 60


@dataclass
class PendingLogin:
    client: OAuthClientInformationFull
    params: AuthorizationParams
    expires_at: float

    @property
    def requested_scopes(self) -> list[str]:
        return [s for s in (self.params.scopes or list(SCOPES)) if s in SCOPES]


class LoginError(Exception):
    pass


class DemoBankProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    def __init__(self, issuer_url: str):
        self.issuer_url = issuer_url.rstrip("/")
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.pending: dict[str, PendingLogin] = {}
        self.codes: dict[str, AuthorizationCode] = {}
        self.access_tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, RefreshToken] = {}
        self._pairs: dict[str, str] = {}  # access <-> refresh, both directions

    # -- clients -----------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        assert client_info.client_id is not None
        self.clients[client_info.client_id] = client_info

    # -- authorization: hand off to the login page ---------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        txn = secrets.token_urlsafe(24)
        self.pending[txn] = PendingLogin(client, params, time.time() + LOGIN_TTL)
        return f"{self.issuer_url}/login?txn={txn}"

    def get_pending(self, txn: str) -> PendingLogin:
        login = self.pending.get(txn)
        if login is None or login.expires_at < time.time():
            raise LoginError("This sign-in link has expired. Start again from the app.")
        return login

    def approve(self, txn: str, username: str, chosen_scopes: list[str]) -> str:
        """Finish a sign-in: issue a one-time code and return the app's redirect URL."""
        login = self.get_pending(txn)
        if username not in DEMO_USERS:
            raise LoginError(f"No account named {username!r}.")
        del self.pending[txn]
        granted = [s for s in login.requested_scopes if s in chosen_scopes or s == REQUIRED_SCOPE]
        code = secrets.token_urlsafe(32)
        self.codes[code] = AuthorizationCode(
            code=code,
            scopes=granted,
            expires_at=time.time() + CODE_TTL,
            client_id=login.client.client_id or "",
            code_challenge=login.params.code_challenge,
            redirect_uri=login.params.redirect_uri,
            redirect_uri_provided_explicitly=login.params.redirect_uri_provided_explicitly,
            resource=login.params.resource,
            subject=username,
        )
        return construct_redirect_uri(
            str(login.params.redirect_uri), code=code, state=login.params.state
        )

    def deny(self, txn: str) -> str:
        login = self.get_pending(txn)
        del self.pending[txn]
        return construct_redirect_uri(
            str(login.params.redirect_uri),
            error="access_denied",
            error_description="The user declined access",
            state=login.params.state,
        )

    # -- tokens ------------------------------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self.codes.get(authorization_code)
        if code is None or code.client_id != client.client_id or code.expires_at < time.time():
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # Codes are single use: remove it before issuing anything.
        if self.codes.pop(authorization_code.code, None) is None:
            raise TokenError("invalid_grant", "authorization code already used")
        return self._issue(
            authorization_code.client_id,
            authorization_code.scopes,
            authorization_code.resource,
            authorization_code.subject,
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        token = self.refresh_tokens.get(refresh_token)
        if token is None or token.client_id != client.client_id:
            return None
        if token.expires_at is not None and token.expires_at < time.time():
            return None
        return token

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        if not set(scopes) <= set(refresh_token.scopes):
            raise TokenError("invalid_scope", "cannot widen scopes on refresh")
        # Rotate: the old refresh token (and its access token) stop working.
        await self.revoke_token(refresh_token)
        return self._issue(
            refresh_token.client_id,
            scopes or refresh_token.scopes,
            refresh_token.resource,
            refresh_token.subject,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        access = self.access_tokens.get(token)
        if access is None or (access.expires_at is not None and access.expires_at < time.time()):
            return None
        return access

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        for value in (token.token, self._pairs.get(token.token)):
            if value is not None:
                self.access_tokens.pop(value, None)
                self.refresh_tokens.pop(value, None)
                self._pairs.pop(value, None)

    def _issue(
        self, client_id: str, scopes: list[str], resource: str | None, subject: str | None
    ) -> OAuthToken:
        now = int(time.time())
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        self.access_tokens[access] = AccessToken(
            token=access,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + ACCESS_TOKEN_TTL,
            resource=resource,
            subject=subject,
        )
        self.refresh_tokens[refresh] = RefreshToken(
            token=refresh,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + REFRESH_TOKEN_TTL,
            resource=resource,
            subject=subject,
        )
        self._pairs[access], self._pairs[refresh] = refresh, access
        return OAuthToken(
            access_token=access,
            expires_in=ACCESS_TOKEN_TTL,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )
