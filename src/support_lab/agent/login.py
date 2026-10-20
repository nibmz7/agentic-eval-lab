"""Browser sign-in for the agent CLI, the way a desktop app does it.

The MCP SDK's `OAuthClientProvider` runs the OAuth protocol: it finds the
authorization server from the MCP server's 401, registers this app, builds the
PKCE challenge, exchanges the code for tokens and refreshes them. This module
supplies the two app-specific parts: opening the browser, and a one-shot local
web server that receives the redirect back (`http://127.0.0.1:<port>/callback`).
"""

from __future__ import annotations

import asyncio
import webbrowser
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider, OAuthFlowError
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

SCOPES = "accounts:read refunds:write tickets:write"
LOGIN_TIMEOUT = 300

_DONE_PAGE = (
    "<!doctype html><meta charset=utf-8><title>Signed in</title>"
    "<body style='font:16px system-ui;padding:40px'><h1>{title}</h1>"
    "<p>You can close this tab and go back to the terminal.</p></body>"
)


class MemoryTokenStorage:
    """Keeps tokens for this process only. A real app would use the OS keychain."""

    def __init__(self) -> None:
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_info = client_info


class LoopbackCallback:
    """Listens on 127.0.0.1:<port> for the authorization server's redirect."""

    def __init__(self, port: int):
        self.port = port
        self.redirect_uri = f"http://127.0.0.1:{port}/callback"
        self._result: asyncio.Future[AuthorizationCodeResult] | None = None
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._result = asyncio.get_running_loop().create_future()
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)

    async def wait(self) -> AuthorizationCodeResult:
        assert self._result is not None, "start() first"
        try:
            return await asyncio.wait_for(self._result, LOGIN_TIMEOUT)
        finally:
            if self._server is not None:
                self._server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_line = (await reader.readline()).decode("latin-1")
        while (await reader.readline()) not in (b"\r\n", b"\n", b""):
            pass  # skip headers
        parts = request_line.split()
        url = urlparse(parts[1] if len(parts) > 1 else "/")
        if url.path != "/callback":
            await _respond(writer, 404, "Not found")
            return
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        assert self._result is not None
        if "code" in query:
            title = "Signed in"
            if not self._result.done():
                self._result.set_result(
                    AuthorizationCodeResult(code=query["code"], state=query.get("state"))
                )
        else:
            title = "Sign-in was cancelled"
            if not self._result.done():
                error = query.get("error_description") or query.get("error", "no code returned")
                self._result.set_exception(OAuthFlowError(f"sign-in failed: {error}"))
        await _respond(writer, 200, _DONE_PAGE.format(title=title))


async def _respond(writer: asyncio.StreamWriter, status: int, body: str) -> None:
    data = body.encode()
    reason = "OK" if status == 200 else "Not Found"
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode()
        + data
    )
    await writer.drain()
    writer.close()


def _open_browser(url: str) -> None:
    print(f"\nOpening your browser to sign in. If it doesn't open, visit:\n  {url}\n")
    webbrowser.open(url)


def oauth_provider(
    server_url: str,
    *,
    callback_port: int = 8765,
    open_url: Callable[[str], None | Awaitable[None]] = _open_browser,
    storage: MemoryTokenStorage | None = None,
) -> OAuthClientProvider:
    """An httpx2 auth handler that signs in through the browser when needed."""
    callback = LoopbackCallback(callback_port)

    async def redirect_handler(authorization_url: str) -> None:
        await callback.start()  # listen before the browser can redirect back
        opened = open_url(authorization_url)
        if opened is not None:
            await opened

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=OAuthClientMetadata(
            client_name="Support agent (terminal)",
            redirect_uris=[AnyUrl(callback.redirect_uri)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=SCOPES,
            token_endpoint_auth_method="none",
        ),
        storage=storage or MemoryTokenStorage(),
        redirect_handler=redirect_handler,
        callback_handler=callback.wait,
    )
