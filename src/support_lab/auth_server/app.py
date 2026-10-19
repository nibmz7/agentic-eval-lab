"""HTTP app for the demo bank's authorization server.

    uv run support-auth            # http://127.0.0.1:9000

Standard OAuth endpoints come from the MCP SDK (`create_auth_routes`):
discovery metadata, /register, /authorize, /token, /revoke. This module adds
the parts a real bank would own: the sign-in and consent page, and a token
introspection endpoint (RFC 7662) that the MCP server calls to check tokens.
"""

from __future__ import annotations

import html
import os
import time

from mcp.server.auth.routes import create_auth_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from .provider import DEMO_USERS, REQUIRED_SCOPE, SCOPES, DemoBankProvider, LoginError

DEFAULT_ISSUER = "http://127.0.0.1:9000"
# Shared with the MCP server so only it can introspect tokens. Dev default only.
DEFAULT_INTROSPECTION_SECRET = "dev-introspection-secret"

PAGE_STYLE = """
:root { --bg:#eef1f5; --card:#fff; --fg:#17202e; --muted:#5a6475; --line:#cfd6e0;
  --brand:#0b5f7a; --err:#a8361b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#10151c; --card:#18202a; --fg:#e5e9ef; --muted:#9aa4b4; --line:#344050;
    --brand:#4cb3cf; --err:#e8906c; color-scheme: dark; }
}
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; display:grid; place-items:center; background:var(--bg);
  color:var(--fg); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding:24px 16px; }
main { width:100%; max-width:400px; background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding:28px 24px; display:grid; gap:18px; }
.bank { font-weight:700; letter-spacing:.02em; color:var(--brand); }
h1 { font-size:20px; margin:0; }
p { margin:0; color:var(--muted); }
label { display:grid; gap:6px; font-weight:600; font-size:14px; }
input[type=text] { font:inherit; padding:10px 12px; border:1px solid var(--line);
  border-radius:8px; background:var(--bg); color:var(--fg); }
fieldset { border:1px solid var(--line); border-radius:8px; padding:10px 14px; margin:0;
  display:grid; gap:8px; }
legend { font-size:13px; color:var(--muted); padding:0 4px; }
.scope { display:flex; gap:10px; align-items:flex-start; font-weight:400; }
.scope code { font-size:12px; color:var(--muted); }
.hint { font-size:13px; }
.error { color:var(--err); font-weight:600; }
.actions { display:flex; gap:10px; }
button { flex:1; font:inherit; font-weight:600; padding:10px; border-radius:8px; cursor:pointer; }
button[value=approve] { background:var(--brand); color:#fff; border:1px solid var(--brand); }
button[value=deny] { background:transparent; color:var(--fg); border:1px solid var(--line); }
button:focus-visible, input:focus-visible { outline:2px solid var(--brand); outline-offset:2px; }
"""


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f'<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{PAGE_STYLE}</style></head>"
        f"<body><main><div class=bank>Demo Bank</div>{body}</main></body></html>",
        status_code=status,
    )


def _login_form(
    provider: DemoBankProvider, txn: str, error: str = "", username: str = ""
) -> HTMLResponse:
    login = provider.get_pending(txn)
    app_name = html.escape(login.client.client_name or "An app")
    scopes = "".join(
        f'<label class=scope><input type=checkbox name=scope value="{s}" checked'
        f"{' disabled' if s == REQUIRED_SCOPE else ''}>"
        f"<span>{html.escape(SCOPES[s])}<br><code>{s}</code></span></label>"
        for s in login.requested_scopes
    )
    users = ", ".join(f"<code>{u}</code>" for u in DEMO_USERS)
    error_html = f"<p class=error role=alert>{html.escape(error)}</p>" if error else ""
    return _page(
        "Sign in to Demo Bank",
        f"""<h1>Sign in</h1>
        <p><strong>{app_name}</strong> wants to access your Demo Bank account.</p>
        {error_html}
        <form method=post action=/login>
          <input type=hidden name=txn value="{html.escape(txn)}">
          <div style="display:grid;gap:18px">
            <label>Username<input type=text name=username id=username
              value="{html.escape(username)}" autocomplete=username autofocus></label>
            <p class=hint>Demo accounts: {users}. No password in this demo.</p>
            <fieldset><legend>Allow {app_name} to</legend>{scopes}</fieldset>
            <div class=actions>
              <button type=submit name=action value=deny>Deny</button>
              <button type=submit name=action value=approve>Allow</button>
            </div>
          </div>
        </form>""",
        status=400 if error else 200,
    )


def build_auth_app(
    issuer_url: str = DEFAULT_ISSUER,
    introspection_secret: str = DEFAULT_INTROSPECTION_SECRET,
    provider: DemoBankProvider | None = None,
) -> Starlette:
    provider = provider or DemoBankProvider(issuer_url)

    async def login_page(request: Request) -> Response:
        try:
            return _login_form(provider, request.query_params.get("txn", ""))
        except LoginError as exc:
            return _page("Sign-in expired", f"<h1>Sign-in expired</h1><p>{exc}</p>", 400)

    async def login_submit(request: Request) -> Response:
        form = await request.form()
        txn = str(form.get("txn", ""))
        username = str(form.get("username", "")).strip()
        try:
            if form.get("action") == "deny":
                return RedirectResponse(provider.deny(txn), status_code=302)
            chosen = [str(s) for s in form.getlist("scope")]
            return RedirectResponse(provider.approve(txn, username, chosen), status_code=302)
        except LoginError as exc:
            try:
                return _login_form(provider, txn, error=str(exc), username=username)
            except LoginError:
                return _page("Sign-in expired", f"<h1>Sign-in expired</h1><p>{exc}</p>", 400)

    async def introspect(request: Request) -> Response:
        if request.headers.get("authorization") != f"Bearer {introspection_secret}":
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        token = str((await request.form()).get("token", ""))
        access = await provider.load_access_token(token)
        if access is None:
            return JSONResponse({"active": False})
        return JSONResponse(
            {
                "active": True,
                "client_id": access.client_id,
                "scope": " ".join(access.scopes),
                "sub": access.subject,
                "aud": access.resource,
                "exp": access.expires_at,
                "iat": int(time.time()),
                "iss": provider.issuer_url,
                "token_type": "Bearer",
            }
        )

    routes = create_auth_routes(
        provider,
        AnyHttpUrl(issuer_url),
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=list(SCOPES), default_scopes=list(SCOPES)
        ),
        revocation_options=RevocationOptions(enabled=True),
    )
    routes += [
        Route("/login", login_page, methods=["GET"]),
        Route("/login", login_submit, methods=["POST"]),
        Route("/introspect", introspect, methods=["POST"]),
    ]
    app = Starlette(routes=routes)
    app.state.provider = provider
    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("SUPPORT_AUTH_HOST", "127.0.0.1")
    port = int(os.environ.get("SUPPORT_AUTH_PORT", "9000"))
    issuer = f"http://{host}:{port}"
    secret = os.environ.get("SUPPORT_INTROSPECTION_SECRET", DEFAULT_INTROSPECTION_SECRET)
    uvicorn.run(build_auth_app(issuer, secret), host=host, port=port, log_level="info")
