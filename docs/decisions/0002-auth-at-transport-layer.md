# 0002: Auth at the transport layer, for now

**Status:** superseded by [0004](0004-oauth-sign-in.md) (Sprint 2.5). The dev tokens are gone; the transport-only gap described below still applies over stdio.

## Context

Over streamable HTTP the MCP server acts as an OAuth resource server: requests carry a bearer token, middleware verifies it, and a tool can read the caller's scopes with `get_access_token()`. Over stdio, and with the in-process `Client` used in tests, there is no HTTP request, so there is no token and no caller identity.

## Decision

- `build_server(store, auth_resource_url=...)` turns on bearer auth. `main()` passes it only for `SUPPORT_TRANSPORT=http`.
- Tokens come from `DevTokenVerifier`, a fixed table standing in for real JWT validation. No authorization server runs. The `issuer_url` is a placeholder.
- Scopes are named `<resource>:<action>`. Every HTTP request needs `accounts:read`. `process_refund` also needs `refunds:write`, and `escalate_ticket` needs `tickets:write`. Both are checked inside the tool.
- Any client allowed to take a risky action must also be allowed to escalate, so it always has a safe way to hand off to a human.
- Without `auth_resource_url`, scope checks are skipped.
- The HTTP server binds to `127.0.0.1` by default.

## Consequences

- **Authorization depends on the transport.** The same refund that HTTP refuses for `dev-readonly` succeeds over stdio. That is acceptable for a local playground, but it is the exact gap that makes transport-only authorization unsafe: any path that skips the middleware skips the check.
- Most tests run in-process and never exercise auth, so auth has its own tests (`tests/test_http_auth.py`) against a real server on a random port.
- Sprint 6 fixes this by passing a caller identity into `SupportStore` and enforcing permissions there, as ADR 0001 already does for refund policy.
