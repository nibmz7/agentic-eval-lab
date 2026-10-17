# 0001: The backend owns policy; evals assert on end state

**Status:** accepted (Sprint 1)

## Context

The brief puts balance validation in a LangGraph "guardrail node". If that were the *only* check, a buggy or prompt-injected agent could issue a bad refund, and the eval would have to infer that from the transcript.

## Decision

- `SupportStore` enforces every refund rule itself (active account, positive amount, ≤ balance, ≤ 500.00 auto-approve limit, idempotency) and records every attempt in `audit_log`.
- The agent's guardrail node still exists. It is defense in depth and gives the agent a chance to explain or escalate *before* hitting a backend error.
- Policy failures cross MCP as `isError` results prefixed with a stable code (`requires_escalation`, `insufficient_balance`, …) so the agent can branch on them.
- `snapshot()` excludes timestamps and the audit log so end states can be compared with `==`.

## Consequences

- The main eval signal is "is the DB in the expected state", which no amount of convincing agent text can fake.
- The audit log separately shows *attempted* bad actions that the backend blocked. An agent that tries a 900.00 refund before escalating ends in a correct state but took a bad trajectory. Trajectory metrics need to catch that.
- Money is stored as integer cents. The MCP layer accepts dollars (number or string) and rejects more than 2 decimal places instead of rounding.
