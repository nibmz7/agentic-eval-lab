# agentic-eval-lab

A playground for building a tool-using agent and, more importantly, **evaluating it in ways you can trust**.

The system under test is a customer-support agent that can check balances, issue refunds, and escalate to a human. The tools are exposed over MCP, the agent loop is a LangGraph state machine, and the eval suite runs under pytest with DeepEval.

```
            ┌──────────────────────┐
 user ────▶ │  LangGraph agent     │  planner → action → guardrail → (retry | escalate | respond)
            └──────────┬───────────┘
                       │ MCP (stdio / in-process)
            ┌──────────▼───────────┐
            │  support-mcp server  │  check_balance · process_refund · escalate_ticket
            └──────────┬───────────┘
                       │
            ┌──────────▼───────────┐
            │  SupportStore (SQLite)│  source of truth for policy + end-state assertions
            └──────────────────────┘
```

## Status

| Sprint | Scope | State |
|---|---|---|
| 1 | MCP tool server + backend + tests | ✅ done |
| 2 | LangGraph agent with retries, guardrail, escalation | ⬜ |
| 3 | Eval suite: trajectory, tool-args, end-state, multi-turn | ⬜ |
| 4 | Langfuse tracing + write-up | ⬜ |
| 5 | Adversarial / red-team + evaluator-integrity tests | ⬜ |

See [docs/ROADMAP.md](docs/ROADMAP.md) for each sprint's exercises, and [docs/decisions/](docs/decisions/) for design notes.

## Quick start

```bash
uv sync
uv run pytest                 # backend + MCP contract tests
uv run ruff format            # format (prettier)
uv run ruff check --fix       # lint (eslint)
uv run pyright                # type-check (tsc --noEmit)
uv run support-mcp            # run the MCP server over stdio
npx @modelcontextprotocol/inspector uv run support-mcp   # poke at it in a UI
```

`SUPPORT_DB=./lab.db uv run support-mcp` persists state to a file. By default the DB is in memory and seeded with four users:

| user | status | balance | what it exercises |
|---|---|---|---|
| `u_alice` | active | 120.00 | happy path, insufficient-balance edge |
| `u_bob` | active | 1500.00 | refunds over the 500.00 limit must escalate |
| `u_carol` | suspended | 80.00 | refunds must be refused |
| `u_dave` | active | 0.00 | nothing to refund |

## Notes

- Uses the **mcp 2.x** Python SDK, where `FastMCP` is now `mcp.server.mcpserver.MCPServer`. Most tutorials still show the 1.x import.
- Policy lives in the backend, not in the prompt. The agent can be wrong and the database will still refuse an invalid refund, so evals can check the final state instead of the agent's claims.
