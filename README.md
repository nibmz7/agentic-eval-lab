# agentic-eval-lab

A playground for building a tool-using agent and, more importantly, **evaluating it in ways you can trust**.

The system under test is a customer-support agent that can check balances, issue refunds, and escalate to a human. The tools are exposed over MCP, the agent loop is a LangGraph state machine, and the eval suite runs under pytest with DeepEval.

```
            ┌──────────────────────┐
 user ────▶ │  LangGraph agent     │  planner → agent → guardrail → tools → (agent | escalate)
            └──────────┬───────────┘
                       │ MCP (stdio / in-process)
            ┌──────────▼───────────┐
            │  support-mcp server  │  tools: check_balance · process_refund · escalate_ticket
            │                      │  resources: policy://refunds · account://{user_id}
            │                      │  prompts: handle_refund_request
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
| 1.5 | Resources, prompts, HTTP transport, bearer-token auth | ✅ done |
| 2 | LangGraph agent with retries, guardrail, escalation | ✅ done |
| 3 | Eval suite: trajectory, tool-args, end-state, multi-turn | ⬜ |
| 4 | Langfuse tracing + write-up | ⬜ |
| 5 | Evaluator-integrity tests | ⬜ |
| 6 | Adversarial, authorization, governance (stretch) | ⬜ |

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

### Over HTTP, with auth

```bash
SUPPORT_TRANSPORT=http uv run support-mcp     # http://127.0.0.1:8000/mcp
```

HTTP requires a bearer token. Two dev tokens are built in:

| token | scopes | read | refund | escalate |
|---|---|---|---|---|
| `dev-agent` | `accounts:read refunds:write tickets:write` | ✅ | ✅ | ✅ |
| `dev-readonly` | `accounts:read` | ✅ | ❌ | ❌ |

In the Inspector, pick the Streamable HTTP transport and set the token under Authentication. Over stdio there is no auth (see [ADR 0002](docs/decisions/0002-auth-at-transport-layer.md)). `SUPPORT_HOST` and `SUPPORT_PORT` override the bind address.

### Data

`SUPPORT_DB=./lab.db uv run support-mcp` persists state to a file. By default the DB is in memory and seeded with four users:

| user | status | balance | what it exercises |
|---|---|---|---|
| `u_alice` | active | 120.00 | happy path, insufficient-balance edge |
| `u_bob` | active | 1500.00 | refunds over the 500.00 limit must escalate |
| `u_carol` | suspended | 80.00 | refunds must be refused |
| `u_dave` | active | 0.00 | nothing to refund |

### Chat with the agent

Needs Claude credentials: `export ANTHROPIC_API_KEY=...`, or `ant auth login`.

```bash
uv run support-agent --customer u_alice
uv run support-agent --customer u_bob --model claude-sonnet-5-5   # cheaper model
```

Every tool call is printed under the reply, including which ones the guardrail blocked. The agent starts its own MCP server over stdio, so it can only reach the database through the server's tools. `SUPPORT_AGENT_MODEL` and `SUPPORT_AGENT_EFFORT` set the defaults (`claude-opus-5-5`, `medium`). How the agent is put together: [ADR 0003](docs/decisions/0003-agent-graph.md).

## Notes

- Uses the **mcp 2.x** Python SDK, where `FastMCP` is now `mcp.server.mcpserver.MCPServer`. Most tutorials still show the 1.x import.
- Policy lives in the backend, not in the prompt. The agent can be wrong and the database will still refuse an invalid refund, so evals can check the final state instead of the agent's claims.
