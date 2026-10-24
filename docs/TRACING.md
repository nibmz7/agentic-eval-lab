# Tracing with Langfuse

A results file tells you *that* a run failed. A trace shows *why*: every step the agent took, what Claude was sent and answered, each tool call and its result, and the judge's reasoning, nested in the order it happened.

Tracing is optional. Without Langfuse keys everything works as before and nothing is sent anywhere.

## Start Langfuse on your machine

You need Docker. The first start downloads a few GB of images and takes a few minutes.

```bash
docker compose -f infra/langfuse/docker-compose.yml up -d
```

Open http://localhost:3000 and sign in as `lab@example.com` / `lab-password`. The project `support-agent` and its API keys already exist. Then, in the terminal you run the agent or evals from:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-lab
export LANGFUSE_SECRET_KEY=sk-lf-lab
export LANGFUSE_BASE_URL=http://localhost:3000
```

These keys are fixed, local-only values from the compose file. Stop Langfuse with `docker compose -f infra/langfuse/docker-compose.yml down` (your traces are kept), or add `-v` to delete them too.

Using Langfuse Cloud instead works the same way: create a project, and use its keys and its URL (for example `https://cloud.langfuse.com`). Your conversations are then stored on Langfuse's servers.

## Trace an eval run

```bash
uv run support-eval run --runs 2 --only refund-over-limit
```

The run prints `Tracing to Langfuse, session eval-<time>.` at the start. Each run of each scenario becomes one trace, and all of them share one Langfuse **session**, so one `support-eval run` is one session. The report ends with the failed runs, each with a link to its trace:

```
failed runs:
  injection-admin-override #3: trajectory: forbidden tools
    trace: http://localhost:3000/project/support-agent/traces/4f2a…
```

If Langfuse isn't running, you'll see `Langfuse keys are set, but Langfuse didn't accept them (...). Not tracing.` and the run carries on untraced.

`judge-check` and `rejudge` are traced too. The judge's calls join the trace of the run they judge.

## Reading a trace

```
eval refund-over-limit #1            the run: customer, what they said, final reply
  support turn                       one customer message and the agent's reply
    LangGraph
      planner
        classify                     Claude call: intent
      agent
        respond                      Claude call: model, tokens, full input and output
      guardrail
        refund check                 the guardrail's verdict on a refund
          check_balance              its balance lookup over MCP
      tools
        escalate_ticket              MCP tool call: arguments and result
      agent
        respond
  judge explains-escalation          the judge on one criterion
    judge call                       its Claude call and reasoning
```

- **Generations** (`classify`, `respond`, `judge call`) show the model, token counts, and exactly what was sent and returned, including thinking blocks.
- **Tools** show arguments and results. A failed call is marked as an error, with its error code (for example `unknown_user`) as the status message.
- **The guardrail** shows `allowed` or the reason it blocked a refund.

## Scores: finding the failures

Each run's results are attached to its trace as scores (1 = pass, 0 = fail):

| Score | Meaning |
|---|---|
| `passed` | The run's verdict. The comment lists the failing checks. |
| `end_state`, `trajectory`, `efficiency`, `escalation`, `per_turn`, `crash` | Each layer that ran. The comment says what failed. |
| `intent` | The planner's label. Reported only, like in the report. |
| `reply: <criterion>` | Each judged criterion, with the judge's score and reason. Doesn't count toward `passed`. |

In Langfuse, open the traces list, filter by the session, and add a filter on the score `passed` = 0 to see only failing runs. Filtering on a layer score (say `trajectory` = 0) groups failures by kind. The scores also make it easy to compare two sessions, for example before and after a prompt change.

## Tracing the chat

`support-agent` is traced too. Each turn is a trace named `chat`, and the conversation is one session, with the customer as the Langfuse user.

## How it's built

- `src/support_lab/tracing.py` is the only module that talks to Langfuse. Everything else calls `tracing.observe(...)`, which does nothing when tracing is off.
- The LangGraph nodes come from Langfuse's LangChain callback handler. LangChain normally runs handlers in a thread pool, so a node's span never became the "current" one inside that node, and the Claude and tool calls inside a node attached to the turn instead. The handler is set to run inline, which fixes the nesting.
- Claude calls are traced by hand in `AnthropicModel` and `ClaudeJudge`, because the agent calls the Anthropic SDK directly rather than through LangChain.
- The Langfuse client gets its own OpenTelemetry tracer provider. The MCP SDK also emits OpenTelemetry spans, through the global provider, and those would otherwise show up as extra traces of their own.
- Costs: Langfuse prices token counts from its own model table. If it shows no cost for a model, the eval report's estimate (`evals/pricing.py`) is still there.

Design notes: [ADR 0008](decisions/0008-tracing.md).
